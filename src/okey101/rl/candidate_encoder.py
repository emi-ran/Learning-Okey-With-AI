"""ID-free, fixed-shape V1 features for variable legal-action candidates."""

from __future__ import annotations

from dataclasses import dataclass

from okey101.engine.actions import (
    Action,
    ActionType,
    AddPair,
    AddToMeld,
    Discard,
    DrawFromStock,
    EndTableActions,
    OpenMelds,
    OpenPairs,
    ReplaceJoker,
    TakePreviousDiscard,
)
from okey101.engine.config import GameConfig
from okey101.engine.melds import MeldKind, MeldTile
from okey101.engine.table import AttachmentSide
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue, build_tile_set

from .action_codec import ActionCandidate, ActionCatalog
from .observation import (
    PlayerObservation,
    VisibleMeld,
    VisibleMeldTile,
    VisibleTile,
)


CANDIDATE_ENCODER_VERSION = 1

ACTION_TYPE_SIZE = 9
TILE_ROLE_SIZE = 54  # 52 normal values + fake Okey + real Okey
VALUE_SIZE = 52
TARGET_KIND_SIZE = 4  # none, run, set, pair area
TARGET_SIDE_SIZE = 4  # none, left, right, set
SCALAR_NAMES = (
    "selected_count",
    "represented_count",
    "represented_score",
    "laid_group_count",
    "laid_run_count",
    "laid_set_count",
    "laid_pair_count",
    "selected_real_okey_count",
    "selected_fake_okey_count",
    "target_tile_count",
    "target_left_usage",
    "target_right_usage",
    "target_set_usage",
    "retrieved_real_okey",
    "nonfinal_playable_discard",
    "real_okey_discard",
)
SCALAR_SIZE = len(SCALAR_NAMES)

# A standard starting hand has at most 22 tiles and must preserve a final
# discard. Pair openings therefore have the largest possible partition:
# floor(21 / 2) == 10 groups.
MAX_LAID_GROUPS = 10
LAID_GROUP_FEATURE_SIZE = 16

CANDIDATE_FEATURE_SIZE = (
    ACTION_TYPE_SIZE
    + TILE_ROLE_SIZE
    + VALUE_SIZE
    + VALUE_SIZE
    + TARGET_KIND_SIZE
    + TARGET_SIDE_SIZE
    + VALUE_SIZE
    + VALUE_SIZE
    + SCALAR_SIZE
    + MAX_LAID_GROUPS * LAID_GROUP_FEATURE_SIZE
    + MAX_LAID_GROUPS
)

_ACTION_INDEX = {
    ActionType.DRAW_FROM_STOCK: 0,
    ActionType.TAKE_PREVIOUS_DISCARD: 1,
    ActionType.OPEN_MELDS: 2,
    ActionType.OPEN_PAIRS: 3,
    ActionType.ADD_TO_MELD: 4,
    ActionType.ADD_PAIR: 5,
    ActionType.REPLACE_JOKER: 6,
    ActionType.END_TABLE_ACTIONS: 7,
    ActionType.DISCARD: 8,
}
_COLOR_INDEX = {color: index for index, color in enumerate(Color)}
_TARGET_KIND_INDEX = {
    "none": 0,
    MeldKind.RUN.value: 1,
    MeldKind.SET.value: 2,
    "pair": 3,
}
_TARGET_SIDE_INDEX = {
    None: 0,
    AttachmentSide.LEFT: 1,
    AttachmentSide.RIGHT: 2,
    AttachmentSide.SET: 3,
}
_CANONICAL_TILES = {tile.id: tile for tile in build_tile_set()}


class CandidateEncodingError(ValueError):
    """Raised when an action cannot be resolved from its public observation."""


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    """One model-facing feature row with no physical or table identifiers."""

    action_type: tuple[float, ...]
    selected_tile_roles: tuple[float, ...]
    represented_values: tuple[float, ...]
    joker_assignments: tuple[float, ...]
    target_kind: tuple[float, ...]
    target_side: tuple[float, ...]
    target_values: tuple[float, ...]
    target_joker_assignments: tuple[float, ...]
    scalars: tuple[float, ...]
    laid_groups: tuple[tuple[float, ...], ...]
    laid_group_mask: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = (
            ("action_type", self.action_type, ACTION_TYPE_SIZE),
            ("selected_tile_roles", self.selected_tile_roles, TILE_ROLE_SIZE),
            ("represented_values", self.represented_values, VALUE_SIZE),
            ("joker_assignments", self.joker_assignments, VALUE_SIZE),
            ("target_kind", self.target_kind, TARGET_KIND_SIZE),
            ("target_side", self.target_side, TARGET_SIDE_SIZE),
            ("target_values", self.target_values, VALUE_SIZE),
            (
                "target_joker_assignments",
                self.target_joker_assignments,
                VALUE_SIZE,
            ),
            ("scalars", self.scalars, SCALAR_SIZE),
            ("laid_groups", self.laid_groups, MAX_LAID_GROUPS),
            ("laid_group_mask", self.laid_group_mask, MAX_LAID_GROUPS),
        )
        for name, values, size in expected:
            if len(values) != size:
                raise ValueError(f"{name} must contain {size} values")
        if any(
            len(group) != LAID_GROUP_FEATURE_SIZE
            for group in self.laid_groups
        ):
            raise ValueError(
                f"each laid group must contain {LAID_GROUP_FEATURE_SIZE} values"
            )

    def as_vector(self) -> tuple[float, ...]:
        """Flatten the documented V1 blocks in a stable order."""

        flat_groups: list[float] = []
        for group in self.laid_groups:
            flat_groups.extend(group)
        return (
            *self.action_type,
            *self.selected_tile_roles,
            *self.represented_values,
            *self.joker_assignments,
            *self.target_kind,
            *self.target_side,
            *self.target_values,
            *self.target_joker_assignments,
            *self.scalars,
            *flat_groups,
            *self.laid_group_mask,
        )


def _one_hot(index: int, size: int) -> tuple[float, ...]:
    values = [0.0] * size
    values[index] = 1.0
    return tuple(values)


def _value_index(value: TileValue) -> int:
    return _COLOR_INDEX[value.color] * 13 + value.number - 1


def _increment_value(values: list[float], value: TileValue) -> None:
    values[_value_index(value)] += 1.0


def _tile_role_index(
    tile: PhysicalTile | VisibleTile,
    okey_value: TileValue,
) -> int:
    if tile.kind is TileKind.FAKE_OKEY:
        return 52
    is_okey = (
        tile.is_real_okey
        if isinstance(tile, VisibleTile)
        else tile.value == okey_value
    )
    if is_okey:
        return 53
    if tile.color is None or tile.number is None:
        raise CandidateEncodingError("normal visible tile has no value")
    return _COLOR_INDEX[tile.color] * 13 + tile.number - 1


def _own_tile(
    observation: PlayerObservation,
    tile_id: int,
) -> PhysicalTile:
    if tile_id not in observation.own_tile_ids:
        raise CandidateEncodingError(
            f"action tile is not visible in the acting hand: {tile_id}"
        )
    try:
        return _CANONICAL_TILES[tile_id]
    except KeyError as error:
        raise CandidateEncodingError(
            f"unknown canonical physical tile id: {tile_id}"
        ) from error


def _visible_meld(
    observation: PlayerObservation,
    meld_id: int,
) -> VisibleMeld:
    for meld in observation.table_melds:
        if meld.meld_id == meld_id:
            return meld
    raise CandidateEncodingError(f"target meld is not publicly visible: {meld_id}")


def _selected_tiles(action: Action) -> tuple[tuple[int, MeldTile | None], ...]:
    if isinstance(action, OpenMelds):
        return tuple(
            (tile.physical_tile.id, tile)
            for meld in action.melds
            for tile in meld.tiles
        )
    if isinstance(action, OpenPairs):
        return tuple(
            (tile.physical_tile.id, tile)
            for pair in action.pairs
            for tile in pair.tiles
        )
    if isinstance(action, AddToMeld):
        return tuple((tile.physical_tile.id, tile) for tile in action.tiles)
    if isinstance(action, AddPair):
        return tuple(
            (tile.physical_tile.id, tile) for tile in action.pair.tiles
        )
    if isinstance(action, ReplaceJoker):
        return ((action.replacement_tile_id, None),)
    if isinstance(action, Discard):
        return ((action.tile_id, None),)
    return ()


def _laid_group_row(
    kind: MeldKind | str,
    tiles: tuple[MeldTile, ...],
    observation: PlayerObservation,
) -> tuple[float, ...]:
    """Compactly identify one resulting public table group.

    Normal tiles are implied by the represented run/set/pair structure. Real
    and fake Okeys need explicit represented-value tokens because their public
    tile token differs from that represented value.
    """

    kind_name = kind.value if isinstance(kind, MeldKind) else kind
    kind_index = {
        MeldKind.RUN.value: 0,
        MeldKind.SET.value: 1,
        "pair": 2,
    }[kind_name]
    represented = tuple(tile.represented_value for tile in tiles)
    color_mask = [0.0] * 4
    for value in represented:
        color_mask[_COLOR_INDEX[value.color]] = 1.0

    real_assignments: list[float] = []
    fake_assignments: list[float] = []
    for meld_tile in tiles:
        physical = _own_tile(
            observation,
            meld_tile.physical_tile.id,
        )
        represented_token = float(_value_index(meld_tile.represented_value) + 1)
        if physical.kind is TileKind.FAKE_OKEY:
            fake_assignments.append(represented_token)
        elif physical.value == observation.okey_value:
            real_assignments.append(represented_token)
    if len(real_assignments) > 2 or len(fake_assignments) > 2:
        raise CandidateEncodingError(
            "a standard group cannot contain more than two real or fake Okeys"
        )
    real_assignments.sort()
    fake_assignments.sort()
    real_assignment_slots = (*real_assignments, *(0.0,) * (2 - len(real_assignments)))
    fake_assignment_slots = (*fake_assignments, *(0.0,) * (2 - len(fake_assignments)))
    numbers = tuple(value.number for value in represented)
    row = (
        *_one_hot(kind_index, 3),
        *color_mask,
        float(min(numbers)),
        float(max(numbers)),
        float(len(tiles)),
        *real_assignment_slots,
        *fake_assignment_slots,
        float(len(real_assignments)),
        float(len(fake_assignments)),
    )
    assert len(row) == LAID_GROUP_FEATURE_SIZE
    return row


def _laid_group_features(
    action: Action,
    observation: PlayerObservation,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    if isinstance(action, OpenMelds):
        rows.extend(
            _laid_group_row(meld.kind, meld.tiles, observation)
            for meld in action.melds
        )
    elif isinstance(action, OpenPairs):
        rows.extend(
            _laid_group_row("pair", pair.tiles, observation)
            for pair in action.pairs
        )
    elif isinstance(action, AddPair):
        rows.append(_laid_group_row("pair", action.pair.tiles, observation))

    if len(rows) > MAX_LAID_GROUPS:
        raise CandidateEncodingError(
            f"candidate has {len(rows)} laid groups; "
            f"V1 capacity is {MAX_LAID_GROUPS}"
        )
    rows.sort()
    active_count = len(rows)
    zero_row = (0.0,) * LAID_GROUP_FEATURE_SIZE
    padded = tuple((*rows, *(zero_row,) * (MAX_LAID_GROUPS - active_count)))
    mask = (1.0,) * active_count + (0.0,) * (MAX_LAID_GROUPS - active_count)
    return padded, mask


def _target_meld_features(
    meld: VisibleMeld,
) -> tuple[list[float], list[float], int]:
    values = [0.0] * VALUE_SIZE
    joker_assignments = [0.0] * VALUE_SIZE
    for tile in meld.tiles:
        _increment_value(values, tile.represented_value)
        if tile.tile.is_real_okey:
            _increment_value(joker_assignments, tile.represented_value)
    return values, joker_assignments, len(meld.tiles)


def _pair_area_features(
    observation: PlayerObservation,
) -> tuple[list[float], list[float], int]:
    values = [0.0] * VALUE_SIZE
    joker_assignments = [0.0] * VALUE_SIZE
    tile_count = 0
    for pair in observation.pair_area:
        for tile in pair.tiles:
            _increment_value(values, tile.represented_value)
            if tile.tile.is_real_okey:
                _increment_value(joker_assignments, tile.represented_value)
            tile_count += 1
    return values, joker_assignments, tile_count


def _current_available_discard(
    observation: PlayerObservation,
) -> VisibleTile:
    if not observation.discard_history:
        raise CandidateEncodingError(
            "take-previous-discard action has no current public discard"
        )
    current = observation.discard_history[-1]
    if current.taken_by_relative is not None:
        raise CandidateEncodingError(
            "current public discard has already been taken"
        )
    return current.tile


def _attachment_usage(
    observation: PlayerObservation,
    meld_id: int,
    side: AttachmentSide,
) -> int:
    return sum(
        usage.count
        for usage in observation.attachment_usage
        if usage.meld_id == meld_id and usage.side is side
    )


def _is_nonfinal_playable_discard(
    tile: PhysicalTile,
    observation: PlayerObservation,
    config: GameConfig,
) -> bool:
    if observation.player_statuses[0].hand_count <= 1:
        return False
    # The engine applies the dedicated Okey-discard rule first and does not
    # double-charge a real Okey as a playable discard.
    if tile.value == observation.okey_value:
        return False

    value = observation.okey_value if tile.kind is TileKind.FAKE_OKEY else tile.value
    assert value is not None
    for meld in observation.table_melds:
        represented = tuple(item.represented_value for item in meld.tiles)
        if meld.kind == MeldKind.RUN.value:
            if value.color != represented[0].color:
                continue
            numbers = tuple(item.number for item in represented)
            side = None
            if value.number == min(numbers) - 1:
                side = AttachmentSide.LEFT
            elif value.number == max(numbers) + 1:
                side = AttachmentSide.RIGHT
            if (
                side is not None
                and _attachment_usage(observation, meld.meld_id, side)
                < config.max_contiguous_attach
            ):
                return True
        elif (
            meld.kind == MeldKind.SET.value
            and len(represented) < 4
            and value.number == represented[0].number
            and value.color not in {item.color for item in represented}
            and _attachment_usage(
                observation,
                meld.meld_id,
                AttachmentSide.SET,
            )
            < config.max_contiguous_attach
        ):
            return True
    return False


def encode_candidate(
    observation: PlayerObservation,
    candidate: ActionCandidate | Action,
    config: GameConfig,
) -> CandidateFeatures:
    """Encode one candidate without exposing physical IDs or table meld IDs."""

    action = candidate.action if isinstance(candidate, ActionCandidate) else candidate
    action_type = _one_hot(_ACTION_INDEX[action.type], ACTION_TYPE_SIZE)
    selected_roles = [0.0] * TILE_ROLE_SIZE
    represented_values = [0.0] * VALUE_SIZE
    joker_assignments = [0.0] * VALUE_SIZE
    target_kind_name = "none"
    target_side = None
    target_values = [0.0] * VALUE_SIZE
    target_joker_assignments = [0.0] * VALUE_SIZE
    target_tile_count = 0
    target_left_usage = 0
    target_right_usage = 0
    target_set_usage = 0
    retrieved_real_okey = 0

    selected_count = 0
    represented_count = 0
    represented_score = 0
    real_okey_count = 0
    fake_okey_count = 0

    if isinstance(action, TakePreviousDiscard):
        visible = _current_available_discard(observation)
        selected_roles[
            _tile_role_index(visible, observation.okey_value)
        ] += 1.0
        selected_count = 1

    for tile_id, meld_tile in _selected_tiles(action):
        tile = _own_tile(observation, tile_id)
        selected_roles[_tile_role_index(tile, observation.okey_value)] += 1.0
        selected_count += 1
        if tile.kind is TileKind.FAKE_OKEY:
            fake_okey_count += 1
        elif tile.value == observation.okey_value:
            real_okey_count += 1
        if meld_tile is not None:
            represented = meld_tile.represented_value
            _increment_value(represented_values, represented)
            represented_count += 1
            represented_score += represented.number
            if tile.value == observation.okey_value:
                _increment_value(joker_assignments, represented)

    laid_group_count = 0
    laid_run_count = 0
    laid_set_count = 0
    laid_pair_count = 0
    if isinstance(action, OpenMelds):
        laid_group_count = len(action.melds)
        laid_run_count = sum(meld.kind is MeldKind.RUN for meld in action.melds)
        laid_set_count = sum(meld.kind is MeldKind.SET for meld in action.melds)
    elif isinstance(action, OpenPairs):
        laid_group_count = len(action.pairs)
        laid_pair_count = len(action.pairs)
    elif isinstance(action, AddPair):
        laid_group_count = 1
        laid_pair_count = 1

    if isinstance(action, (AddToMeld, ReplaceJoker)):
        meld = _visible_meld(observation, action.meld_id)
        target_kind_name = meld.kind
        (
            target_values,
            target_joker_assignments,
            target_tile_count,
        ) = _target_meld_features(meld)
        if isinstance(action, AddToMeld):
            target_side = action.side
        else:
            joker_tile = next(
                (
                    tile
                    for tile in meld.tiles
                    if tile.tile.tile_id == action.joker_tile_id
                ),
                None,
            )
            if joker_tile is None or not joker_tile.tile.is_real_okey:
                raise CandidateEncodingError(
                    "replacement target is not a visible real Okey"
                )
            _increment_value(represented_values, joker_tile.represented_value)
            represented_count += 1
            represented_score += joker_tile.represented_value.number
            retrieved_real_okey = 1

        for usage in observation.attachment_usage:
            if usage.meld_id != action.meld_id:
                continue
            if usage.side is AttachmentSide.LEFT:
                target_left_usage = usage.count
            elif usage.side is AttachmentSide.RIGHT:
                target_right_usage = usage.count
            else:
                target_set_usage = usage.count
    elif isinstance(action, AddPair):
        target_kind_name = "pair"
        (
            target_values,
            target_joker_assignments,
            target_tile_count,
        ) = _pair_area_features(observation)

    discard_is_playable = 0
    discard_is_real_okey = 0
    if isinstance(action, Discard):
        tile = _own_tile(observation, action.tile_id)
        discard_is_playable = int(
            _is_nonfinal_playable_discard(tile, observation, config)
        )
        discard_is_real_okey = int(tile.value == observation.okey_value)

    laid_groups, laid_group_mask = _laid_group_features(action, observation)
    scalars = tuple(
        float(value)
        for value in (
            selected_count,
            represented_count,
            represented_score,
            laid_group_count,
            laid_run_count,
            laid_set_count,
            laid_pair_count,
            real_okey_count,
            fake_okey_count,
            target_tile_count,
            target_left_usage,
            target_right_usage,
            target_set_usage,
            retrieved_real_okey,
            discard_is_playable,
            discard_is_real_okey,
        )
    )
    features = CandidateFeatures(
        action_type=action_type,
        selected_tile_roles=tuple(selected_roles),
        represented_values=tuple(represented_values),
        joker_assignments=tuple(joker_assignments),
        target_kind=_one_hot(
            _TARGET_KIND_INDEX[target_kind_name],
            TARGET_KIND_SIZE,
        ),
        target_side=_one_hot(
            _TARGET_SIDE_INDEX[target_side],
            TARGET_SIDE_SIZE,
        ),
        target_values=tuple(target_values),
        target_joker_assignments=tuple(target_joker_assignments),
        scalars=scalars,
        laid_groups=laid_groups,
        laid_group_mask=laid_group_mask,
    )
    assert len(features.as_vector()) == CANDIDATE_FEATURE_SIZE
    return features


def encode_catalog(
    observation: PlayerObservation,
    catalog: ActionCatalog,
    config: GameConfig,
) -> tuple[CandidateFeatures, ...]:
    """Encode every catalog row while preserving candidate order."""

    return tuple(
        encode_candidate(observation, candidate, config)
        for candidate in catalog.candidates
    )
