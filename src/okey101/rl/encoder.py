"""Fixed-shape, dependency-free neural features for player observations.

The encoder accepts only :class:`PlayerObservation`, never ``GameState``.  That
API boundary keeps opponent hands and stock identities unavailable to feature
code.  Integer categorical fields are embedding-ready; callers may stack the
tuples with NumPy or Torch in a separate batching layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from okey101.engine.config import GameConfig
from okey101.engine.player import OpenedMode
from okey101.engine.state import DrawSource, TurnPhase
from okey101.engine.table import AttachmentSide
from okey101.engine.tiles import Color, TileKind, TileValue

from .observation import PlayerObservation, VisibleMeldTile, VisibleTile

OBS_SCHEMA_VERSION = 1
MAX_PLAYERS = 4
MAX_TABLE_GROUPS = 53
MAX_TABLE_TILES = 106
MAX_RECENT_DISCARD_EVENTS = 64
HAND_VALUE_COUNT = 53

GLOBAL_CONTINUOUS_NAMES = (
    "turn_number",
    "round_id",
    "stock_count",
    "progressive_series_threshold",
    "progressive_pair_threshold",
    "table_tile_count",
    "discard_event_count",
    "self_score",
    "self_immediate_penalty",
)
GLOBAL_CATEGORICAL_NAMES = (
    "phase",
    "current_player_relative",
    "starting_player_relative",
    "indicator",
    "okey_value",
    "draw_source",
)
TURN_FLAG_NAMES = (
    "must_use_taken_discard",
    "opened_this_turn",
    "stock_exhausted_after_draw",
)

RULE_CONTINUOUS_NAMES = (
    "player_count",
    "initial_hand_size",
    "opening_min_score",
    "opening_min_pairs",
    "max_contiguous_attach",
    "rounds",
    "scoring.normal_finish_reward",
    "scoring.same_turn_open_finish_reward",
    "scoring.same_turn_open_okey_finish_reward",
    "scoring.okey_finish_reward",
    "scoring.pair_finish_reward",
    "scoring.elden_finish_reward",
    "scoring.elden_okey_finish_reward",
    "scoring.pair_okey_finish_reward",
    "scoring.unopened_end_penalty",
    "scoring.pair_remaining_multiplier",
    "scoring.same_turn_open_finish_opponent_multiplier",
    "scoring.okey_finish_opponent_multiplier",
    "scoring.elden_finish_opponent_multiplier",
    "scoring.pair_finish_opponent_multiplier",
    "scoring.playable_discard_penalty",
    "scoring.normal_okey_discard_penalty",
    "scoring.opened_player_okey_in_hand_surcharge",
)
RULE_FLAG_NAMES = (
    "starting_player_extra_tile",
    "progressive_opening",
    "require_final_discard",
    "void_round_counts_toward_match",
    "scoring.multiply_okey_in_hand_surcharge_by_pair",
    "scoring.multiply_okey_in_hand_surcharge_by_finish",
)

_COLORS = tuple(Color)
_COLOR_INDEX = {color: index for index, color in enumerate(_COLORS)}
_MODE_TOKEN = {
    OpenedMode.NONE: 1,
    OpenedMode.SERIES: 2,
    OpenedMode.PAIRS: 3,
}
_PHASE_TOKEN = {
    TurnPhase.DRAW_DECISION: 1,
    TurnPhase.TABLE_ACTIONS: 2,
    TurnPhase.DISCARD: 3,
    TurnPhase.TERMINAL: 4,
}
_DRAW_SOURCE_TOKEN = {
    None: 0,
    DrawSource.DEAL: 1,
    DrawSource.STOCK: 2,
    DrawSource.PREVIOUS_DISCARD: 3,
}
_GROUP_KIND_TOKEN = {"run": 1, "set": 2, "pair": 3}
_SIDE_INDEX = {
    AttachmentSide.LEFT: 0,
    AttachmentSide.RIGHT: 1,
    AttachmentSide.SET: 2,
}


@dataclass(frozen=True, slots=True)
class EncodedObservationV1:
    """Versioned fixed-shape feature payload made only of Python scalars."""

    schema_version: int
    hand_counts: tuple[float, ...]
    global_continuous: tuple[float, ...]
    global_categorical: tuple[int, ...]
    turn_flags: tuple[int, ...]
    taken_discard_token: int
    rule_continuous: tuple[float, ...]
    rule_flags: tuple[int, ...]
    player_modes: tuple[int, ...]
    player_continuous: tuple[tuple[float, ...], ...]
    player_mask: tuple[int, ...]
    table_group_kinds: tuple[int, ...]
    table_group_sizes: tuple[float, ...]
    table_group_attachment_usage: tuple[tuple[float, ...], ...]
    table_group_mask: tuple[int, ...]
    table_tile_tokens: tuple[int, ...]
    table_represented_tokens: tuple[int, ...]
    table_group_slots: tuple[int, ...]
    table_positions: tuple[int, ...]
    table_is_real_okey: tuple[int, ...]
    table_tile_mask: tuple[int, ...]
    discarded_value_counts: tuple[tuple[float, ...], ...]
    taken_value_counts: tuple[tuple[float, ...], ...]
    recent_discard_tile_tokens: tuple[int, ...]
    recent_discarder_tokens: tuple[int, ...]
    recent_taker_tokens: tuple[int, ...]
    recent_discard_taken: tuple[int, ...]
    recent_discard_ages: tuple[float, ...]
    recent_discard_mask: tuple[int, ...]
    history_truncated: int


@dataclass(frozen=True, slots=True)
class _CanonicalGroup:
    kind_token: int
    tiles: tuple[VisibleMeldTile, ...]
    attachment_usage: tuple[int, int, int]


def value_token(value: TileValue) -> int:
    """Map one normal value to the stable 1..52 vocabulary."""

    return 1 + _COLOR_INDEX[value.color] * 13 + value.number - 1


def tile_token(tile: VisibleTile) -> int:
    """Map one visible physical tile to value identity, discarding copy IDs."""

    if tile.kind is TileKind.FAKE_OKEY:
        return 53
    if tile.color is None or tile.number is None:
        raise ValueError("normal visible tile requires color and number")
    return value_token(TileValue(tile.color, tile.number))


def _pad(values: list[int] | list[float], length: int, fill: int | float):
    if len(values) > length:
        raise ValueError(f"feature length {len(values)} exceeds fixed size {length}")
    return tuple((*values, *(fill for _ in range(length - len(values)))))


def _canonical_groups(observation: PlayerObservation) -> tuple[_CanonicalGroup, ...]:
    usage_by_meld: dict[int, list[int]] = {}
    for usage in observation.attachment_usage:
        counts = usage_by_meld.setdefault(usage.meld_id, [0, 0, 0])
        counts[_SIDE_INDEX[usage.side]] += usage.count

    groups: list[tuple[tuple[object, ...], _CanonicalGroup]] = []
    observed_melds: set[int] = set()
    for meld in observation.table_melds:
        try:
            kind_token = _GROUP_KIND_TOKEN[meld.kind]
        except KeyError as error:
            raise ValueError(f"unsupported visible meld kind: {meld.kind!r}") from error
        observed_melds.add(meld.meld_id)
        usage = tuple(usage_by_meld.get(meld.meld_id, (0, 0, 0)))
        group = _CanonicalGroup(kind_token, meld.tiles, usage)
        key = (
            kind_token,
            tuple(value_token(tile.represented_value) for tile in meld.tiles),
            tuple(tile_token(tile.tile) for tile in meld.tiles),
            usage,
        )
        groups.append((key, group))

    unknown_usage = set(usage_by_meld) - observed_melds
    if unknown_usage:
        raise ValueError(
            f"attachment usage references unknown table melds: {sorted(unknown_usage)}"
        )

    for pair in observation.pair_area:
        tiles = pair.tiles
        group = _CanonicalGroup(_GROUP_KIND_TOKEN["pair"], tiles, (0, 0, 0))
        key = (
            _GROUP_KIND_TOKEN["pair"],
            tuple(value_token(tile.represented_value) for tile in tiles),
            tuple(tile_token(tile.tile) for tile in tiles),
            (0, 0, 0),
        )
        groups.append((key, group))

    groups.sort(key=lambda item: item[0])
    return tuple(group for _key, group in groups)


def _rule_features(config: GameConfig) -> tuple[tuple[float, ...], tuple[int, ...]]:
    scoring = config.scoring
    continuous = (
        config.player_count / 4.0,
        config.initial_hand_size / 21.0,
        config.opening_min_score / 101.0,
        config.opening_min_pairs / 5.0,
        config.max_contiguous_attach / 2.0,
        config.rounds / 100.0,
        scoring.normal_finish_reward / 101.0,
        scoring.same_turn_open_finish_reward / 101.0,
        scoring.same_turn_open_okey_finish_reward / 101.0,
        scoring.okey_finish_reward / 101.0,
        scoring.pair_finish_reward / 101.0,
        scoring.elden_finish_reward / 101.0,
        scoring.elden_okey_finish_reward / 101.0,
        scoring.pair_okey_finish_reward / 101.0,
        scoring.unopened_end_penalty / 101.0,
        float(scoring.pair_remaining_multiplier),
        float(scoring.same_turn_open_finish_opponent_multiplier),
        float(scoring.okey_finish_opponent_multiplier),
        float(scoring.elden_finish_opponent_multiplier),
        float(scoring.pair_finish_opponent_multiplier),
        scoring.playable_discard_penalty / 101.0,
        scoring.normal_okey_discard_penalty / 101.0,
        scoring.opened_player_okey_in_hand_surcharge / 101.0,
    )
    flags = (
        int(config.starting_player_extra_tile),
        int(config.progressive_opening),
        int(config.require_final_discard),
        int(config.void_round_counts_toward_match),
        int(scoring.multiply_okey_in_hand_surcharge_by_pair),
        int(scoring.multiply_okey_in_hand_surcharge_by_finish),
    )
    return continuous, flags


def encode_observation(
    observation: PlayerObservation,
    config: GameConfig,
) -> EncodedObservationV1:
    """Encode one hidden-information-safe observation into schema V1."""

    player_count = len(observation.player_statuses)
    if player_count != config.player_count:
        raise ValueError("observation player count does not match GameConfig")
    if not 2 <= player_count <= MAX_PLAYERS:
        raise ValueError(f"encoder V1 supports 2..{MAX_PLAYERS} players")
    if len(observation.own_normal_counts) != 52:
        raise ValueError("own_normal_counts must contain 52 values")
    if tuple(status.relative_seat for status in observation.player_statuses) != tuple(
        range(player_count)
    ):
        raise ValueError("player statuses must be in contiguous relative-seat order")

    hand_counts = tuple(
        count / 2.0
        for count in (*observation.own_normal_counts, observation.own_fake_okey_count)
    )
    self_status = observation.player_statuses[0]
    groups = _canonical_groups(observation)
    if len(groups) > MAX_TABLE_GROUPS:
        raise ValueError("public table exceeds the V1 group capacity")
    table_tile_count = sum(len(group.tiles) for group in groups)
    if table_tile_count > MAX_TABLE_TILES:
        raise ValueError("public table exceeds the V1 tile capacity")

    global_continuous = (
        observation.turn_number / 106.0,
        observation.round_id / max(config.rounds, 1),
        observation.stock_count / 106.0,
        observation.progressive_series_threshold / 101.0,
        observation.progressive_pair_threshold / 5.0,
        table_tile_count / 106.0,
        len(observation.discard_history) / 106.0,
        self_status.score / 101.0,
        self_status.immediate_penalty / 101.0,
    )
    global_categorical = (
        _PHASE_TOKEN[observation.phase],
        observation.current_player_relative + 1,
        observation.starting_player_relative + 1,
        tile_token(observation.indicator),
        value_token(observation.okey_value),
        _DRAW_SOURCE_TOKEN[observation.draw_source],
    )
    turn_flags = (
        int(observation.must_use_taken_discard),
        int(observation.opened_this_turn),
        int(observation.stock_exhausted_after_draw),
    )
    rule_continuous, rule_flags = _rule_features(config)

    player_modes = [_MODE_TOKEN[status.opened_mode] for status in observation.player_statuses]
    player_continuous = [
        (
            status.hand_count / max(config.starter_hand_size, 1),
            (status.score - self_status.score) / 101.0,
            status.immediate_penalty / 101.0,
        )
        for status in observation.player_statuses
    ]

    group_kinds: list[int] = []
    group_sizes: list[float] = []
    group_usage: list[tuple[float, ...]] = []
    table_tokens: list[int] = []
    represented_tokens: list[int] = []
    group_slots: list[int] = []
    positions: list[int] = []
    real_okey_flags: list[int] = []
    usage_scale = max(config.max_contiguous_attach, 1)
    for group_slot, group in enumerate(groups, start=1):
        group_kinds.append(group.kind_token)
        group_sizes.append(len(group.tiles) / 13.0)
        group_usage.append(
            tuple(count / usage_scale for count in group.attachment_usage)
        )
        for position, meld_tile in enumerate(group.tiles, start=1):
            table_tokens.append(tile_token(meld_tile.tile))
            represented_tokens.append(value_token(meld_tile.represented_value))
            group_slots.append(group_slot)
            positions.append(position)
            real_okey_flags.append(int(meld_tile.tile.is_real_okey))

    discarded_counts = [[0.0] * HAND_VALUE_COUNT for _ in range(MAX_PLAYERS)]
    taken_counts = [[0.0] * HAND_VALUE_COUNT for _ in range(MAX_PLAYERS)]
    for record in observation.discard_history:
        token_index = tile_token(record.tile) - 1
        discarded_counts[record.player_relative][token_index] += 0.5
        if record.taken_by_relative is not None:
            taken_counts[record.taken_by_relative][token_index] += 0.5

    recent = observation.discard_history[-MAX_RECENT_DISCARD_EVENTS:]
    recent_tokens = [tile_token(record.tile) for record in recent]
    recent_discarders = [record.player_relative + 1 for record in recent]
    recent_takers = [
        0 if record.taken_by_relative is None else record.taken_by_relative + 1
        for record in recent
    ]
    recent_taken = [int(record.taken_by_relative is not None) for record in recent]
    recent_ages = [
        (observation.turn_number - record.turn_number) / 106.0
        for record in recent
    ]

    encoded = EncodedObservationV1(
        schema_version=OBS_SCHEMA_VERSION,
        hand_counts=hand_counts,
        global_continuous=global_continuous,
        global_categorical=global_categorical,
        turn_flags=turn_flags,
        taken_discard_token=(
            0
            if observation.taken_discard is None
            else tile_token(observation.taken_discard)
        ),
        rule_continuous=rule_continuous,
        rule_flags=rule_flags,
        player_modes=_pad(player_modes, MAX_PLAYERS, 0),
        player_continuous=tuple(
            (*player_continuous, *((0.0, 0.0, 0.0),) * (MAX_PLAYERS - player_count))
        ),
        player_mask=_pad([1] * player_count, MAX_PLAYERS, 0),
        table_group_kinds=_pad(group_kinds, MAX_TABLE_GROUPS, 0),
        table_group_sizes=_pad(group_sizes, MAX_TABLE_GROUPS, 0.0),
        table_group_attachment_usage=tuple(
            (
                *group_usage,
                *((0.0, 0.0, 0.0),) * (MAX_TABLE_GROUPS - len(group_usage)),
            )
        ),
        table_group_mask=_pad([1] * len(groups), MAX_TABLE_GROUPS, 0),
        table_tile_tokens=_pad(table_tokens, MAX_TABLE_TILES, 0),
        table_represented_tokens=_pad(represented_tokens, MAX_TABLE_TILES, 0),
        table_group_slots=_pad(group_slots, MAX_TABLE_TILES, 0),
        table_positions=_pad(positions, MAX_TABLE_TILES, 0),
        table_is_real_okey=_pad(real_okey_flags, MAX_TABLE_TILES, 0),
        table_tile_mask=_pad([1] * table_tile_count, MAX_TABLE_TILES, 0),
        discarded_value_counts=tuple(tuple(row) for row in discarded_counts),
        taken_value_counts=tuple(tuple(row) for row in taken_counts),
        recent_discard_tile_tokens=_pad(
            recent_tokens, MAX_RECENT_DISCARD_EVENTS, 0
        ),
        recent_discarder_tokens=_pad(
            recent_discarders, MAX_RECENT_DISCARD_EVENTS, 0
        ),
        recent_taker_tokens=_pad(recent_takers, MAX_RECENT_DISCARD_EVENTS, 0),
        recent_discard_taken=_pad(recent_taken, MAX_RECENT_DISCARD_EVENTS, 0),
        recent_discard_ages=_pad(recent_ages, MAX_RECENT_DISCARD_EVENTS, 0.0),
        recent_discard_mask=_pad(
            [1] * len(recent), MAX_RECENT_DISCARD_EVENTS, 0
        ),
        history_truncated=int(
            len(observation.discard_history) > MAX_RECENT_DISCARD_EVENTS
        ),
    )
    validate_encoding(encoded)
    return encoded


def validate_encoding(encoded: EncodedObservationV1) -> None:
    """Raise ``ValueError`` when a payload violates the V1 shape contract."""

    expected_lengths = (
        ("hand_counts", encoded.hand_counts, HAND_VALUE_COUNT),
        ("global_continuous", encoded.global_continuous, len(GLOBAL_CONTINUOUS_NAMES)),
        (
            "global_categorical",
            encoded.global_categorical,
            len(GLOBAL_CATEGORICAL_NAMES),
        ),
        ("turn_flags", encoded.turn_flags, len(TURN_FLAG_NAMES)),
        ("rule_continuous", encoded.rule_continuous, len(RULE_CONTINUOUS_NAMES)),
        ("rule_flags", encoded.rule_flags, len(RULE_FLAG_NAMES)),
        ("player_modes", encoded.player_modes, MAX_PLAYERS),
        ("player_continuous", encoded.player_continuous, MAX_PLAYERS),
        ("player_mask", encoded.player_mask, MAX_PLAYERS),
        ("table_group_kinds", encoded.table_group_kinds, MAX_TABLE_GROUPS),
        ("table_group_sizes", encoded.table_group_sizes, MAX_TABLE_GROUPS),
        (
            "table_group_attachment_usage",
            encoded.table_group_attachment_usage,
            MAX_TABLE_GROUPS,
        ),
        ("table_group_mask", encoded.table_group_mask, MAX_TABLE_GROUPS),
        ("table_tile_tokens", encoded.table_tile_tokens, MAX_TABLE_TILES),
        (
            "table_represented_tokens",
            encoded.table_represented_tokens,
            MAX_TABLE_TILES,
        ),
        ("table_group_slots", encoded.table_group_slots, MAX_TABLE_TILES),
        ("table_positions", encoded.table_positions, MAX_TABLE_TILES),
        ("table_is_real_okey", encoded.table_is_real_okey, MAX_TABLE_TILES),
        ("table_tile_mask", encoded.table_tile_mask, MAX_TABLE_TILES),
        ("discarded_value_counts", encoded.discarded_value_counts, MAX_PLAYERS),
        ("taken_value_counts", encoded.taken_value_counts, MAX_PLAYERS),
        (
            "recent_discard_tile_tokens",
            encoded.recent_discard_tile_tokens,
            MAX_RECENT_DISCARD_EVENTS,
        ),
        (
            "recent_discarder_tokens",
            encoded.recent_discarder_tokens,
            MAX_RECENT_DISCARD_EVENTS,
        ),
        (
            "recent_taker_tokens",
            encoded.recent_taker_tokens,
            MAX_RECENT_DISCARD_EVENTS,
        ),
        (
            "recent_discard_taken",
            encoded.recent_discard_taken,
            MAX_RECENT_DISCARD_EVENTS,
        ),
        (
            "recent_discard_ages",
            encoded.recent_discard_ages,
            MAX_RECENT_DISCARD_EVENTS,
        ),
        (
            "recent_discard_mask",
            encoded.recent_discard_mask,
            MAX_RECENT_DISCARD_EVENTS,
        ),
    )
    for name, values, expected in expected_lengths:
        if len(values) != expected:
            raise ValueError(f"{name} must have length {expected}")
    if encoded.schema_version != OBS_SCHEMA_VERSION:
        raise ValueError(f"unsupported observation schema {encoded.schema_version}")
    if encoded.history_truncated not in (0, 1):
        raise ValueError("history_truncated must be binary")
    if any(len(row) != 3 for row in encoded.player_continuous):
        raise ValueError("each player_continuous row must have length 3")
    if any(len(row) != 3 for row in encoded.table_group_attachment_usage):
        raise ValueError("each table group usage row must have length 3")
    if any(len(row) != HAND_VALUE_COUNT for row in encoded.discarded_value_counts):
        raise ValueError("each discarded count row must have length 53")
    if any(len(row) != HAND_VALUE_COUNT for row in encoded.taken_value_counts):
        raise ValueError("each taken count row must have length 53")
