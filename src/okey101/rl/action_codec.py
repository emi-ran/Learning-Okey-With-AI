"""Stable, lossless state-local IDs for engine-generated legal actions."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NewType, TypeAlias

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
from okey101.engine.config import RulesConfig
from okey101.engine.legal_actions import get_legal_actions
from okey101.engine.melds import Meld, MeldKind, MeldTile
from okey101.engine.pairs import Pair
from okey101.engine.state import GameState
from okey101.engine.table import AttachmentSide
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue


ACTION_CODEC_VERSION = 1

CandidateId = NewType("CandidateId", int)
ActionKey: TypeAlias = tuple[object, ...]

_ACTION_RANK = {
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
_COLOR_RANK = {color: index for index, color in enumerate(Color)}
_KIND_RANK = {
    TileKind.NORMAL: 0,
    TileKind.FAKE_OKEY: 1,
}
_MELD_RANK = {
    MeldKind.RUN: 0,
    MeldKind.SET: 1,
}
_SIDE_RANK = {
    AttachmentSide.LEFT: 0,
    AttachmentSide.RIGHT: 1,
    AttachmentSide.SET: 2,
}


class DuplicateActionKeyError(ValueError):
    """Raised when a legal-action source emits the same canonical action twice."""


class UnknownActionError(KeyError):
    """Raised when an action is not present in a catalog."""


def _value_key(value: TileValue) -> tuple[int, int]:
    return (_COLOR_RANK[value.color], value.number)


def _physical_tile_key(
    tile: PhysicalTile,
) -> tuple[int, int, int, int]:
    return (
        tile.id,
        _KIND_RANK[tile.kind],
        -1 if tile.color is None else _COLOR_RANK[tile.color],
        -1 if tile.number is None else tile.number,
    )


def _meld_tile_key(
    tile: MeldTile,
) -> tuple[tuple[int, int, int, int], int, int]:
    color, number = _value_key(tile.represented_value)
    return (_physical_tile_key(tile.physical_tile), color, number)


def _meld_key(meld: Meld) -> tuple[object, ...]:
    return (
        _MELD_RANK[meld.kind],
        tuple(_meld_tile_key(tile) for tile in meld.tiles),
    )


def _pair_key(pair: Pair) -> tuple[object, ...]:
    return (
        _value_key(pair.represented_value),
        tuple(sorted(_meld_tile_key(tile) for tile in pair.tiles)),
    )


def canonical_action_key(action: Action) -> ActionKey:
    """Return a fully comparable key containing every strategic action choice.

    Physical tile IDs are retained, so equal-valued physical copies remain
    distinct candidates. Order-independent meld and pair groups are sorted to
    avoid assigning multiple IDs to equivalent group orderings.
    """

    rank = _ACTION_RANK[action.type]
    if isinstance(
        action,
        (DrawFromStock, TakePreviousDiscard, EndTableActions),
    ):
        return (rank,)
    if isinstance(action, OpenMelds):
        return (rank, tuple(sorted(_meld_key(meld) for meld in action.melds)))
    if isinstance(action, OpenPairs):
        return (rank, tuple(sorted(_pair_key(pair) for pair in action.pairs)))
    if isinstance(action, AddToMeld):
        return (
            rank,
            action.meld_id,
            _SIDE_RANK[action.side],
            tuple(_meld_tile_key(tile) for tile in action.tiles),
        )
    if isinstance(action, AddPair):
        return (rank, _pair_key(action.pair))
    if isinstance(action, ReplaceJoker):
        return (
            rank,
            action.meld_id,
            action.joker_tile_id,
            action.replacement_tile_id,
        )
    if isinstance(action, Discard):
        return (rank, action.tile_id)
    raise TypeError(f"Unsupported action type: {type(action).__name__}")


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """One legal action and its stable ID inside a single state catalog."""

    candidate_id: CandidateId
    action: Action
    key: ActionKey
    action_type: ActionType


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    """Canonical legal actions for one ``GameState`` and ``RulesConfig``.

    Candidate IDs are deliberately state-local. A catalog must be rebuilt
    after every environment step.
    """

    candidates: tuple[ActionCandidate, ...]

    def __post_init__(self) -> None:
        expected_ids = tuple(range(len(self.candidates)))
        actual_ids = tuple(int(candidate.candidate_id) for candidate in self.candidates)
        if actual_ids != expected_ids:
            raise ValueError("candidate IDs must be contiguous from zero")
        keys = tuple(candidate.key for candidate in self.candidates)
        if any(left >= right for left, right in zip(keys, keys[1:])):
            raise ValueError("candidate keys must be unique and strictly sorted")

    def __len__(self) -> int:
        return len(self.candidates)

    def decode(self, candidate_id: int) -> Action:
        """Return the action for a state-local candidate ID."""

        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise TypeError("candidate_id must be an integer")
        if not 0 <= candidate_id < len(self.candidates):
            raise IndexError(f"candidate_id outside catalog: {candidate_id}")
        return self.candidates[candidate_id].action

    def encode(self, action: Action) -> CandidateId:
        """Return the state-local ID of ``action`` or raise if it is absent."""

        key = canonical_action_key(action)
        keys = tuple(candidate.key for candidate in self.candidates)
        index = bisect_left(keys, key)
        if index >= len(keys) or keys[index] != key:
            raise UnknownActionError("action is not present in this catalog")
        return CandidateId(index)


def catalog_from_actions(actions: Sequence[Action]) -> ActionCatalog:
    """Build a deterministic catalog without dropping any physical candidate."""

    keyed = sorted(
        ((canonical_action_key(action), action) for action in actions),
        key=lambda item: item[0],
    )
    for (left_key, _left), (right_key, _right) in zip(keyed, keyed[1:]):
        if left_key == right_key:
            raise DuplicateActionKeyError(
                f"duplicate canonical legal action key: {left_key!r}"
            )
    return ActionCatalog(
        tuple(
            ActionCandidate(
                candidate_id=CandidateId(index),
                action=action,
                key=key,
                action_type=action.type,
            )
            for index, (key, action) in enumerate(keyed)
        )
    )


def build_action_catalog(
    state: GameState,
    config: RulesConfig,
) -> ActionCatalog:
    """Enumerate and canonically identify every legal engine action."""

    return catalog_from_actions(get_legal_actions(state, config))
