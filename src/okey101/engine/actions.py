from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from .melds import Meld, MeldTile
from .pairs import Pair
from .table import AttachmentSide


class ActionType(str, Enum):
    DRAW_FROM_STOCK = "draw_from_stock"
    TAKE_PREVIOUS_DISCARD = "take_previous_discard"
    OPEN_MELDS = "open_melds"
    OPEN_PAIRS = "open_pairs"
    ADD_TO_MELD = "add_to_meld"
    ADD_PAIR = "add_pair"
    REPLACE_JOKER = "replace_joker"
    END_TABLE_ACTIONS = "end_table_actions"
    DISCARD = "discard"


@dataclass(frozen=True, slots=True)
class DrawFromStock:
    type: ActionType = ActionType.DRAW_FROM_STOCK


@dataclass(frozen=True, slots=True)
class TakePreviousDiscard:
    type: ActionType = ActionType.TAKE_PREVIOUS_DISCARD


@dataclass(frozen=True, slots=True)
class OpenMelds:
    melds: tuple[Meld, ...]
    type: ActionType = ActionType.OPEN_MELDS


@dataclass(frozen=True, slots=True)
class OpenPairs:
    pairs: tuple[Pair, ...]
    type: ActionType = ActionType.OPEN_PAIRS


@dataclass(frozen=True, slots=True)
class AddToMeld:
    meld_id: int
    tiles: tuple[MeldTile, ...]
    side: AttachmentSide
    type: ActionType = ActionType.ADD_TO_MELD


@dataclass(frozen=True, slots=True)
class AddPair:
    pair: Pair
    type: ActionType = ActionType.ADD_PAIR


@dataclass(frozen=True, slots=True)
class ReplaceJoker:
    meld_id: int
    joker_tile_id: int
    replacement_tile_id: int
    type: ActionType = ActionType.REPLACE_JOKER


@dataclass(frozen=True, slots=True)
class EndTableActions:
    type: ActionType = ActionType.END_TABLE_ACTIONS


@dataclass(frozen=True, slots=True)
class Discard:
    tile_id: int
    type: ActionType = ActionType.DISCARD


Action: TypeAlias = (
    DrawFromStock
    | TakePreviousDiscard
    | OpenMelds
    | OpenPairs
    | AddToMeld
    | AddPair
    | ReplaceJoker
    | EndTableActions
    | Discard
)
