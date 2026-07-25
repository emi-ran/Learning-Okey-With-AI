from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable

from .tiles import PhysicalTile


class OpenedMode(str, Enum):
    NONE = "none"
    SERIES = "series"
    PAIRS = "pairs"


@dataclass(frozen=True, slots=True)
class PlayerState:
    hand: tuple[PhysicalTile, ...] = ()
    opened_mode: OpenedMode = OpenedMode.NONE
    opening_turn: int | None = None
    immediate_penalty: int = 0
    score: int = 0

    def add_tiles(self, tiles: Iterable[PhysicalTile]) -> PlayerState:
        return replace(self, hand=(*self.hand, *tiles))

    def remove_tiles(self, tile_ids: Iterable[int]) -> PlayerState:
        ids = tuple(tile_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("The same physical tile cannot be removed twice")

        requested = set(ids)
        hand_ids = {tile.id for tile in self.hand}
        missing = requested - hand_ids
        if missing:
            raise ValueError(f"Tiles are not in the player's hand: {sorted(missing)}")

        return replace(self, hand=tuple(tile for tile in self.hand if tile.id not in requested))

    def tile(self, tile_id: int) -> PhysicalTile:
        for tile in self.hand:
            if tile.id == tile_id:
                return tile
        raise ValueError(f"Tile {tile_id} is not in the player's hand")

    def open(self, mode: OpenedMode, turn_number: int) -> PlayerState:
        if self.opened_mode is not OpenedMode.NONE:
            raise ValueError("Player has already opened")
        if mode is OpenedMode.NONE:
            raise ValueError("Opening mode cannot be NONE")
        return replace(self, opened_mode=mode, opening_turn=turn_number)

    def add_immediate_penalty(self, amount: int) -> PlayerState:
        if amount < 0:
            raise ValueError("Immediate penalty cannot be negative")
        return replace(self, immediate_penalty=self.immediate_penalty + amount)
