"""Deterministic local-RNG deck operations."""

from __future__ import annotations

from collections.abc import Iterable
from random import Random

from .tiles import PhysicalTile, build_tile_set


class Deck:
    """A controlled mutable stack of unique physical tiles."""

    __slots__ = ("_tiles",)

    def __init__(self, tiles: Iterable[PhysicalTile]) -> None:
        self._tiles = list(tiles)
        ids = [tile.id for tile in self._tiles]
        if len(ids) != len(set(ids)):
            raise ValueError("a deck cannot contain duplicate physical tile ids")

    @classmethod
    def standard(cls) -> Deck:
        """Return an unshuffled canonical 106-tile deck."""

        return cls(build_tile_set())

    @classmethod
    def shuffled(cls, seed: int | None = None) -> Deck:
        """Return a shuffled standard deck without touching global RNG state."""

        tiles = list(build_tile_set())
        Random(seed).shuffle(tiles)
        return cls(tiles)

    @property
    def tiles(self) -> tuple[PhysicalTile, ...]:
        return tuple(self._tiles)

    def __len__(self) -> int:
        return len(self._tiles)

    def draw(self) -> PhysicalTile:
        """Draw the next tile from the top of the stack."""

        if not self._tiles:
            raise IndexError("cannot draw from an empty deck")
        return self._tiles.pop()

    def draw_many(self, count: int) -> tuple[PhysicalTile, ...]:
        if count < 0:
            raise ValueError("draw count cannot be negative")
        if count > len(self._tiles):
            raise IndexError("not enough tiles remaining in deck")
        return tuple(self.draw() for _ in range(count))


def build_deck(seed: int | None = None, *, shuffle: bool = True) -> Deck:
    """Convenience constructor used by round setup."""

    return Deck.shuffled(seed) if shuffle else Deck.standard()
