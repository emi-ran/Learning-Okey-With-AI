"""Pair validation with explicit wildcard representation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .joker import effective_value, is_real_okey
from .melds import MeldTile
from .tiles import PhysicalTile, TileValue


@dataclass(frozen=True, slots=True)
class Pair:
    tiles: tuple[MeldTile, MeldTile]

    @property
    def represented_value(self) -> TileValue:
        return self.tiles[0].represented_value

    @property
    def physical_tiles(self) -> tuple[PhysicalTile, PhysicalTile]:
        return (self.tiles[0].physical_tile, self.tiles[1].physical_tile)


def build_pair(
    tiles: Iterable[PhysicalTile],
    okey_value: TileValue,
    *,
    represented_value: TileValue | None = None,
) -> Pair:
    """Build a legal pair; two real Okeys default to the round's Okey value."""

    physical_tiles = tuple(tiles)
    if len(physical_tiles) != 2:
        raise ValueError("a pair must contain exactly two physical tiles")
    if physical_tiles[0].id == physical_tiles[1].id:
        raise ValueError("a physical tile cannot be used twice")

    fixed_values = [
        value
        for tile in physical_tiles
        if (value := effective_value(tile, okey_value)) is not None
    ]
    if fixed_values and any(value != fixed_values[0] for value in fixed_values[1:]):
        raise ValueError("unrelated normal tiles do not form a pair")
    target = represented_value or (fixed_values[0] if fixed_values else okey_value)
    if fixed_values and target != fixed_values[0]:
        raise ValueError("represented value conflicts with a fixed tile")

    assigned = tuple(
        MeldTile(tile, target) for tile in sorted(physical_tiles, key=lambda item: item.id)
    )
    pair = Pair((assigned[0], assigned[1]))
    if not validate_pair(pair, okey_value):
        raise ValueError("tiles do not form a legal pair")
    return pair


def validate_pair(pair: Pair, okey_value: TileValue) -> bool:
    """Validate equality, physical uniqueness and both Okey assignments."""

    if len(pair.tiles) != 2:
        return False
    first, second = pair.tiles
    if first.physical_tile.id == second.physical_tile.id:
        return False
    if first.represented_value != second.represented_value:
        return False
    for pair_tile in pair.tiles:
        if not is_real_okey(pair_tile.physical_tile, okey_value):
            if effective_value(pair_tile.physical_tile, okey_value) != pair_tile.represented_value:
                return False
    return True
