"""Stable keys for order-independent solver results."""

from __future__ import annotations

from collections.abc import Iterable

from okey101.engine.melds import Meld
from okey101.engine.pairs import Pair
from okey101.engine.tiles import Color, TileValue


_COLOR_INDEX = {color: index for index, color in enumerate(Color)}


def value_key(value: TileValue) -> tuple[int, int]:
    return (_COLOR_INDEX[value.color], value.number)


def meld_key(meld: Meld) -> tuple[object, ...]:
    return (
        meld.kind.value,
        tuple(
            (
                meld_tile.physical_tile.id,
                *value_key(meld_tile.represented_value),
            )
            for meld_tile in meld.tiles
        ),
    )


def pair_key(pair: Pair) -> tuple[object, ...]:
    return (
        value_key(pair.represented_value),
        tuple(sorted(tile.physical_tile.id for tile in pair.tiles)),
    )


def meld_group_key(melds: Iterable[Meld]) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted((meld_key(meld) for meld in melds)))


def pair_group_key(pairs: Iterable[Pair]) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted((pair_key(pair) for pair in pairs)))


def physical_ids(items: Iterable[Meld | Pair]) -> tuple[int, ...]:
    ids: list[int] = []
    for item in items:
        ids.extend(tile.physical_tile.id for tile in item.tiles)
    return tuple(sorted(ids))
