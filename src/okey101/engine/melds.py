"""Run/set validation with explicit, deterministic Okey assignments."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

from .joker import effective_value, is_real_okey
from .tiles import Color, PhysicalTile, TileValue


class MeldKind(str, Enum):
    RUN = "run"
    SERIES = "run"
    SET = "set"


@dataclass(frozen=True, slots=True)
class MeldTile:
    physical_tile: PhysicalTile
    represented_value: TileValue

    @property
    def physical_tile_id(self) -> int:
        return self.physical_tile.id


@dataclass(frozen=True, slots=True)
class Meld:
    kind: MeldKind
    tiles: tuple[MeldTile, ...]

    @property
    def score(self) -> int:
        return sum(tile.represented_value.number for tile in self.tiles)

    @property
    def physical_tiles(self) -> tuple[PhysicalTile, ...]:
        return tuple(tile.physical_tile for tile in self.tiles)


def _tile_sort_key(tile: PhysicalTile) -> tuple[int, int, int]:
    value = tile.value
    if value is None:
        return (len(Color), 14, tile.id)
    return (*value.sort_key, tile.id)


def _assign_to_target(
    tiles: tuple[PhysicalTile, ...],
    target: tuple[TileValue, ...],
    okey_value: TileValue,
) -> tuple[MeldTile, ...] | None:
    available = list(target)
    fixed: list[tuple[PhysicalTile, TileValue]] = []
    jokers: list[PhysicalTile] = []
    for tile in tiles:
        value = effective_value(tile, okey_value)
        if value is None:
            jokers.append(tile)
            continue
        if value not in available:
            return None
        available.remove(value)
        fixed.append((tile, value))
    if len(available) != len(jokers):
        return None

    assignments = fixed + list(zip(sorted(jokers, key=lambda item: item.id), available))
    by_value = {value: index for index, value in enumerate(target)}
    assignments.sort(key=lambda item: (by_value[item[1]], item[0].id))
    return tuple(MeldTile(tile, value) for tile, value in assignments)


def _candidate_runs(length: int) -> Iterable[tuple[TileValue, ...]]:
    if not 3 <= length <= 13:
        return
    for color in Color:
        for start in range(1, 15 - length):
            yield tuple(TileValue(color, number) for number in range(start, start + length))


def _candidate_sets(length: int) -> Iterable[tuple[TileValue, ...]]:
    if length not in (3, 4):
        return
    for number in range(1, 14):
        for colors in combinations(Color, length):
            yield tuple(TileValue(color, number) for color in colors)


def find_meld_assignments(
    tiles: Iterable[PhysicalTile],
    okey_value: TileValue,
    *,
    kind: MeldKind | None = None,
) -> tuple[Meld, ...]:
    """Enumerate every legal run/set interpretation in canonical order."""

    physical_tiles = tuple(tiles)
    ids = [tile.id for tile in physical_tiles]
    if len(ids) != len(set(ids)):
        return ()

    kinds = (MeldKind.RUN, MeldKind.SET) if kind is None else (MeldKind(kind),)
    results: list[Meld] = []
    for candidate_kind in kinds:
        targets = (
            _candidate_runs(len(physical_tiles))
            if candidate_kind is MeldKind.RUN
            else _candidate_sets(len(physical_tiles))
        )
        for target in targets:
            assignment = _assign_to_target(physical_tiles, target, okey_value)
            if assignment is not None:
                results.append(Meld(candidate_kind, assignment))
    return tuple(results)


def build_meld(
    tiles: Iterable[PhysicalTile],
    okey_value: TileValue,
    *,
    kind: MeldKind | None = None,
) -> Meld:
    """Build the first canonical legal interpretation of physical *tiles*."""

    assignments = find_meld_assignments(tiles, okey_value, kind=kind)
    if not assignments:
        raise ValueError("tiles do not form a legal meld")
    return assignments[0]


def validate_meld(meld: Meld, okey_value: TileValue) -> bool:
    """Validate both meld shape and every stored represented value."""

    ids = [tile.physical_tile.id for tile in meld.tiles]
    if len(ids) != len(set(ids)):
        return False

    for meld_tile in meld.tiles:
        tile = meld_tile.physical_tile
        if not is_real_okey(tile, okey_value):
            if effective_value(tile, okey_value) != meld_tile.represented_value:
                return False

    values = [tile.represented_value for tile in meld.tiles]
    if meld.kind is MeldKind.RUN:
        if len(values) < 3:
            return False
        colors = {value.color for value in values}
        numbers = sorted(value.number for value in values)
        return len(colors) == 1 and numbers == list(range(numbers[0], numbers[0] + len(numbers)))
    if meld.kind is MeldKind.SET:
        return (
            len(values) in (3, 4)
            and len({value.number for value in values}) == 1
            and len({value.color for value in values}) == len(values)
        )
    return False
