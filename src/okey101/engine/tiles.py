"""Physical and logical tile primitives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum


class Color(str, Enum):
    """The four colors, in the canonical engine order."""

    RED = "red"
    YELLOW = "yellow"
    BLUE = "blue"
    BLACK = "black"


class TileKind(str, Enum):
    NORMAL = "normal"
    FAKE_OKEY = "fake_okey"


@dataclass(frozen=True, slots=True)
class TileValue:
    """The value identity shared by the two copies of a normal tile."""

    color: Color
    number: int

    def __post_init__(self) -> None:
        if not isinstance(self.color, Color):
            raise TypeError("color must be a Color")
        if not 1 <= self.number <= 13:
            raise ValueError("tile number must be between 1 and 13")

    @property
    def sort_key(self) -> tuple[int, int]:
        return (_COLOR_INDEX[self.color], self.number)


@dataclass(frozen=True, slots=True)
class PhysicalTile:
    """One of the 106 conserved physical pieces in a round."""

    id: int
    kind: TileKind
    color: Color | None = None
    number: int | None = None

    def __post_init__(self) -> None:
        if self.id < 0:
            raise ValueError("physical tile id cannot be negative")
        if not isinstance(self.kind, TileKind):
            raise TypeError("kind must be a TileKind")
        if self.kind is TileKind.NORMAL:
            if not isinstance(self.color, Color):
                raise TypeError("a normal tile must have a Color")
            if self.number is None or not 1 <= self.number <= 13:
                raise ValueError("a normal tile number must be between 1 and 13")
        elif self.color is not None or self.number is not None:
            raise ValueError("a fake Okey has no permanent color or number")

    @property
    def value(self) -> TileValue | None:
        if self.kind is TileKind.FAKE_OKEY:
            return None
        assert self.color is not None and self.number is not None
        return TileValue(self.color, self.number)

    @property
    def is_fake_okey(self) -> bool:
        return self.kind is TileKind.FAKE_OKEY


_COLOR_INDEX = {color: index for index, color in enumerate(Color)}


def build_tile_set() -> tuple[PhysicalTile, ...]:
    """Build the canonical 104 normal + 2 fake Okey physical tile set."""

    tiles: list[PhysicalTile] = []
    tile_id = 0
    for color in Color:
        for number in range(1, 14):
            for _copy in range(2):
                tiles.append(PhysicalTile(tile_id, TileKind.NORMAL, color, number))
                tile_id += 1
    tiles.extend(
        (
            PhysicalTile(tile_id, TileKind.FAKE_OKEY),
            PhysicalTile(tile_id + 1, TileKind.FAKE_OKEY),
        )
    )
    return tuple(tiles)


def validate_standard_tile_set(tiles: tuple[PhysicalTile, ...]) -> None:
    """Raise if *tiles* is not exactly one canonical 101 Okey set."""

    if len(tiles) != 106:
        raise ValueError("a standard tile set must contain 106 physical tiles")
    ids = [tile.id for tile in tiles]
    if len(set(ids)) != 106:
        raise ValueError("all physical tile ids must be unique")
    normal_counts = Counter(
        tile.value for tile in tiles if tile.kind is TileKind.NORMAL
    )
    expected_values = {TileValue(color, number) for color in Color for number in range(1, 14)}
    if set(normal_counts) != expected_values or any(count != 2 for count in normal_counts.values()):
        raise ValueError("each normal tile value must have exactly two physical copies")
    if sum(tile.kind is TileKind.FAKE_OKEY for tile in tiles) != 2:
        raise ValueError("a standard tile set must contain exactly two fake Okeys")


def is_real_okey(tile: PhysicalTile, okey_value: TileValue) -> bool:
    """Whether a physical normal tile is the wildcard in this round."""

    return tile.kind is TileKind.NORMAL and tile.value == okey_value


def effective_value(
    tile: PhysicalTile,
    okey_value: TileValue,
) -> TileValue | None:
    """Return a fixed value, or ``None`` when the tile is a real wildcard."""

    if is_real_okey(tile, okey_value):
        return None
    if tile.kind is TileKind.FAKE_OKEY:
        return okey_value
    return tile.value
