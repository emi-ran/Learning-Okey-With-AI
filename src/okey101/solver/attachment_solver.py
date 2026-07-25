"""Legal run/set extension candidates, independent of table IDs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from okey101.engine.melds import Meld, MeldKind, MeldTile
from okey101.engine.table import AttachmentSide
from okey101.engine.tiles import Color, PhysicalTile, TileValue

from .meld_generator import assign_targets


@dataclass(frozen=True, slots=True)
class AttachmentCandidate:
    tiles: tuple[MeldTile, ...]
    side: AttachmentSide

    @property
    def tile_ids(self) -> tuple[int, ...]:
        return tuple(tile.physical_tile.id for tile in self.tiles)


def _targets(
    meld: Meld,
    side: AttachmentSide,
    amount: int,
) -> tuple[tuple[TileValue, ...], ...]:
    represented = tuple(tile.represented_value for tile in meld.tiles)
    if meld.kind is MeldKind.RUN:
        color = represented[0].color
        if side is AttachmentSide.LEFT:
            start = represented[0].number - amount
            if start < 1:
                return ()
            return (
                tuple(
                    TileValue(color, number)
                    for number in range(start, represented[0].number)
                ),
            )
        if side is AttachmentSide.RIGHT:
            end = represented[-1].number + amount
            if end > 13:
                return ()
            return (
                tuple(
                    TileValue(color, number)
                    for number in range(represented[-1].number + 1, end + 1)
                ),
            )
        return ()

    if side is not AttachmentSide.SET:
        return ()
    number = represented[0].number
    used_colors = {value.color for value in represented}
    missing = tuple(
        TileValue(color, number) for color in Color if color not in used_colors
    )
    return tuple(combinations(missing, amount))


def generate_attachments(
    meld: Meld,
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    max_tiles: int = 2,
    required_tile_id: int | None = None,
) -> tuple[AttachmentCandidate, ...]:
    """Enumerate one/two-tile extensions with explicit Okey assignments."""

    if max_tiles < 1:
        return ()
    sides = (
        (AttachmentSide.LEFT, AttachmentSide.RIGHT)
        if meld.kind is MeldKind.RUN
        else (AttachmentSide.SET,)
    )
    candidates: dict[tuple[object, ...], AttachmentCandidate] = {}
    for side in sides:
        for amount in range(1, min(max_tiles, len(hand)) + 1):
            for targets in _targets(meld, side, amount):
                for assignment in assign_targets(hand, targets, okey_value):
                    ids = tuple(tile.physical_tile.id for tile in assignment)
                    if required_tile_id is not None and required_tile_id not in ids:
                        continue
                    candidate = AttachmentCandidate(assignment, side)
                    key = (
                        side.value,
                        tuple(
                            (
                                tile.physical_tile.id,
                                tile.represented_value.sort_key,
                            )
                            for tile in assignment
                        ),
                    )
                    candidates[key] = candidate
    return tuple(candidates[key] for key in sorted(candidates))
