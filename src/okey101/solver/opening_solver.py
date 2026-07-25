"""Reference set-packing solver for legal meld openings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from okey101.engine.melds import Meld
from okey101.engine.tiles import PhysicalTile, TileValue

from .canonicalization import meld_group_key, meld_key
from .meld_generator import generate_melds


@dataclass(frozen=True, slots=True)
class OpeningCandidate:
    melds: tuple[Meld, ...]
    score: int
    tile_ids: tuple[int, ...]


def find_legal_openings(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    threshold: int = 101,
    required_tile_id: int | None = None,
    preserve_final_discard: bool = False,
) -> tuple[OpeningCandidate, ...]:
    """Enumerate order-independent, physically disjoint meld groups."""

    if threshold < 1:
        raise ValueError("opening threshold must be positive")
    hand_ids = {tile.id for tile in hand}
    if required_tile_id is not None and required_tile_id not in hand_ids:
        return ()

    melds = generate_melds(hand, okey_value)
    meld_ids = tuple(
        frozenset(tile.physical_tile.id for tile in meld.tiles) for meld in melds
    )
    results: dict[tuple[tuple[object, ...], ...], OpeningCandidate] = {}
    max_used = len(hand) - int(preserve_final_discard)

    def visit(
        start: int,
        selected: tuple[Meld, ...],
        used_ids: frozenset[int],
        score: int,
    ) -> None:
        if (
            selected
            and score >= threshold
            and (required_tile_id is None or required_tile_id in used_ids)
        ):
            ordered = tuple(sorted(selected, key=meld_key))
            key = meld_group_key(ordered)
            results[key] = OpeningCandidate(
                melds=ordered,
                score=score,
                tile_ids=tuple(sorted(used_ids)),
            )
        for index in range(start, len(melds)):
            ids = meld_ids[index]
            combined = used_ids | ids
            if ids & used_ids or len(combined) > max_used:
                continue
            visit(
                index + 1,
                (*selected, melds[index]),
                combined,
                score + melds[index].score,
            )

    visit(0, (), frozenset(), 0)
    return tuple(results[key] for key in sorted(results))
