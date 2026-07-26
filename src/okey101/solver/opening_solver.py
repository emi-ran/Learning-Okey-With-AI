"""Reference set-packing solver for legal meld openings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from okey101.engine.melds import Meld
from okey101.engine.tiles import PhysicalTile, TileValue

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
    return _search_legal_openings(
        hand,
        okey_value,
        threshold=threshold,
        required_tile_id=required_tile_id,
        preserve_final_discard=preserve_final_discard,
        stop_after_first=False,
    )


def _has_legal_opening(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    threshold: int,
    required_tile_id: int,
    preserve_final_discard: bool,
) -> bool:
    return bool(
        _search_legal_openings(
            hand,
            okey_value,
            threshold=threshold,
            required_tile_id=required_tile_id,
            preserve_final_discard=preserve_final_discard,
            stop_after_first=True,
        )
    )


def _search_legal_openings(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    threshold: int,
    required_tile_id: int | None,
    preserve_final_discard: bool,
    stop_after_first: bool,
) -> tuple[OpeningCandidate, ...]:
    hand_ids = {tile.id for tile in hand}
    if required_tile_id is not None and required_tile_id not in hand_ids:
        return ()

    melds = generate_melds(hand, okey_value)
    sorted_hand_ids = tuple(sorted(hand_ids))
    id_bits = {
        tile_id: 1 << index for index, tile_id in enumerate(sorted_hand_ids)
    }
    meld_masks = tuple(
        sum(id_bits[tile.physical_tile.id] for tile in meld.tiles)
        for meld in melds
    )
    meld_scores = tuple(meld.score for meld in melds)
    suffix_tile_masks = [0] * (len(melds) + 1)
    for index in range(len(melds) - 1, -1, -1):
        suffix_tile_masks[index] = suffix_tile_masks[index + 1] | meld_masks[index]

    results: list[OpeningCandidate] = []
    selected_indexes: list[int] = []
    max_used = len(hand) - int(preserve_final_discard)
    required_mask = (
        id_bits[required_tile_id] if required_tile_id is not None else 0
    )

    def visit(
        start: int,
        used_mask: int,
        score: int,
    ) -> bool:
        if (
            selected_indexes
            and score >= threshold
            and (not required_mask or used_mask & required_mask)
        ):
            results.append(
                OpeningCandidate(
                    melds=tuple(melds[index] for index in selected_indexes),
                    score=score,
                    tile_ids=tuple(
                        tile_id
                        for index, tile_id in enumerate(sorted_hand_ids)
                        if used_mask & (1 << index)
                    ),
                )
            )
            if stop_after_first:
                return True

        used_count = used_mask.bit_count()
        if used_count + 3 > max_used:
            return False
        if score + (max_used - used_count) * 13 < threshold:
            return False
        if (
            required_mask
            and not used_mask & required_mask
            and not suffix_tile_masks[start] & required_mask
        ):
            return False

        for index in range(start, len(melds)):
            meld_mask = meld_masks[index]
            if meld_mask & used_mask:
                continue
            combined = used_mask | meld_mask
            if combined.bit_count() > max_used:
                continue
            selected_indexes.append(index)
            found = visit(index + 1, combined, score + meld_scores[index])
            selected_indexes.pop()
            if found and stop_after_first:
                return True
        return False

    visit(0, 0, 0)
    return tuple(results)
