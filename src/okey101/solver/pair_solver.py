"""Physical pair candidates and pair-opening set packing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from okey101.engine.melds import MeldTile
from okey101.engine.pairs import Pair
from okey101.engine.tiles import Color, PhysicalTile, TileValue

from .canonicalization import pair_key
from .meld_generator import index_hand


_ALL_TILE_VALUES = tuple(
    TileValue(color, number) for color in Color for number in range(1, 14)
)


@dataclass(frozen=True, slots=True)
class PairOpeningCandidate:
    pairs: tuple[Pair, ...]
    tile_ids: tuple[int, ...]

    @property
    def pair_count(self) -> int:
        return len(self.pairs)


def generate_pairs(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    required_tile_id: int | None = None,
) -> tuple[Pair, ...]:
    """Enumerate every legal two-physical-tile pair."""

    fixed_by_value, jokers = index_hand(hand, okey_value)
    candidates: dict[tuple[object, ...], Pair] = {}

    def add_pair(
        first: PhysicalTile,
        second: PhysicalTile,
        represented_value: TileValue,
    ) -> None:
        if (
            required_tile_id is not None
            and required_tile_id not in (first.id, second.id)
        ):
            return
        physical_tiles = sorted((first, second), key=lambda tile: tile.id)
        pair = Pair(
            (
                MeldTile(physical_tiles[0], represented_value),
                MeldTile(physical_tiles[1], represented_value),
            )
        )
        candidates[pair_key(pair)] = pair

    for represented_value, fixed_tiles in fixed_by_value.items():
        for first, second in combinations(fixed_tiles, 2):
            add_pair(first, second, represented_value)
        for fixed_tile in fixed_tiles:
            for joker in jokers:
                add_pair(fixed_tile, joker, represented_value)

    for first, second in combinations(jokers, 2):
        for represented_value in _ALL_TILE_VALUES:
            add_pair(first, second, represented_value)

    return tuple(candidates[key] for key in sorted(candidates))


def find_pair_openings(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    threshold: int = 5,
    required_tile_id: int | None = None,
    preserve_final_discard: bool = False,
) -> tuple[PairOpeningCandidate, ...]:
    """Enumerate disjoint pair groups meeting the configured count."""

    if threshold < 1:
        raise ValueError("pair threshold must be positive")
    return _search_pair_openings(
        hand,
        okey_value,
        threshold=threshold,
        required_tile_id=required_tile_id,
        preserve_final_discard=preserve_final_discard,
        stop_after_first=False,
    )


def _has_pair_opening(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    threshold: int,
    required_tile_id: int,
    preserve_final_discard: bool,
) -> bool:
    return bool(
        _search_pair_openings(
            hand,
            okey_value,
            threshold=threshold,
            required_tile_id=required_tile_id,
            preserve_final_discard=preserve_final_discard,
            stop_after_first=True,
        )
    )


def _search_pair_openings(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    threshold: int,
    required_tile_id: int | None,
    preserve_final_discard: bool,
    stop_after_first: bool,
) -> tuple[PairOpeningCandidate, ...]:
    hand_ids = {tile.id for tile in hand}
    if required_tile_id is not None and required_tile_id not in hand_ids:
        return ()

    pairs = generate_pairs(hand, okey_value)
    sorted_hand_ids = tuple(sorted(hand_ids))
    id_bits = {
        tile_id: 1 << index for index, tile_id in enumerate(sorted_hand_ids)
    }
    pair_masks = tuple(
        sum(id_bits[tile.physical_tile.id] for tile in pair.tiles)
        for pair in pairs
    )
    suffix_tile_masks = [0] * (len(pairs) + 1)
    for index in range(len(pairs) - 1, -1, -1):
        suffix_tile_masks[index] = suffix_tile_masks[index + 1] | pair_masks[index]

    results: list[PairOpeningCandidate] = []
    selected_indexes: list[int] = []
    max_used = len(hand) - int(preserve_final_discard)
    required_mask = (
        id_bits[required_tile_id] if required_tile_id is not None else 0
    )

    def visit(
        start: int,
        used_mask: int,
    ) -> bool:
        if (
            len(selected_indexes) >= threshold
            and (not required_mask or used_mask & required_mask)
        ):
            results.append(
                PairOpeningCandidate(
                    pairs=tuple(pairs[index] for index in selected_indexes),
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
        if len(selected_indexes) + (max_used - used_count) // 2 < threshold:
            return False
        if (
            required_mask
            and not used_mask & required_mask
            and not suffix_tile_masks[start] & required_mask
        ):
            return False

        for index in range(start, len(pairs)):
            pair_mask = pair_masks[index]
            if pair_mask & used_mask:
                continue
            combined = used_mask | pair_mask
            if combined.bit_count() > max_used:
                continue
            selected_indexes.append(index)
            found = visit(index + 1, combined)
            selected_indexes.pop()
            if found and stop_after_first:
                return True
        return False

    visit(0, 0)
    return tuple(results)


generate_pair_candidates = generate_pairs
