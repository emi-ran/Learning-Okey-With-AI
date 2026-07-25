"""Physical pair candidates and pair-opening set packing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from okey101.engine.joker import effective_value
from okey101.engine.pairs import Pair, build_pair
from okey101.engine.tiles import Color, PhysicalTile, TileValue

from .canonicalization import pair_group_key, pair_key


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

    ids = [tile.id for tile in hand]
    if len(ids) != len(set(ids)):
        raise ValueError("hand contains a duplicate physical tile id")
    candidates: dict[tuple[object, ...], Pair] = {}
    for first, second in combinations(hand, 2):
        if required_tile_id is not None and required_tile_id not in (first.id, second.id):
            continue
        both_are_wild = (
            effective_value(first, okey_value) is None
            and effective_value(second, okey_value) is None
        )
        represented_values = (
            tuple(TileValue(color, number) for color in Color for number in range(1, 14))
            if both_are_wild
            else (None,)
        )
        for represented_value in represented_values:
            try:
                pair = build_pair(
                    (first, second),
                    okey_value,
                    represented_value=represented_value,
                )
            except ValueError:
                continue
            candidates[pair_key(pair)] = pair
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
    hand_ids = {tile.id for tile in hand}
    if required_tile_id is not None and required_tile_id not in hand_ids:
        return ()
    pairs = generate_pairs(hand, okey_value)
    pair_ids = tuple(
        frozenset(tile.physical_tile.id for tile in pair.tiles) for pair in pairs
    )
    results: dict[tuple[tuple[object, ...], ...], PairOpeningCandidate] = {}
    max_used = len(hand) - int(preserve_final_discard)

    def visit(
        start: int,
        selected: tuple[Pair, ...],
        used_ids: frozenset[int],
    ) -> None:
        if (
            len(selected) >= threshold
            and (required_tile_id is None or required_tile_id in used_ids)
        ):
            ordered = tuple(sorted(selected, key=pair_key))
            key = pair_group_key(ordered)
            results[key] = PairOpeningCandidate(
                pairs=ordered,
                tile_ids=tuple(sorted(used_ids)),
            )
        for index in range(start, len(pairs)):
            ids = pair_ids[index]
            combined = used_ids | ids
            if ids & used_ids or len(combined) > max_used:
                continue
            visit(index + 1, (*selected, pairs[index]), combined)

    visit(0, (), frozenset())
    return tuple(results[key] for key in sorted(results))


generate_pair_candidates = generate_pairs
