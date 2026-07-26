"""Pattern-driven exhaustive meld generation without subset brute force."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from okey101.engine.joker import effective_value
from okey101.engine.melds import Meld, MeldKind, MeldTile
from okey101.engine.tiles import Color, PhysicalTile, TileValue

from .canonicalization import meld_key


RUN_TARGETS = tuple(
    tuple(TileValue(color, number) for number in range(start, end + 1))
    for color in Color
    for start in range(1, 12)
    for end in range(start + 2, 14)
)
SET_TARGETS = tuple(
    tuple(TileValue(color, number) for color in colors)
    for number in range(1, 14)
    for length in (3, 4)
    for colors in combinations(Color, length)
)


def _value_bit(value: TileValue) -> int:
    color_index, number = value.sort_key
    return 1 << (color_index * 13 + number - 1)


RUN_TARGET_MASKS = tuple(
    sum(_value_bit(value) for value in targets) for targets in RUN_TARGETS
)
SET_TARGET_MASKS = tuple(
    sum(_value_bit(value) for value in targets) for targets in SET_TARGETS
)


def index_hand(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
) -> tuple[dict[TileValue, tuple[PhysicalTile, ...]], tuple[PhysicalTile, ...]]:
    """Index fixed values separately from the round's physical real Okeys."""

    ids = [tile.id for tile in hand]
    if len(ids) != len(set(ids)):
        raise ValueError("hand contains a duplicate physical tile id")
    fixed_by_value: dict[TileValue, list[PhysicalTile]] = {}
    jokers: list[PhysicalTile] = []
    for tile in hand:
        value = effective_value(tile, okey_value)
        if value is None:
            jokers.append(tile)
        else:
            fixed_by_value.setdefault(value, []).append(tile)
    return (
        {
            value: tuple(sorted(tiles, key=lambda candidate: candidate.id))
            for value, tiles in fixed_by_value.items()
        },
        tuple(sorted(jokers, key=lambda candidate: candidate.id)),
    )


def assign_indexed_targets(
    fixed_by_value: dict[TileValue, tuple[PhysicalTile, ...]],
    jokers: tuple[PhysicalTile, ...],
    targets: Sequence[TileValue],
) -> tuple[tuple[MeldTile, ...], ...]:
    """Assign distinct physical tiles to one ordered logical pattern."""

    if sum(target not in fixed_by_value for target in targets) > len(jokers):
        return ()

    if len(set(targets)) != len(targets):
        return _assign_repeated_targets(fixed_by_value, jokers, targets)

    results: list[tuple[MeldTile, ...]] = []
    assigned: list[MeldTile] = []

    def visit(
        target_index: int,
        first_available_joker: int,
    ) -> None:
        if target_index == len(targets):
            results.append(tuple(assigned))
            return
        target = targets[target_index]
        for tile in fixed_by_value.get(target, ()):
            assigned.append(MeldTile(tile, target))
            visit(target_index + 1, first_available_joker)
            assigned.pop()
        for joker_index in range(first_available_joker, len(jokers)):
            assigned.append(MeldTile(jokers[joker_index], target))
            # Bind lower-ID identical Okeys to earlier missing values and omit
            # only the physically swapped equivalent.
            visit(target_index + 1, joker_index + 1)
            assigned.pop()

    visit(0, 0)
    return tuple(results)


def _assign_repeated_targets(
    fixed_by_value: dict[TileValue, tuple[PhysicalTile, ...]],
    jokers: tuple[PhysicalTile, ...],
    targets: Sequence[TileValue],
) -> tuple[tuple[MeldTile, ...], ...]:
    """General fallback for callers outside the run/set pattern tables."""

    results: list[tuple[MeldTile, ...]] = []

    def visit(
        target_index: int,
        used_ids: frozenset[int],
        assigned: tuple[MeldTile, ...],
        last_joker_id: int | None,
    ) -> None:
        if target_index == len(targets):
            results.append(assigned)
            return
        target = targets[target_index]
        for tile in fixed_by_value.get(target, ()):
            if tile.id not in used_ids:
                visit(
                    target_index + 1,
                    used_ids | {tile.id},
                    (*assigned, MeldTile(tile, target)),
                    last_joker_id,
                )
        for tile in jokers:
            if tile.id in used_ids:
                continue
            if last_joker_id is not None and tile.id < last_joker_id:
                continue
            visit(
                target_index + 1,
                used_ids | {tile.id},
                (*assigned, MeldTile(tile, target)),
                tile.id,
            )

    visit(0, frozenset(), (), None)
    return tuple(results)


def assign_targets(
    hand: Sequence[PhysicalTile],
    targets: Sequence[TileValue],
    okey_value: TileValue,
) -> tuple[tuple[MeldTile, ...], ...]:
    fixed_by_value, jokers = index_hand(hand, okey_value)
    return assign_indexed_targets(fixed_by_value, jokers, targets)


def generate_melds(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
    *,
    required_tile_id: int | None = None,
) -> tuple[Meld, ...]:
    """Enumerate every physical run/set candidate in canonical order."""

    fixed_by_value, jokers = index_hand(hand, okey_value)
    fixed_value_mask = sum(_value_bit(value) for value in fixed_by_value)
    candidates: dict[tuple[object, ...], Meld] = {}
    for kind, targets_collection, target_masks in (
        (MeldKind.RUN, RUN_TARGETS, RUN_TARGET_MASKS),
        (MeldKind.SET, SET_TARGETS, SET_TARGET_MASKS),
    ):
        for targets, target_mask in zip(targets_collection, target_masks):
            if (target_mask & ~fixed_value_mask).bit_count() > len(jokers):
                continue
            for assignment in assign_indexed_targets(fixed_by_value, jokers, targets):
                if (
                    required_tile_id is not None
                    and all(
                        meld_tile.physical_tile.id != required_tile_id
                        for meld_tile in assignment
                    )
                ):
                    continue
                meld = Meld(kind, assignment)
                candidates[meld_key(meld)] = meld
    return tuple(candidates[key] for key in sorted(candidates))


generate_meld_candidates = generate_melds
