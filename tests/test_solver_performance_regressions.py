from __future__ import annotations

import hashlib
from itertools import combinations

from okey101.engine.config import RulesConfig
from okey101.engine.legal_actions import get_legal_actions
from okey101.engine.melds import find_meld_assignments
from okey101.engine.pairs import build_pair
from okey101.engine.round import RoundEngine
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue
from okey101.solver.canonicalization import (
    meld_group_key,
    meld_key,
    pair_group_key,
    pair_key,
)
from okey101.solver.meld_generator import generate_melds
from okey101.solver.opening_solver import find_legal_openings
from okey101.solver.pair_solver import find_pair_openings, generate_pairs


def _digest(items: object) -> str:
    return hashlib.sha256(repr(tuple(items)).encode()).hexdigest()


def test_seed_8_starter_solver_and_action_fingerprints_are_stable() -> None:
    state = RoundEngine().reset(8)
    hand = state.current_player_state.hand
    melds = generate_melds(hand, state.okey_value)
    openings = find_legal_openings(
        hand,
        state.okey_value,
        threshold=state.progressive_series_threshold,
        preserve_final_discard=True,
    )
    pairs = generate_pairs(hand, state.okey_value)
    pair_openings = find_pair_openings(
        hand,
        state.okey_value,
        threshold=state.progressive_pair_threshold,
        preserve_final_discard=True,
    )
    actions = get_legal_actions(state, RulesConfig())

    assert len(melds) == 672
    assert _digest(meld_key(meld) for meld in melds) == (
        "96c466ef7ea9a91d8d9f3f97217000c2e8d00b5a017a847ac5f879d79d6df9c1"
    )
    assert len(openings) == 2098
    assert _digest(
        (meld_group_key(candidate.melds), candidate.score, candidate.tile_ids)
        for candidate in openings
    ) == "8b29dade69bc137345605d75a9d46490c7d0ee8a6bb4b0c8360f5bac70a99025"
    assert len(pairs) == 94
    assert _digest(pair_key(pair) for pair in pairs) == (
        "d42a7f14f037de58d2d2623761b7d6d472aef4449dcbb0d9650c392b81eee39b"
    )
    assert len(pair_openings) == 0
    assert _digest(
        (pair_group_key(candidate.pairs), candidate.tile_ids)
        for candidate in pair_openings
    ) == "2e38e77b22c314a449e91fafed92a43826ac6aa403ae6a8acb6cf58239fbaf5d"
    assert len(actions) == 2099
    assert _digest(sorted(map(repr, actions))) == (
        "0745e87c74963f8d75353106463d0420c85ec7b0feb84a44392282a60bf39274"
    )


def test_direct_pair_generation_matches_reference_validation() -> None:
    state = RoundEngine().reset(8)
    hand = state.current_player_state.hand
    expected = {}
    for first, second in combinations(hand, 2):
        represented_values = (
            tuple(
                TileValue(color, number)
                for color in Color
                for number in range(1, 14)
            )
            if first.value == state.okey_value and second.value == state.okey_value
            else (None,)
        )
        for represented_value in represented_values:
            try:
                pair = build_pair(
                    (first, second),
                    state.okey_value,
                    represented_value=represented_value,
                )
            except ValueError:
                continue
            expected[pair_key(pair)] = pair

    actual = generate_pairs(hand, state.okey_value)

    assert tuple(map(pair_key, actual)) == tuple(sorted(expected))


def test_meld_and_opening_solvers_match_naive_small_hand_reference() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    hand = (
        PhysicalTile(0, TileKind.NORMAL, Color.RED, 2),
        PhysicalTile(1, TileKind.NORMAL, Color.RED, 3),
        PhysicalTile(2, TileKind.NORMAL, Color.RED, 4),
        PhysicalTile(3, TileKind.NORMAL, Color.RED, 5),
        PhysicalTile(4, TileKind.NORMAL, Color.BLUE, 7),
        PhysicalTile(5, TileKind.NORMAL, Color.YELLOW, 7),
        PhysicalTile(6, TileKind.NORMAL, Color.BLACK, 7),
        PhysicalTile(7, TileKind.NORMAL, Color.BLUE, 13),
    )
    expected_melds = {}
    for size in range(3, len(hand) + 1):
        for subset in combinations(hand, size):
            for meld in find_meld_assignments(subset, okey_value):
                expected_melds[meld_key(meld)] = meld

    actual_melds = generate_melds(hand, okey_value)

    assert tuple(map(meld_key, actual_melds)) == tuple(sorted(expected_melds))

    threshold = 21
    expected_openings = {}
    meld_ids = tuple(
        frozenset(tile.physical_tile.id for tile in meld.tiles)
        for meld in actual_melds
    )

    def visit(
        start: int,
        selected: tuple,
        used_ids: frozenset[int],
        score: int,
    ) -> None:
        if selected and score >= threshold:
            expected_openings[meld_group_key(selected)] = (
                score,
                tuple(sorted(used_ids)),
            )
        for index in range(start, len(actual_melds)):
            ids = meld_ids[index]
            if ids & used_ids or len(used_ids | ids) >= len(hand):
                continue
            visit(
                index + 1,
                (*selected, actual_melds[index]),
                used_ids | ids,
                score + actual_melds[index].score,
            )

    visit(0, (), frozenset(), 0)
    actual_openings = find_legal_openings(
        hand,
        okey_value,
        threshold=threshold,
        preserve_final_discard=True,
    )
    actual_by_key = {
        meld_group_key(candidate.melds): (
            candidate.score,
            candidate.tile_ids,
        )
        for candidate in actual_openings
    }

    assert actual_by_key == expected_openings
