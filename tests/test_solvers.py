from __future__ import annotations

from okey101.engine.melds import MeldKind, build_meld
from okey101.engine.table import AttachmentSide
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue
from okey101.solver.attachment_solver import generate_attachments
from okey101.solver.canonicalization import meld_key, pair_key
from okey101.solver.meld_generator import generate_melds
from okey101.solver.opening_solver import find_legal_openings
from okey101.solver.pair_solver import find_pair_openings, generate_pairs


def normal(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def test_meld_generation_is_exhaustive_physical_and_deterministic() -> None:
    okey = TileValue(Color.BLUE, 9)
    hand = (
        normal(1, Color.RED, 5),
        normal(2, Color.RED, 8),
        normal(3, Color.BLUE, 9),
        normal(4, Color.BLUE, 9),
    )
    forward = generate_melds(hand, okey)
    reverse = generate_melds(tuple(reversed(hand)), okey)
    assert tuple(map(meld_key, forward)) == tuple(map(meld_key, reverse))
    assert len(forward) == len(set(map(meld_key, forward)))
    assert all(
        len({tile.physical_tile.id for tile in meld.tiles}) == len(meld.tiles)
        for meld in forward
    )

    forced = [
        meld
        for meld in forward
        if meld.kind is MeldKind.RUN
        and {tile.physical_tile.id for tile in meld.tiles} == {1, 2, 3, 4}
        and [tile.represented_value.number for tile in meld.tiles] == [5, 6, 7, 8]
    ]
    assert len(forced) == 1
    assert [tile.physical_tile.id for tile in forced[0].tiles[1:3]] == [3, 4]


def test_required_tile_filter_is_physical_not_value_based() -> None:
    okey = TileValue(Color.BLACK, 13)
    hand = (
        normal(1, Color.RED, 3),
        normal(2, Color.RED, 3),
        normal(3, Color.RED, 4),
        normal(4, Color.RED, 5),
    )
    candidates = generate_melds(hand, okey, required_tile_id=2)
    assert candidates
    assert all(2 in {tile.physical_tile.id for tile in meld.tiles} for meld in candidates)
    assert any(
        1 not in {tile.physical_tile.id for tile in meld.tiles} for meld in candidates
    )


def _opening_hand() -> tuple[PhysicalTile, ...]:
    specs = (
        *((Color.RED, number) for number in range(10, 14)),
        *((Color.BLUE, number) for number in range(10, 14)),
        *((Color.YELLOW, number) for number in range(3, 6)),
        (Color.BLACK, 1),
    )
    return tuple(
        normal(tile_id, color, number)
        for tile_id, (color, number) in enumerate(specs)
    )


def test_opening_solver_enforces_score_disjointness_and_required_tile() -> None:
    hand = _opening_hand()
    okey = TileValue(Color.BLACK, 13)
    openings = find_legal_openings(
        hand,
        okey,
        threshold=101,
        required_tile_id=9,
        preserve_final_discard=True,
    )
    assert openings
    assert any(candidate.score == 104 for candidate in openings)
    assert all(candidate.score >= 101 and 9 in candidate.tile_ids for candidate in openings)
    assert all(len(candidate.tile_ids) == len(set(candidate.tile_ids)) for candidate in openings)
    assert find_legal_openings(hand, okey, threshold=105) == ()


def test_opening_solver_can_preserve_a_final_discard() -> None:
    hand = _opening_hand()[:-1]
    okey = TileValue(Color.BLACK, 13)
    assert find_legal_openings(hand, okey, threshold=101)
    assert not find_legal_openings(
        hand,
        okey,
        threshold=101,
        preserve_final_discard=True,
    )


def test_pair_solver_enumerates_disjoint_threshold_groups() -> None:
    okey = TileValue(Color.BLACK, 13)
    hand = tuple(
        normal(tile_id, Color.RED, number)
        for tile_id, number in enumerate((1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 12))
    )
    pairs = generate_pairs(hand, okey)
    assert len(pairs) == 5
    assert len(set(map(pair_key, pairs))) == 5

    openings = find_pair_openings(
        hand,
        okey,
        threshold=5,
        required_tile_id=7,
        preserve_final_discard=True,
    )
    assert len(openings) == 1
    assert openings[0].pair_count == 5
    assert 7 in openings[0].tile_ids
    assert len(openings[0].tile_ids) == 10


def test_two_okey_pair_enumerates_all_explicit_representations() -> None:
    okey = TileValue(Color.BLUE, 9)
    pairs = generate_pairs(
        (
            normal(1, Color.BLUE, 9),
            normal(2, Color.BLUE, 9),
        ),
        okey,
    )
    assert len(pairs) == 52
    assert {pair.represented_value for pair in pairs} == {
        TileValue(color, number) for color in Color for number in range(1, 14)
    }


def test_attachment_solver_generates_both_run_sides_up_to_two_tiles() -> None:
    okey = TileValue(Color.BLUE, 9)
    meld = build_meld(
        (
            normal(20, Color.RED, 3),
            normal(21, Color.RED, 4),
            normal(22, Color.RED, 5),
        ),
        okey,
        kind=MeldKind.RUN,
    )
    hand = (
        normal(1, Color.RED, 1),
        normal(2, Color.RED, 2),
        normal(3, Color.RED, 6),
        normal(4, Color.RED, 7),
    )
    candidates = generate_attachments(meld, hand, okey, max_tiles=2)
    shapes = {
        (
            candidate.side,
            tuple(tile.represented_value.number for tile in candidate.tiles),
        )
        for candidate in candidates
    }
    assert shapes == {
        (AttachmentSide.LEFT, (2,)),
        (AttachmentSide.LEFT, (1, 2)),
        (AttachmentSide.RIGHT, (6,)),
        (AttachmentSide.RIGHT, (6, 7)),
    }


def test_set_attachment_and_required_tile_preserve_joker_assignment() -> None:
    okey = TileValue(Color.BLUE, 9)
    meld = build_meld(
        (
            normal(20, Color.RED, 8),
            normal(21, Color.YELLOW, 8),
            normal(22, Color.BLUE, 8),
        ),
        okey,
        kind=MeldKind.SET,
    )
    hand = (
        normal(1, Color.BLACK, 8),
        normal(2, Color.BLUE, 9),
    )
    candidates = generate_attachments(
        meld,
        hand,
        okey,
        required_tile_id=2,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.side is AttachmentSide.SET
    assert candidate.tile_ids == (2,)
    assert candidate.tiles[0].represented_value == TileValue(Color.BLACK, 8)
