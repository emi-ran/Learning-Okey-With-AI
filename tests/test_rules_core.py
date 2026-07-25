from __future__ import annotations

from collections import Counter

import pytest

from okey101.engine.config import GameConfig, RulesConfig
from okey101.engine.deck import Deck
from okey101.engine.joker import (
    effective_value,
    is_real_okey,
    okey_value_for_indicator,
)
from okey101.engine.melds import (
    Meld,
    MeldKind,
    MeldTile,
    build_meld,
    find_meld_assignments,
    validate_meld,
)
from okey101.engine.pairs import Pair, build_pair, validate_pair
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue, build_tile_set


def normal(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def fake(tile_id: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.FAKE_OKEY)


def test_standard_set_has_106_unique_physical_tiles() -> None:
    tiles = build_tile_set()
    assert len(tiles) == 106
    assert len({tile.id for tile in tiles}) == 106
    assert sum(tile.kind is TileKind.NORMAL for tile in tiles) == 104
    assert sum(tile.kind is TileKind.FAKE_OKEY for tile in tiles) == 2
    counts = Counter(tile.value for tile in tiles if tile.kind is TileKind.NORMAL)
    assert len(counts) == 52
    assert set(counts.values()) == {2}


def test_physical_tile_shape_is_validated() -> None:
    with pytest.raises(TypeError):
        PhysicalTile(1, TileKind.NORMAL)
    with pytest.raises(ValueError):
        PhysicalTile(1, TileKind.FAKE_OKEY, Color.RED, 1)
    with pytest.raises(ValueError):
        TileValue(Color.RED, 14)


def test_indicator_successor_and_wrap() -> None:
    assert okey_value_for_indicator(TileValue(Color.RED, 1)) == TileValue(Color.RED, 2)
    assert okey_value_for_indicator(TileValue(Color.BLUE, 12)) == TileValue(Color.BLUE, 13)
    assert okey_value_for_indicator(TileValue(Color.YELLOW, 13)) == TileValue(Color.YELLOW, 1)
    with pytest.raises(ValueError):
        okey_value_for_indicator(fake(104))


def test_real_and_fake_okey_have_distinct_semantics() -> None:
    okey = TileValue(Color.YELLOW, 12)
    real = normal(1, Color.YELLOW, 12)
    fake_okey = fake(104)
    assert is_real_okey(real, okey)
    assert effective_value(real, okey) is None
    assert not is_real_okey(fake_okey, okey)
    assert effective_value(fake_okey, okey) == okey


def test_shuffled_deck_is_reproducible_and_uses_local_rng() -> None:
    first = Deck.shuffled(8128)
    second = Deck.shuffled(8128)
    third = Deck.shuffled(8129)
    assert [tile.id for tile in first.tiles] == [tile.id for tile in second.tiles]
    assert [tile.id for tile in first.tiles] != [tile.id for tile in third.tiles]
    drawn = first.draw_many(106)
    assert len({tile.id for tile in drawn}) == 106
    assert len(first) == 0
    with pytest.raises(IndexError):
        first.draw()


@pytest.mark.parametrize(
    ("numbers", "valid"),
    [
        ((1, 2, 3), True),
        ((11, 12, 13), True),
        ((3, 4, 5, 6), True),
        ((12, 13, 1), False),
        ((13, 1, 2), False),
        ((1, 3, 4), False),
    ],
)
def test_run_rules(numbers: tuple[int, ...], valid: bool) -> None:
    okey = TileValue(Color.BLACK, 13)
    tiles = tuple(normal(index, Color.RED, number) for index, number in enumerate(numbers))
    assignments = find_meld_assignments(tiles, okey, kind=MeldKind.RUN)
    assert bool(assignments) is valid


def test_set_requires_three_or_four_distinct_colors() -> None:
    okey = TileValue(Color.BLACK, 13)
    three = [normal(1, Color.RED, 8), normal(2, Color.BLUE, 8), normal(3, Color.YELLOW, 8)]
    four = [*three, normal(4, Color.BLACK, 8)]
    duplicate_color = [normal(1, Color.RED, 8), normal(2, Color.RED, 8), normal(3, Color.BLUE, 8)]
    assert build_meld(three, okey, kind=MeldKind.SET)
    assert build_meld(four, okey, kind=MeldKind.SET)
    with pytest.raises(ValueError):
        build_meld(duplicate_color, okey, kind=MeldKind.SET)
    with pytest.raises(ValueError):
        build_meld(three[:2], okey, kind=MeldKind.SET)


def test_single_and_multiple_okeys_receive_explicit_run_assignments() -> None:
    okey = TileValue(Color.BLUE, 9)
    one_joker = [
        normal(1, Color.RED, 5),
        normal(2, Color.RED, 6),
        normal(3, Color.BLUE, 9),
    ]
    assignments = find_meld_assignments(one_joker, okey, kind=MeldKind.RUN)
    assert [
        [tile.represented_value.number for tile in meld.tiles] for meld in assignments
    ] == [[4, 5, 6], [5, 6, 7]]
    assert all(validate_meld(meld, okey) for meld in assignments)

    two_jokers = [
        normal(1, Color.RED, 5),
        normal(2, Color.BLUE, 9),
        normal(3, Color.BLUE, 9),
        normal(4, Color.RED, 8),
    ]
    meld = build_meld(two_jokers, okey, kind=MeldKind.RUN)
    assert [tile.represented_value.number for tile in meld.tiles] == [5, 6, 7, 8]
    joker_values = {
        tile.physical_tile.id: tile.represented_value.number
        for tile in meld.tiles
        if is_real_okey(tile.physical_tile, okey)
    }
    assert joker_values == {2: 6, 3: 7}


def test_two_okeys_have_multiple_legal_set_assignments() -> None:
    okey = TileValue(Color.BLACK, 11)
    tiles = [
        normal(1, Color.RED, 8),
        normal(2, Color.BLACK, 11),
        normal(3, Color.BLACK, 11),
    ]
    assignments = find_meld_assignments(tiles, okey, kind=MeldKind.SET)
    assert len(assignments) == 3
    assert all(validate_meld(meld, okey) for meld in assignments)
    assert {
        frozenset(tile.represented_value.color for tile in meld.tiles)
        for meld in assignments
    } == {
        frozenset((Color.RED, Color.YELLOW, Color.BLUE)),
        frozenset((Color.RED, Color.YELLOW, Color.BLACK)),
        frozenset((Color.RED, Color.BLUE, Color.BLACK)),
    }


def test_assignment_order_is_independent_of_input_order() -> None:
    okey = TileValue(Color.BLUE, 9)
    tiles = (
        normal(9, Color.RED, 8),
        normal(2, Color.BLUE, 9),
        normal(4, Color.RED, 5),
        normal(1, Color.BLUE, 9),
    )
    forward = find_meld_assignments(tiles, okey, kind=MeldKind.RUN)
    reverse = find_meld_assignments(reversed(tiles), okey, kind=MeldKind.RUN)
    assert forward == reverse


def test_fake_okey_is_fixed_not_wildcard_in_melds() -> None:
    okey = TileValue(Color.YELLOW, 12)
    legal = [normal(1, Color.YELLOW, 10), normal(2, Color.YELLOW, 11), fake(104)]
    illegal = [normal(1, Color.RED, 5), normal(2, Color.RED, 6), fake(104)]
    meld = build_meld(legal, okey, kind=MeldKind.RUN)
    assert meld.tiles[-1].represented_value == okey
    with pytest.raises(ValueError):
        build_meld(illegal, okey, kind=MeldKind.RUN)


def test_validate_meld_rejects_forged_assignments() -> None:
    okey = TileValue(Color.BLUE, 9)
    forged = Meld(
        MeldKind.RUN,
        (
            MeldTile(normal(1, Color.RED, 5), TileValue(Color.RED, 5)),
            MeldTile(normal(2, Color.RED, 6), TileValue(Color.RED, 7)),
            MeldTile(normal(3, Color.RED, 7), TileValue(Color.RED, 6)),
        ),
    )
    assert not validate_meld(forged, okey)


def test_pair_validation_covers_duplicates_and_okeys() -> None:
    okey = TileValue(Color.BLUE, 9)
    duplicate = [normal(1, Color.RED, 7), normal(2, Color.RED, 7)]
    normal_and_okey = [normal(3, Color.YELLOW, 4), normal(4, Color.BLUE, 9)]
    two_okeys = [normal(4, Color.BLUE, 9), normal(5, Color.BLUE, 9)]

    assert validate_pair(build_pair(duplicate, okey), okey)
    mixed = build_pair(normal_and_okey, okey)
    assert mixed.represented_value == TileValue(Color.YELLOW, 4)
    both = build_pair(two_okeys, okey, represented_value=TileValue(Color.BLACK, 13))
    assert both.represented_value == TileValue(Color.BLACK, 13)
    assert validate_pair(both, okey)

    with pytest.raises(ValueError):
        build_pair([normal(6, Color.RED, 7), normal(7, Color.BLUE, 7)], okey)


def test_fake_okey_pairs_as_the_normal_okey_value() -> None:
    okey = TileValue(Color.YELLOW, 12)
    pair = build_pair([fake(104), normal(1, Color.YELLOW, 12)], okey)
    assert pair.represented_value == okey
    assert validate_pair(pair, okey)


def test_fake_okey_cannot_complete_an_unrelated_pair() -> None:
    okey = TileValue(Color.YELLOW, 12)
    with pytest.raises(ValueError):
        build_pair([fake(104), normal(1, Color.RED, 12)], okey)


def test_config_defaults_and_validation() -> None:
    rules = RulesConfig()
    assert rules.player_count == 4
    assert rules.starter_hand_size == 22
    assert rules.opening_min_score == 101
    assert rules.opening_min_pairs == 5
    assert not rules.progressive_opening
    assert rules.max_contiguous_attach == 2
    assert GameConfig(rounds=3).rounds == 3
    with pytest.raises(ValueError):
        RulesConfig(player_count=1)
