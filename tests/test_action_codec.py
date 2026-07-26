from __future__ import annotations

import json
from dataclasses import replace

import pytest

from okey101.engine.actions import (
    AddPair,
    AddToMeld,
    Discard,
    DrawFromStock,
    EndTableActions,
    OpenMelds,
    OpenPairs,
    ReplaceJoker,
    TakePreviousDiscard,
)
from okey101.engine.config import GameConfig, RulesConfig
from okey101.engine.melds import Meld, MeldKind, MeldTile, build_meld
from okey101.engine.pairs import Pair
from okey101.engine.player import OpenedMode, PlayerState
from okey101.engine.round import RoundEngine
from okey101.engine.state import GameState, TurnPhase
from okey101.engine.table import AttachmentSide, TableMeld, TableState
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue
from okey101.rl.action_codec import (
    ActionCatalog,
    DuplicateActionKeyError,
    UnknownActionError,
    build_action_catalog,
    canonical_action_key,
    catalog_from_actions,
)
from okey101.rl.masks import CandidateCapacityError, build_action_mask


def normal(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def state_for_discards(
    hand: tuple[PhysicalTile, ...],
    *,
    table: TableState | None = None,
) -> GameState:
    indicator = normal(24, Color.RED, 13)
    return GameState(
        round_id=1,
        turn_number=1,
        current_player=0,
        starting_player=0,
        indicator=indicator,
        okey_value=TileValue(Color.RED, 1),
        stock=(normal(103, Color.BLACK, 13),),
        discard_pile=(),
        players=(
            PlayerState(hand=hand, opened_mode=OpenedMode.SERIES),
            PlayerState(hand=(normal(30, Color.YELLOW, 3),)),
            PlayerState(hand=(normal(32, Color.YELLOW, 4),)),
            PlayerState(hand=(normal(34, Color.YELLOW, 5),)),
        ),
        table=table or TableState(),
        phase=TurnPhase.DISCARD,
    )


def all_action_types() -> tuple[object, ...]:
    red_two = normal(2, Color.RED, 2)
    red_three = normal(4, Color.RED, 3)
    red_four = normal(6, Color.RED, 4)
    meld = Meld(
        MeldKind.RUN,
        (
            MeldTile(red_two, TileValue(Color.RED, 2)),
            MeldTile(red_three, TileValue(Color.RED, 3)),
            MeldTile(red_four, TileValue(Color.RED, 4)),
        ),
    )
    pair = Pair(
        (
            MeldTile(normal(8, Color.RED, 5), TileValue(Color.RED, 5)),
            MeldTile(normal(9, Color.RED, 5), TileValue(Color.RED, 5)),
        )
    )
    return (
        DrawFromStock(),
        TakePreviousDiscard(),
        OpenMelds((meld,)),
        OpenPairs((pair,)),
        AddToMeld(
            meld_id=7,
            tiles=(MeldTile(normal(10, Color.RED, 6), TileValue(Color.RED, 6)),),
            side=AttachmentSide.RIGHT,
        ),
        AddPair(pair),
        ReplaceJoker(meld_id=7, joker_tile_id=2, replacement_tile_id=10),
        EndTableActions(),
        Discard(tile_id=10),
    )


def test_catalog_is_order_independent_and_round_trips_every_action_type() -> None:
    actions = all_action_types()
    catalog = catalog_from_actions(tuple(reversed(actions)))

    assert len(catalog) == len(actions)
    assert tuple(int(candidate.candidate_id) for candidate in catalog.candidates) == tuple(
        range(len(actions))
    )
    assert [candidate.action for candidate in catalog.candidates] == list(actions)
    for action in actions:
        candidate_id = catalog.encode(action)
        assert catalog.decode(candidate_id) == action
        assert catalog.candidates[candidate_id].action_type is action.type


def test_equal_value_physical_copies_remain_distinct_candidates() -> None:
    first = normal(14, Color.RED, 8)
    second = normal(15, Color.RED, 8)

    catalog = catalog_from_actions((Discard(second.id), Discard(first.id)))

    assert len(catalog) == 2
    assert tuple(catalog.decode(index) for index in range(2)) == (
        Discard(first.id),
        Discard(second.id),
    )
    assert canonical_action_key(Discard(first.id)) != canonical_action_key(
        Discard(second.id)
    )


def test_ordered_attachment_sequences_remain_distinct() -> None:
    first = MeldTile(normal(2, Color.RED, 2), TileValue(Color.RED, 2))
    second = MeldTile(normal(4, Color.RED, 3), TileValue(Color.RED, 3))
    forward = AddToMeld(7, (first, second), AttachmentSide.LEFT)
    reversed_action = AddToMeld(7, (second, first), AttachmentSide.LEFT)

    assert canonical_action_key(forward) != canonical_action_key(reversed_action)


def test_duplicate_canonical_action_is_rejected_instead_of_silently_dropped() -> None:
    with pytest.raises(DuplicateActionKeyError):
        catalog_from_actions((Discard(2), Discard(2)))


def test_catalog_keeps_playable_and_real_okey_discards_enabled() -> None:
    okey = normal(0, Color.RED, 1)
    playable = normal(8, Color.RED, 5)
    ordinary = normal(102, Color.BLACK, 13)
    table_meld = build_meld(
        (
            normal(2, Color.RED, 2),
            normal(4, Color.RED, 3),
            normal(6, Color.RED, 4),
        ),
        TileValue(Color.RED, 1),
    )
    state = state_for_discards(
        (ordinary, playable, okey),
        table=TableState(
            melds=(TableMeld(0, table_meld),),
            next_meld_id=1,
        ),
    )

    catalog = build_action_catalog(state, RulesConfig())
    mask = build_action_mask(catalog)

    assert {candidate.action for candidate in catalog.candidates} == {
        Discard(okey.id),
        Discard(playable.id),
        Discard(ordinary.id),
    }
    assert mask == (True, True, True)
    assert mask[catalog.encode(Discard(okey.id))]
    assert mask[catalog.encode(Discard(playable.id))]


def test_mask_disables_only_padding_and_never_truncates() -> None:
    catalog = catalog_from_actions((DrawFromStock(), TakePreviousDiscard()))

    assert build_action_mask(catalog) == (True, True)
    assert build_action_mask(catalog, capacity=5) == (
        True,
        True,
        False,
        False,
        False,
    )
    with pytest.raises(CandidateCapacityError):
        build_action_mask(catalog, capacity=1)
    with pytest.raises(TypeError):
        build_action_mask(catalog, capacity=True)


def test_empty_terminal_catalog_can_be_padded() -> None:
    state = replace(
        state_for_discards((normal(102, Color.BLACK, 13),)),
        terminal=True,
        phase=TurnPhase.TERMINAL,
    )

    catalog = build_action_catalog(state, RulesConfig())

    assert len(catalog) == 0
    assert build_action_mask(catalog) == ()
    assert build_action_mask(catalog, capacity=3) == (False, False, False)


def test_catalog_rejects_invalid_ids_and_unknown_actions() -> None:
    catalog = catalog_from_actions((DrawFromStock(),))

    with pytest.raises(IndexError):
        catalog.decode(-1)
    with pytest.raises(IndexError):
        catalog.decode(1)
    with pytest.raises(TypeError):
        catalog.decode(True)
    with pytest.raises(UnknownActionError):
        catalog.encode(EndTableActions())


def test_identical_serialized_state_has_identical_catalog() -> None:
    config = GameConfig()
    original = RoundEngine(config)
    original.reset(seed=42)
    original_catalog = build_action_catalog(original.state, config)
    payload = json.loads(json.dumps(original.serialize_state()))

    restored = RoundEngine(config)
    restored.load_state(payload)
    restored_catalog = build_action_catalog(restored.state, config)

    assert restored_catalog == original_catalog


def test_opponent_hidden_hand_changes_do_not_change_catalog() -> None:
    state = state_for_discards((normal(102, Color.BLACK, 13),))
    changed = state.replace_player(
        2,
        PlayerState(
            hand=(
                normal(50, Color.BLUE, 1),
                normal(52, Color.BLUE, 2),
            )
        ),
    )

    assert build_action_catalog(state, RulesConfig()) == build_action_catalog(
        changed,
        RulesConfig(),
    )


def test_catalog_constructor_rejects_noncanonical_candidate_ids() -> None:
    valid = catalog_from_actions((DrawFromStock(),))
    candidate = valid.candidates[0]

    with pytest.raises(ValueError, match="contiguous"):
        ActionCatalog((replace(candidate, candidate_id=2),))
