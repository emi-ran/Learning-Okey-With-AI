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
from okey101.engine.config import GameConfig
from okey101.engine.invariants import InvariantError
from okey101.engine.melds import (
    Meld,
    MeldKind,
    MeldTile,
    build_meld,
)
from okey101.engine.pairs import build_pair
from okey101.engine.player import OpenedMode, PlayerState
from okey101.engine.round import (
    ReplayError,
    RoundEngine,
    create_round_state,
    deserialize_action,
    replay_from_seed_and_actions,
    serialize_action,
)
from okey101.engine.state import (
    DiscardRecord,
    DrawSource,
    EventType,
    GameState,
    TerminalReason,
    TurnContext,
    TurnPhase,
)
from okey101.engine.table import AttachmentSide, TableState
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue, build_tile_set
from okey101.engine.transition import IllegalAction, apply_action


TILES = build_tile_set()


def normal(color: Color, number: int, copy: int = 0) -> PhysicalTile:
    matches = [
        tile
        for tile in TILES
        if tile.kind is TileKind.NORMAL
        and tile.color is color
        and tile.number == number
    ]
    return matches[copy]


def state_for(
    hand: tuple[PhysicalTile, ...],
    *,
    opened_mode: OpenedMode = OpenedMode.NONE,
    phase: TurnPhase = TurnPhase.TABLE_ACTIONS,
    stock: tuple[PhysicalTile, ...] = (),
    discard_pile: tuple[PhysicalTile, ...] = (),
    discard_history: tuple[DiscardRecord, ...] = (),
    table: TableState | None = None,
    okey_value: TileValue = TileValue(Color.YELLOW, 1),
    turn_context: TurnContext | None = None,
    players: tuple[PlayerState, ...] | None = None,
) -> GameState:
    if players is None:
        players = (
            PlayerState(hand=hand, opened_mode=opened_mode),
            PlayerState(),
            PlayerState(),
            PlayerState(),
        )
    return GameState(
        round_id=1,
        turn_number=3,
        current_player=0,
        starting_player=0,
        indicator=normal(Color.YELLOW, 13),
        okey_value=okey_value,
        stock=stock,
        discard_pile=discard_pile,
        players=players,
        discard_history=discard_history,
        table=table or TableState(),
        phase=phase,
        turn_context=turn_context
        or TurnContext(opened_mode_at_start=opened_mode),
    )


def meld_tiles(*tiles: PhysicalTile) -> tuple[MeldTile, ...]:
    return tuple(MeldTile(tile, tile.value) for tile in tiles)  # type: ignore[arg-type]


def test_deal_is_deterministic_and_conserves_all_physical_tiles() -> None:
    first, first_events = create_round_state(seed=418, starting_player=2)
    second, second_events = create_round_state(seed=418, starting_player=2)

    assert first == second
    assert first_events == second_events
    assert first.phase is TurnPhase.TABLE_ACTIONS
    assert first.turn_context.draw_source is DrawSource.DEAL
    assert [len(player.hand) for player in first.players] == [21, 21, 22, 21]
    assert len(first.stock) == 20

    locations = [
        first.indicator.id,
        *(tile.id for tile in first.stock),
        *(tile.id for player in first.players for tile in player.hand),
    ]
    assert len(locations) == 106
    assert len(set(locations)) == 106


def test_round_engine_reset_replays_seed_without_global_rng_state() -> None:
    engine = RoundEngine()
    first = engine.reset(seed=91, starting_player=1)
    serialized = engine.serialize_state()
    second = engine.reset(seed=91, starting_player=1)

    assert first == second
    assert serialized == engine.serialize_state()
    assert engine.event_log[0].type is EventType.DEAL


def test_json_state_round_trip_restores_table_joker_assignments_and_turn_context() -> None:
    state, _ = create_round_state(seed=812, starting_player=0)
    target_color = next(color for color in Color if color is not state.okey_value.color)

    def located_tile(color: Color, number: int, copy: int = 0) -> PhysicalTile:
        candidates = [
            tile
            for tile in (
                *state.stock,
                *(hand_tile for player in state.players for hand_tile in player.hand),
            )
            if tile.color is color and tile.number == number
        ]
        return candidates[copy]

    real_okey = located_tile(state.okey_value.color, state.okey_value.number)
    run = build_meld(
        (
            located_tile(target_color, 2),
            real_okey,
            located_tile(target_color, 4),
        ),
        state.okey_value,
    )
    pair_color = next(
        color
        for color in Color
        if color not in {state.okey_value.color, target_color}
    )
    pair = build_pair(
        (
            located_tile(pair_color, 8, 0),
            located_tile(pair_color, 8, 1),
        ),
        state.okey_value,
    )
    relocated_ids = {
        *(tile.id for tile in run.physical_tiles),
        *(tile.id for tile in pair.physical_tiles),
    }
    players = tuple(
        replace(
            player,
            hand=tuple(tile for tile in player.hand if tile.id not in relocated_ids),
            opened_mode=OpenedMode.SERIES if player_id == 0 else player.opened_mode,
            opening_turn=1 if player_id == 0 else player.opening_turn,
        )
        for player_id, player in enumerate(state.players)
    )
    table, (meld_id,) = TableState(pairs=(pair,)).add_melds((run,))
    drawn_tile_id = players[0].hand[0].id
    state = replace(
        state,
        turn_number=9,
        current_player=0,
        stock=tuple(tile for tile in state.stock if tile.id not in relocated_ids),
        players=players,
        table=table,
        phase=TurnPhase.TABLE_ACTIONS,
        turn_context=TurnContext(
            draw_source=DrawSource.PREVIOUS_DISCARD,
            drawn_tile_id=drawn_tile_id,
            taken_discard_tile_id=drawn_tile_id,
            opened_mode_at_start=OpenedMode.SERIES,
            attachment_usage=(),
        ).add_attachment_usage(meld_id, AttachmentSide.RIGHT, 1),
    )

    source = RoundEngine()
    source.state = state
    payload = json.loads(json.dumps(source.serialize_state()))
    restored_engine = RoundEngine()
    restored = restored_engine.load_state(payload)

    assert restored == state
    assert restored_engine.serialize_state() == payload
    restored_joker = next(
        tile
        for tile in restored.table.meld(meld_id).meld.tiles
        if tile.physical_tile.id == real_okey.id
    )
    assert restored_joker.represented_value == TileValue(target_color, 3)
    assert restored.turn_context.must_use_taken_discard
    assert restored.turn_context.attachment_count(meld_id, AttachmentSide.RIGHT) == 1
    assert len(restored.table.pairs) == 1


def test_load_state_rejects_malformed_enum_with_field_path() -> None:
    engine = RoundEngine()
    engine.reset(seed=1, starting_player=0)
    payload = engine.serialize_state()
    payload["phase"] = "not-a-phase"

    with pytest.raises(ValueError, match=r"\$\.phase"):
        RoundEngine().load_state(payload)


def test_load_state_rejects_rebound_physical_tile_identity() -> None:
    engine = RoundEngine()
    engine.reset(seed=2, starting_player=0)
    payload = engine.serialize_state()
    hand = payload["players"][0]["hand"]
    tile = next(item for item in hand if item["kind"] == "normal")
    tile["number"] = tile["number"] % 13 + 1

    with pytest.raises(InvariantError, match="PHYSICAL_TILE_IDENTITY"):
        RoundEngine().load_state(payload)


def test_replay_from_seed_and_actions_reproduces_state_and_events() -> None:
    seed = 144
    starting_player = 2
    original = RoundEngine()
    state = original.reset(seed=seed, starting_player=starting_player)
    actions = (
        EndTableActions(),
        Discard(state.current_player_state.hand[0].id),
        DrawFromStock(),
    )
    for action in actions:
        original.step(action)

    replayed = replay_from_seed_and_actions(
        seed,
        actions,
        starting_player=starting_player,
    )

    assert replayed.state == original.state
    assert replayed.action_history == list(actions)
    assert replayed.event_log == original.event_log


def test_action_serialization_round_trips_embedded_physical_tiles() -> None:
    action = OpenMelds(
        (
            build_meld(
                (
                    normal(Color.RED, 2),
                    normal(Color.RED, 3),
                    normal(Color.RED, 4),
                ),
                TileValue(Color.YELLOW, 1),
            ),
        )
    )

    payload = json.loads(json.dumps(serialize_action(action)))

    assert deserialize_action(payload) == action


def test_replay_error_reports_failing_action_index() -> None:
    with pytest.raises(ReplayError, match="action 0") as error:
        replay_from_seed_and_actions(
            5,
            (DrawFromStock(),),
            starting_player=0,
        )

    assert error.value.action_index == 0
    assert isinstance(error.value.action, DrawFromStock)


def test_stock_draw_decrements_stock_but_previous_discard_pickup_does_not() -> None:
    stock_tile = normal(Color.BLUE, 9)
    discard_tile = normal(Color.RED, 5)
    table_meld = build_meld(
        (
            normal(Color.RED, 2),
            normal(Color.RED, 3),
            normal(Color.RED, 4),
        ),
        TileValue(Color.YELLOW, 1),
    )
    table, _meld_ids = TableState().add_melds((table_meld,))
    base = state_for(
        (normal(Color.BLACK, 13),),
        opened_mode=OpenedMode.SERIES,
        phase=TurnPhase.DRAW_DECISION,
        stock=(stock_tile,),
        discard_pile=(discard_tile,),
        discard_history=(
            DiscardRecord(
                tile=discard_tile,
                player_id=3,
                turn_number=2,
            ),
        ),
        table=table,
    )

    drawn, events = apply_action(base, DrawFromStock(), GameConfig())
    assert drawn.stock == ()
    assert drawn.current_player_state.hand[-1] == stock_tile
    assert drawn.turn_context.stock_exhausted_after_draw
    assert events[0].type is EventType.DRAW_STOCK

    taken, events = apply_action(base, TakePreviousDiscard(), GameConfig())
    assert taken.stock == (stock_tile,)
    assert taken.discard_pile == ()
    assert taken.discard_history[0].tile == discard_tile
    assert taken.discard_history[0].player_id == 3
    assert taken.discard_history[0].taken_by == 0
    assert taken.current_player_state.hand[-1] == discard_tile
    assert taken.turn_context.must_use_taken_discard
    assert events[0].type is EventType.TAKE_DISCARD


def test_taken_discard_must_be_used_by_the_same_player_before_discard_phase() -> None:
    table_meld = build_meld(
        (normal(Color.RED, 2), normal(Color.RED, 3), normal(Color.RED, 4)),
        TileValue(Color.YELLOW, 1),
    )
    table, (meld_id,) = TableState().add_melds((table_meld,))
    base = state_for(
        (normal(Color.BLACK, 13),),
        opened_mode=OpenedMode.SERIES,
        phase=TurnPhase.DRAW_DECISION,
        discard_pile=(normal(Color.RED, 5),),
        table=table,
    )

    taken, _ = apply_action(base, TakePreviousDiscard(), GameConfig())
    with pytest.raises(IllegalAction, match="must be used"):
        apply_action(taken, EndTableActions(), GameConfig())

    used, _ = apply_action(
        taken,
        AddToMeld(
            meld_id=meld_id,
            tiles=meld_tiles(normal(Color.RED, 5)),
            side=AttachmentSide.RIGHT,
        ),
        GameConfig(),
    )
    assert not used.turn_context.must_use_taken_discard
    discard_phase, _ = apply_action(used, EndTableActions(), GameConfig())
    assert discard_phase.phase is TurnPhase.DISCARD


def test_transition_rejects_unusable_previous_discard_even_when_called_directly() -> None:
    dead_discard = normal(Color.BLUE, 11)
    state = state_for(
        (normal(Color.BLACK, 13),),
        opened_mode=OpenedMode.SERIES,
        phase=TurnPhase.DRAW_DECISION,
        discard_pile=(dead_discard,),
    )

    with pytest.raises(IllegalAction, match="cannot be used legally"):
        apply_action(state, TakePreviousDiscard(), GameConfig())


def test_action_tiles_must_match_the_canonical_hand_identity() -> None:
    actual = normal(Color.BLACK, 9)
    forged = PhysicalTile(actual.id, TileKind.NORMAL, Color.RED, 2)
    red_three = normal(Color.RED, 3)
    red_four = normal(Color.RED, 4)
    state = state_for(
        (actual, red_three, red_four, normal(Color.BLUE, 13)),
        okey_value=TileValue(Color.YELLOW, 1),
    )
    forged_meld = build_meld(
        (forged, red_three, red_four),
        state.okey_value,
    )

    with pytest.raises(IllegalAction, match="does not match hand identity"):
        apply_action(
            state,
            OpenMelds((forged_meld,)),
            GameConfig(opening_min_score=9),
        )


def test_opening_uses_only_laid_meld_score_and_updates_progressive_threshold() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    melds = (
        build_meld(
            (normal(Color.RED, 11), normal(Color.RED, 12), normal(Color.RED, 13)),
            okey_value,
        ),
        build_meld(
            (normal(Color.BLUE, 11), normal(Color.BLUE, 12), normal(Color.BLUE, 13)),
            okey_value,
        ),
        build_meld(
            (normal(Color.BLACK, 10), normal(Color.BLACK, 11), normal(Color.BLACK, 12)),
            okey_value,
        ),
    )
    hand = (
        *(tile for meld in melds for tile in meld.physical_tiles),
        normal(Color.YELLOW, 7),
    )
    state = state_for(hand, okey_value=okey_value)

    opened, events = apply_action(
        state,
        OpenMelds(melds),
        GameConfig(progressive_opening=True),
    )

    assert opened.current_player_state.opened_mode is OpenedMode.SERIES
    assert opened.current_player_state.opening_turn == state.turn_number
    assert opened.progressive_series_threshold == 106
    assert opened.progressive_pair_threshold == 5
    assert opened.turn_context.opened_this_turn
    assert events[0].details["score"] == 105

    too_high = replace(state, progressive_series_threshold=106)
    with pytest.raises(IllegalAction, match="below the required"):
        apply_action(too_high, OpenMelds(melds), GameConfig())


def test_opening_scores_real_okey_by_its_represented_value() -> None:
    okey_value = TileValue(Color.RED, 5)
    red_three = normal(Color.RED, 3)
    red_four = normal(Color.RED, 4)
    real_okey = normal(Color.RED, 5)
    meld = Meld(
        MeldKind.RUN,
        (
            MeldTile(red_three, TileValue(Color.RED, 3)),
            MeldTile(red_four, TileValue(Color.RED, 4)),
            MeldTile(real_okey, TileValue(Color.RED, 5)),
        ),
    )
    state = replace(
        state_for(
            (*meld.physical_tiles, normal(Color.BLACK, 13)),
            okey_value=okey_value,
        ),
        progressive_series_threshold=12,
    )

    opened, _events = apply_action(
        state,
        OpenMelds((meld,)),
        GameConfig(opening_min_score=12),
    )

    assert meld.score == 12
    assert opened.current_player_state.opened_mode is OpenedMode.SERIES


def test_pair_opening_and_series_player_can_add_to_existing_pair_area() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    pairs = tuple(
        build_pair(
            (normal(Color.RED, number, 0), normal(Color.RED, number, 1)),
            okey_value,
        )
        for number in range(1, 6)
    )
    opening_hand = (
        *(tile for pair in pairs for tile in pair.physical_tiles),
        normal(Color.BLACK, 13),
    )
    opened, _ = apply_action(
        state_for(opening_hand, okey_value=okey_value),
        OpenPairs(pairs),
        GameConfig(progressive_opening=True),
    )
    assert opened.current_player_state.opened_mode is OpenedMode.PAIRS
    assert len(opened.table.pairs) == 5
    assert opened.progressive_pair_threshold == 6

    added_pair = build_pair(
        (normal(Color.BLUE, 8, 0), normal(Color.BLUE, 8, 1)),
        okey_value,
    )
    series_player = replace(
        opened,
        players=(
            PlayerState(
                hand=(*added_pair.physical_tiles, normal(Color.BLACK, 12)),
                opened_mode=OpenedMode.SERIES,
            ),
            *opened.players[1:],
        ),
        turn_context=TurnContext(opened_mode_at_start=OpenedMode.SERIES),
    )
    added, _ = apply_action(series_player, AddPair(added_pair), GameConfig())
    assert len(added.table.pairs) == 6
    assert len(added.current_player_state.hand) == 1


def test_attachment_limit_is_per_meld_extension_and_resets_with_turn_context() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    red = build_meld(
        (normal(Color.RED, 3), normal(Color.RED, 4), normal(Color.RED, 5)),
        okey_value,
    )
    blue = build_meld(
        (normal(Color.BLUE, 3), normal(Color.BLUE, 4), normal(Color.BLUE, 5)),
        okey_value,
    )
    table, (red_id, blue_id) = TableState().add_melds((red, blue))
    state = state_for(
        (
            normal(Color.RED, 6),
            normal(Color.RED, 7),
            normal(Color.RED, 8),
            normal(Color.BLUE, 6),
            normal(Color.BLACK, 13),
        ),
        opened_mode=OpenedMode.SERIES,
        table=table,
        okey_value=okey_value,
    )

    state, _ = apply_action(
        state,
        AddToMeld(red_id, meld_tiles(normal(Color.RED, 6)), AttachmentSide.RIGHT),
        GameConfig(),
    )
    state, _ = apply_action(
        state,
        AddToMeld(red_id, meld_tiles(normal(Color.RED, 7)), AttachmentSide.RIGHT),
        GameConfig(),
    )
    with pytest.raises(IllegalAction, match="At most 2"):
        apply_action(
            state,
            AddToMeld(red_id, meld_tiles(normal(Color.RED, 8)), AttachmentSide.RIGHT),
            GameConfig(),
        )

    state, _ = apply_action(
        state,
        AddToMeld(blue_id, meld_tiles(normal(Color.BLUE, 6)), AttachmentSide.RIGHT),
        GameConfig(),
    )
    assert len(state.table.meld(red_id).meld.tiles) == 5
    assert len(state.table.meld(blue_id).meld.tiles) == 4


def test_attachment_cannot_reorganize_or_insert_into_an_existing_meld() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    meld = build_meld(
        (normal(Color.RED, 3), normal(Color.RED, 4), normal(Color.RED, 5)),
        okey_value,
    )
    table, (meld_id,) = TableState().add_melds((meld,))
    state = state_for(
        (normal(Color.RED, 2), normal(Color.RED, 7), normal(Color.BLACK, 13)),
        opened_mode=OpenedMode.SERIES,
        table=table,
        okey_value=okey_value,
    )

    with pytest.raises(IllegalAction, match="Invalid meld"):
        apply_action(
            state,
            AddToMeld(meld_id, meld_tiles(normal(Color.RED, 7)), AttachmentSide.RIGHT),
            GameConfig(),
        )


def test_joker_replacement_keeps_assignment_and_returns_real_okey_to_hand() -> None:
    okey_value = TileValue(Color.BLUE, 1)
    real_okey = normal(Color.BLUE, 1)
    meld = build_meld(
        (normal(Color.RED, 2), real_okey, normal(Color.RED, 4)),
        okey_value,
    )
    table, (meld_id,) = TableState().add_melds((meld,))
    replacement = normal(Color.RED, 3)
    state = state_for(
        (replacement, normal(Color.BLACK, 13)),
        opened_mode=OpenedMode.SERIES,
        table=table,
        okey_value=okey_value,
    )

    updated, events = apply_action(
        state,
        ReplaceJoker(meld_id, real_okey.id, replacement.id),
        GameConfig(),
    )

    assert real_okey in updated.current_player_state.hand
    assert replacement not in updated.current_player_state.hand
    table_tiles = updated.table.meld(meld_id).meld.tiles
    replaced = next(tile for tile in table_tiles if tile.physical_tile.id == replacement.id)
    assert replaced.represented_value == TileValue(Color.RED, 3)
    assert events[0].type is EventType.REPLACE_JOKER
    assert updated.phase is TurnPhase.TABLE_ACTIONS


def test_joker_replacement_requires_open_player_and_exact_normal_tile() -> None:
    okey_value = TileValue(Color.BLUE, 1)
    real_okey = normal(Color.BLUE, 1)
    meld = build_meld(
        (normal(Color.RED, 2), real_okey, normal(Color.RED, 4)),
        okey_value,
    )
    table, (meld_id,) = TableState().add_melds((meld,))
    exact = normal(Color.RED, 3)
    unopened = state_for(
        (exact, normal(Color.BLACK, 13)),
        table=table,
        okey_value=okey_value,
    )
    with pytest.raises(IllegalAction, match="must open"):
        apply_action(
            unopened,
            ReplaceJoker(meld_id, real_okey.id, exact.id),
            GameConfig(),
        )

    wrong = normal(Color.RED, 5)
    opened = state_for(
        (wrong, normal(Color.BLACK, 13)),
        opened_mode=OpenedMode.SERIES,
        table=table,
        okey_value=okey_value,
    )
    with pytest.raises(IllegalAction, match="does not match"):
        apply_action(
            opened,
            ReplaceJoker(meld_id, real_okey.id, wrong.id),
            GameConfig(),
        )


def test_recovered_okey_can_be_used_in_a_new_meld_during_same_turn() -> None:
    okey_value = TileValue(Color.BLUE, 1)
    real_okey = normal(Color.BLUE, 1)
    table_meld = build_meld(
        (normal(Color.RED, 2), real_okey, normal(Color.RED, 4)),
        okey_value,
    )
    table, (meld_id,) = TableState().add_melds((table_meld,))
    replacement = normal(Color.RED, 3)
    blue_five = normal(Color.BLUE, 5)
    blue_six = normal(Color.BLUE, 6)
    state = state_for(
        (
            replacement,
            blue_five,
            blue_six,
            normal(Color.BLACK, 13),
        ),
        opened_mode=OpenedMode.SERIES,
        table=table,
        okey_value=okey_value,
    )
    replaced, _events = apply_action(
        state,
        ReplaceJoker(meld_id, real_okey.id, replacement.id),
        GameConfig(),
    )
    new_meld = build_meld(
        (blue_five, blue_six, real_okey),
        okey_value,
    )

    updated, _events = apply_action(
        replaced,
        OpenMelds((new_meld,)),
        GameConfig(),
    )

    assert real_okey not in updated.current_player_state.hand
    assert len(updated.table.melds) == 2


def test_final_discard_is_required_and_determines_explicit_finish_reason() -> None:
    final_tile = normal(Color.BLACK, 11)
    state = state_for((final_tile,), phase=TurnPhase.DISCARD)
    finished, events = apply_action(state, Discard(final_tile.id), GameConfig())

    assert finished.terminal
    assert finished.terminal_reason is TerminalReason.NORMAL_FINISH
    assert finished.winner == 0
    assert [event.type for event in events] == [
        EventType.DISCARD,
        EventType.FINISH,
        EventType.ROUND_END,
    ]
    with pytest.raises(IllegalAction, match="Terminal state"):
        apply_action(finished, Discard(final_tile.id), GameConfig())

    meld = build_meld(
        (normal(Color.RED, 2), normal(Color.RED, 3), normal(Color.RED, 4)),
        TileValue(Color.YELLOW, 1),
    )
    with pytest.raises(IllegalAction, match="final discard"):
        apply_action(
            replace(state_for(meld.physical_tiles), progressive_series_threshold=9),
            OpenMelds((meld,)),
            GameConfig(opening_min_score=9),
        )


def test_final_okey_discard_has_combined_same_turn_reason_and_no_penalty() -> None:
    okey_value = TileValue(Color.BLUE, 1)
    final_okey = normal(Color.BLUE, 1)
    state = state_for(
        (final_okey,),
        opened_mode=OpenedMode.SERIES,
        phase=TurnPhase.DISCARD,
        okey_value=okey_value,
        turn_context=TurnContext(
            opened_mode_at_start=OpenedMode.NONE,
            opened_this_turn=True,
        ),
    )
    finished, events = apply_action(state, Discard(final_okey.id), GameConfig())

    assert finished.terminal_reason is TerminalReason.SAME_TURN_OPEN_OKEY_FINISH
    assert finished.current_player_state.immediate_penalty == 0
    assert not any(event.type is EventType.PENALTY for event in events)


@pytest.mark.parametrize(
    ("discard_okey", "expected_reason"),
    (
        (False, TerminalReason.PAIR_FINISH),
        (True, TerminalReason.PAIR_OKEY_FINISH),
    ),
)
def test_pair_opened_final_discard_has_explicit_finish_reason(
    discard_okey: bool,
    expected_reason: TerminalReason,
) -> None:
    okey_value = TileValue(Color.BLUE, 1)
    final_tile = (
        normal(Color.BLUE, 1)
        if discard_okey
        else normal(Color.BLACK, 13)
    )
    players = (
        PlayerState(hand=(final_tile,), opened_mode=OpenedMode.PAIRS),
        PlayerState(),
        PlayerState(),
        PlayerState(),
    )
    state = state_for(
        (final_tile,),
        players=players,
        phase=TurnPhase.DISCARD,
        okey_value=okey_value,
    )

    finished, _events = apply_action(
        state,
        Discard(final_tile.id),
        GameConfig(),
    )

    assert finished.terminal_reason is expected_reason
    assert finished.current_player_state.immediate_penalty == 0


def test_nonfinal_playable_discard_is_legal_and_accumulates_immediate_penalty() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    meld = build_meld(
        (normal(Color.RED, 2), normal(Color.RED, 3), normal(Color.RED, 4)),
        okey_value,
    )
    table, _ = TableState().add_melds((meld,))
    playable = normal(Color.RED, 5)
    state = state_for(
        (playable, normal(Color.BLACK, 13)),
        opened_mode=OpenedMode.SERIES,
        phase=TurnPhase.DISCARD,
        table=table,
        okey_value=okey_value,
    )

    updated, events = apply_action(state, Discard(playable.id), GameConfig())

    assert not updated.terminal
    assert updated.current_player == 1
    assert updated.players[0].immediate_penalty == 101
    assert any(event.type is EventType.PENALTY for event in events)


def test_transition_penalties_accumulate_across_multiple_events() -> None:
    okey_value = TileValue(Color.BLUE, 1)
    real_okey = normal(Color.BLUE, 1)
    players = (
        PlayerState(
            hand=(real_okey, normal(Color.BLACK, 13)),
            opened_mode=OpenedMode.SERIES,
            immediate_penalty=101,
        ),
        PlayerState(),
        PlayerState(),
        PlayerState(),
    )
    state = state_for(
        players[0].hand,
        players=players,
        phase=TurnPhase.DISCARD,
        okey_value=okey_value,
    )

    updated, events = apply_action(
        state,
        Discard(real_okey.id),
        GameConfig(),
    )

    assert updated.players[0].immediate_penalty == 202
    assert any(event.type is EventType.PENALTY for event in events)


def test_last_stock_tile_drawn_allows_turn_then_ends_after_nonfinal_discard() -> None:
    okey_value = TileValue(Color.BLUE, 1)
    real_okey = normal(Color.BLUE, 1)
    state = state_for(
        (normal(Color.BLACK, 13),),
        opened_mode=OpenedMode.SERIES,
        phase=TurnPhase.DRAW_DECISION,
        stock=(real_okey,),
        okey_value=okey_value,
    )

    state, _ = apply_action(state, DrawFromStock(), GameConfig())
    assert not state.terminal
    assert state.phase is TurnPhase.TABLE_ACTIONS
    state, _ = apply_action(state, EndTableActions(), GameConfig())
    state, events = apply_action(state, Discard(real_okey.id), GameConfig())

    assert state.terminal
    assert state.terminal_reason is TerminalReason.STOCK_EXHAUSTED
    assert state.winner is None
    assert state.current_player_state.immediate_penalty == 101
    assert events[-1].type is EventType.ROUND_END


def test_fourth_pair_opening_voids_round_without_turn_advance() -> None:
    okey_value = TileValue(Color.YELLOW, 1)
    pairs = tuple(
        build_pair(
            (normal(Color.BLUE, number, 0), normal(Color.BLUE, number, 1)),
            okey_value,
        )
        for number in range(2, 7)
    )
    hand = (
        *(tile for pair in pairs for tile in pair.physical_tiles),
        normal(Color.BLACK, 13),
    )
    players = (
        PlayerState(hand=hand),
        PlayerState(opened_mode=OpenedMode.PAIRS),
        PlayerState(opened_mode=OpenedMode.PAIRS),
        PlayerState(opened_mode=OpenedMode.PAIRS),
    )
    state = state_for(hand, players=players, okey_value=okey_value)

    finished, events = apply_action(state, OpenPairs(pairs), GameConfig())

    assert finished.terminal
    assert finished.terminal_reason is TerminalReason.ALL_PLAYERS_OPENED_PAIRS
    assert finished.winner is None
    assert [event.type for event in events] == [
        EventType.OPEN_PAIRS,
        EventType.ROUND_END,
    ]
