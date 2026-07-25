from __future__ import annotations

from dataclasses import replace

import pytest

from okey101.engine.actions import Discard, OpenPairs
from okey101.engine.config import GameConfig
from okey101.engine.joker import is_real_okey
from okey101.engine.match import MatchEngine, derive_round_seed
from okey101.engine.pairs import build_pair
from okey101.engine.player import OpenedMode, PlayerState
from okey101.engine.state import EventType, TerminalReason, TurnContext, TurnPhase
from okey101.engine.tiles import Color, PhysicalTile, TileKind


def _force_normal_finish(match: MatchEngine):
    state = match.current_round
    winner = state.starting_player
    final_tile = next(
        tile
        for tile in state.players[winner].hand
        if not is_real_okey(tile, state.okey_value)
    )
    players = list(state.players)
    players[winner] = PlayerState(
        hand=(final_tile,),
        opened_mode=OpenedMode.SERIES,
        opening_turn=0,
    )
    match.round_engine.state = replace(
        state,
        current_player=winner,
        players=tuple(players),
        phase=TurnPhase.DISCARD,
        turn_context=TurnContext(opened_mode_at_start=OpenedMode.SERIES),
    )
    return match.step(Discard(final_tile.id))


def _force_all_pairs_void(match: MatchEngine):
    state = match.current_round
    current = state.starting_player
    next_id = 2000 + state.round_id * 100
    pairs = []
    physical_tiles = []
    for number in range(1, 6):
        first = PhysicalTile(next_id, TileKind.NORMAL, Color.BLUE, number)
        second = PhysicalTile(next_id + 1, TileKind.NORMAL, Color.BLUE, number)
        next_id += 2
        physical_tiles.extend((first, second))
        pairs.append(build_pair((first, second), state.okey_value))
    spare = PhysicalTile(next_id, TileKind.NORMAL, Color.BLACK, 13)

    players = [
        PlayerState(opened_mode=OpenedMode.PAIRS)
        for _ in state.players
    ]
    players[current] = PlayerState(hand=(*physical_tiles, spare))
    match.round_engine.state = replace(
        state,
        current_player=current,
        players=tuple(players),
        phase=TurnPhase.TABLE_ACTIONS,
        progressive_pair_threshold=5,
        turn_context=TurnContext(opened_mode_at_start=OpenedMode.NONE),
    )
    return match.step(OpenPairs(tuple(pairs)))


def test_match_rotates_starting_player_accumulates_scores_and_auto_deals() -> None:
    match = MatchEngine(GameConfig(rounds=2))
    first = match.reset(seed=87, starting_player=2)

    assert first.round_id == 1
    assert first.starting_player == 2
    assert match.current_round_seed == derive_round_seed(87, 0)

    second, events = _force_normal_finish(match)

    assert not match.is_terminal()
    assert match.completed_rounds == 1
    assert second.round_id == 2
    assert second.starting_player == 3
    assert match.current_round_seed == derive_round_seed(87, 1)
    assert events[-2].type is EventType.ROUND_END
    assert events[-1].type is EventType.DEAL
    assert match.get_scores() == (202, 202, -101, 202)
    assert tuple(player.score for player in second.players) == match.get_scores()

    terminal, events = _force_normal_finish(match)

    assert match.is_terminal()
    assert terminal.terminal_reason is TerminalReason.NORMAL_FINISH
    assert events[-1].type is EventType.ROUND_END
    assert match.get_scores() == (404, 404, 101, 101)
    assert tuple(player.score for player in terminal.players) == match.get_scores()
    assert [record.starting_player for record in match.round_records] == [2, 3]
    assert all(record.counts_toward_match for record in match.round_records)
    assert match.get_legal_actions() == ()
    with pytest.raises(RuntimeError, match="Terminal match"):
        match.step(Discard(0))


def test_same_seed_and_policy_reproduce_round_states_and_seed_sequence() -> None:
    config = GameConfig(rounds=2)
    first = MatchEngine(config)
    second = MatchEngine(config)

    assert first.reset(seed=-41, starting_player=1) == second.reset(
        seed=-41,
        starting_player=1,
    )
    first_next, _ = _force_normal_finish(first)
    second_next, _ = _force_normal_finish(second)

    assert first_next == second_next
    assert first.current_round_seed == second.current_round_seed
    assert first.round_records == second.round_records


def test_generated_seed_is_exposed_and_can_replay_the_initial_round() -> None:
    match = MatchEngine()
    initial = match.reset()
    generated_seed = match.seed

    assert generated_seed is not None
    replayed = MatchEngine().reset(seed=generated_seed)
    assert replayed == initial


def test_void_round_scores_zero_and_redeals_without_consuming_round_quota() -> None:
    match = MatchEngine(GameConfig(rounds=1))
    match.reset(seed=12, starting_player=0)

    redealt, events = _force_all_pairs_void(match)

    assert not match.is_terminal()
    assert match.completed_rounds == 0
    assert match.rounds_played == 1
    assert match.round_records[0].terminal_reason is TerminalReason.ALL_PLAYERS_OPENED_PAIRS
    assert not match.round_records[0].counts_toward_match
    assert match.round_records[0].scores == (0, 0, 0, 0)
    assert match.get_scores() == (0, 0, 0, 0)
    assert redealt.round_id == 2
    assert redealt.starting_player == 1
    assert events[-1].type is EventType.DEAL

    _force_normal_finish(match)
    assert match.is_terminal()
    assert match.completed_rounds == 1
    assert match.get_scores() == (202, -101, 202, 202)


def test_config_can_make_void_round_consume_match_quota() -> None:
    match = MatchEngine(
        GameConfig(rounds=1, void_round_counts_toward_match=True)
    )
    match.reset(seed=12)

    terminal, _ = _force_all_pairs_void(match)

    assert match.is_terminal()
    assert match.completed_rounds == 1
    assert terminal.terminal_reason is TerminalReason.ALL_PLAYERS_OPENED_PAIRS
    assert match.get_scores() == (0, 0, 0, 0)


def test_reset_clears_match_totals_history_and_terminal_state() -> None:
    match = MatchEngine(GameConfig(rounds=1))
    match.reset(seed=3)
    _force_normal_finish(match)
    assert match.is_terminal()

    reset_state = match.reset(seed=3)

    assert not match.is_terminal()
    assert match.completed_rounds == 0
    assert match.round_records == []
    assert match.action_history == []
    assert match.get_scores() == (0, 0, 0, 0)
    assert reset_state.round_id == 1


def test_match_requires_reset_and_round_seed_derivation_validates_inputs() -> None:
    match = MatchEngine()

    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        _ = match.current_round
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        match.get_scores()
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        match.step(Discard(0))

    assert derive_round_seed(5, 0) == derive_round_seed(5, 0)
    assert derive_round_seed(5, 0) != derive_round_seed(5, 1)
    with pytest.raises(ValueError, match="negative"):
        derive_round_seed(5, -1)
    with pytest.raises(TypeError, match="integer"):
        derive_round_seed(True, 0)
