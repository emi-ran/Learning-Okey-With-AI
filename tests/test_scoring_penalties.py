from __future__ import annotations

from dataclasses import replace

import pytest

from okey101.engine.config import GameConfig, ScoringConfig
from okey101.engine.penalties import calculate_discard_penalty
from okey101.engine.player import OpenedMode, PlayerState
from okey101.engine.scoring import calculate_round_scores, score_round
from okey101.engine.state import GameState, TerminalReason, TurnPhase
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue


def normal(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def terminal_state(
    players: tuple[PlayerState, ...],
    reason: TerminalReason,
    *,
    winner: int | None,
) -> GameState:
    indicator = normal(1000, Color.RED, 4)
    return GameState(
        round_id=1,
        turn_number=9,
        current_player=winner or 0,
        starting_player=0,
        indicator=indicator,
        okey_value=TileValue(Color.RED, 5),
        stock=(),
        discard_pile=(),
        players=players,
        phase=TurnPhase.TERMINAL,
        terminal=True,
        terminal_reason=reason,
        winner=winner,
    )


def test_stock_exhaustion_scores_each_opening_mode_and_keeps_immediate_penalties() -> None:
    real_okey = normal(1, Color.RED, 5)
    fake_okey = PhysicalTile(2, TileKind.FAKE_OKEY)
    players = (
        PlayerState(
            hand=(real_okey,),
            opened_mode=OpenedMode.NONE,
            immediate_penalty=101,
        ),
        PlayerState(
            hand=(normal(3, Color.BLUE, 8), fake_okey, real_okey),
            opened_mode=OpenedMode.SERIES,
            immediate_penalty=7,
        ),
        PlayerState(
            hand=(normal(4, Color.BLACK, 9), normal(5, Color.YELLOW, 11), real_okey),
            opened_mode=OpenedMode.PAIRS,
        ),
    )
    result = score_round(
        terminal_state(players, TerminalReason.STOCK_EXHAUSTED, winner=None),
        GameConfig(),
    )

    assert result.totals == (303, 8 + 5 + 101 + 7, (9 + 11) * 2 + 101)
    assert result.players[1].hand is not None
    assert result.players[1].hand.normal_value_sum == 13
    assert result.players[1].hand.okey_count == 1


def test_okey_finish_multiplies_hand_components_but_not_default_surcharge() -> None:
    okey = normal(8, Color.RED, 5)
    players = (
        PlayerState(immediate_penalty=3, opened_mode=OpenedMode.SERIES),
        PlayerState(
            hand=(normal(9, Color.BLUE, 10), normal(10, Color.BLACK, 10), okey),
            opened_mode=OpenedMode.PAIRS,
        ),
        PlayerState(hand=(okey,), opened_mode=OpenedMode.NONE),
    )

    scores = calculate_round_scores(
        terminal_state(players, TerminalReason.OKEY_FINISH, winner=0),
        GameConfig(),
    )

    assert scores == (-199, 20 * 2 * 2 + 101, 202 * 2)


def test_same_turn_open_finish_is_not_silently_scored_as_elden_finish() -> None:
    players = (
        PlayerState(opened_mode=OpenedMode.SERIES),
        PlayerState(opened_mode=OpenedMode.NONE),
    )

    scores = calculate_round_scores(
        terminal_state(
            players,
            TerminalReason.SAME_TURN_OPEN_FINISH,
            winner=0,
        ),
        ScoringConfig(),
    )

    assert scores == (-202, 202)


@pytest.mark.parametrize(
    ("reason", "winner_score", "opponent_multiplier"),
    (
        (TerminalReason.PAIR_FINISH, -202, 2),
        (TerminalReason.PAIR_OKEY_FINISH, -404, 4),
    ),
)
def test_pair_finish_score_matrix(
    reason: TerminalReason,
    winner_score: int,
    opponent_multiplier: int,
) -> None:
    players = (
        PlayerState(opened_mode=OpenedMode.PAIRS),
        PlayerState(
            hand=(normal(30, Color.BLUE, 10),),
            opened_mode=OpenedMode.SERIES,
        ),
        PlayerState(opened_mode=OpenedMode.NONE),
    )

    scores = calculate_round_scores(
        terminal_state(players, reason, winner=0),
        ScoringConfig(),
    )

    assert scores == (
        winner_score,
        10 * opponent_multiplier,
        202 * opponent_multiplier,
    )


def test_okey_surcharge_multiplier_flags_are_independent() -> None:
    player = PlayerState(
        hand=(normal(11, Color.BLUE, 10), normal(12, Color.RED, 5)),
        opened_mode=OpenedMode.PAIRS,
    )
    config = ScoringConfig(
        multiply_okey_in_hand_surcharge_by_pair=True,
        multiply_okey_in_hand_surcharge_by_finish=True,
    )
    state = terminal_state(
        (PlayerState(), player),
        TerminalReason.OKEY_FINISH,
        winner=0,
    )

    assert calculate_round_scores(state, config)[1] == 10 * 2 * 2 + 101 * 2 * 2


def test_opened_hand_okey_surcharge_is_fixed_even_with_both_okeys() -> None:
    player = PlayerState(
        hand=(
            normal(13, Color.RED, 5),
            normal(14, Color.RED, 5),
        ),
        opened_mode=OpenedMode.SERIES,
    )
    state = terminal_state(
        (player,),
        TerminalReason.STOCK_EXHAUSTED,
        winner=None,
    )

    result = score_round(state, ScoringConfig())

    assert result.totals == (101,)
    assert result.players[0].hand is not None
    assert result.players[0].hand.okey_count == 2


def test_all_pair_open_round_is_void() -> None:
    players = tuple(
        PlayerState(opened_mode=OpenedMode.PAIRS, immediate_penalty=101)
        for _ in range(4)
    )
    state = terminal_state(
        players,
        TerminalReason.ALL_PLAYERS_OPENED_PAIRS,
        winner=None,
    )

    assert calculate_round_scores(state, GameConfig()) == (0, 0, 0, 0)


def test_discard_penalty_checks_finish_before_okey_and_playability() -> None:
    okey = normal(20, Color.RED, 5)
    state = terminal_state(
        (PlayerState(hand=(okey,)),),
        TerminalReason.STOCK_EXHAUSTED,
        winner=None,
    )
    state = replace(
        state,
        terminal=False,
        terminal_reason=None,
        phase=TurnPhase.DISCARD,
    )
    config = GameConfig(
        scoring=ScoringConfig(
            normal_okey_discard_penalty=37,
            playable_discard_penalty=41,
        )
    )

    assert calculate_discard_penalty(
        state,
        okey,
        is_final=True,
        is_playable=True,
        config=config,
    ) == 0
    assert calculate_discard_penalty(
        state,
        okey,
        is_final=False,
        is_playable=True,
        config=config,
    ) == 37
    assert calculate_discard_penalty(
        state,
        normal(21, Color.BLUE, 7),
        is_final=False,
        is_playable=True,
        config=config,
    ) == 41
