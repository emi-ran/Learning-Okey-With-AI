from __future__ import annotations

from dataclasses import replace

import pytest

from okey101.engine.invariants import (
    InvariantError,
    find_invariant_violations,
    validate_invariants,
)
from okey101.engine.joker import okey_value_for_indicator
from okey101.engine.player import PlayerState
from okey101.engine.state import GameState, TerminalReason, TurnPhase
from okey101.engine.tiles import build_tile_set


def conserved_state() -> GameState:
    tiles = build_tile_set()
    indicator = tiles[0]
    return GameState(
        round_id=1,
        turn_number=0,
        current_player=0,
        starting_player=0,
        indicator=indicator,
        okey_value=okey_value_for_indicator(indicator),
        stock=tiles[1:],
        discard_pile=(),
        players=(PlayerState(), PlayerState(), PlayerState(), PlayerState()),
        phase=TurnPhase.DRAW_DECISION,
    )


def test_valid_state_conserves_all_106_physical_tiles() -> None:
    validate_invariants(conserved_state())


def test_duplicate_tile_reports_both_owning_locations_and_missing_count() -> None:
    state = conserved_state()
    duplicated = state.stock[0]
    broken = state.replace_player(0, PlayerState(hand=(duplicated,)))

    violations = find_invariant_violations(broken)
    duplicate = next(
        violation
        for violation in violations
        if violation.code == "DUPLICATE_TILE_ID"
    )

    assert "stock[0]" in duplicate.message
    assert "players[0].hand[0]" in duplicate.message
    with pytest.raises(InvariantError, match="DUPLICATE_TILE_ID"):
        validate_invariants(broken)


def test_terminal_state_requires_terminal_phase_reason_and_winner_consistency() -> None:
    state = conserved_state()
    broken = replace(
        state,
        terminal=True,
        terminal_reason=TerminalReason.NORMAL_FINISH,
        phase=TurnPhase.DRAW_DECISION,
        winner=None,
    )

    codes = {
        violation.code
        for violation in find_invariant_violations(broken)
    }

    assert {"TERMINAL_PHASE", "MISSING_WINNER"} <= codes
