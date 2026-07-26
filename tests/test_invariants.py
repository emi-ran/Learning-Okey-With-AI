from __future__ import annotations

from dataclasses import replace

import pytest

from okey101.engine.invariants import (
    InvariantError,
    find_invariant_violations,
    validate_invariants,
)
from okey101.engine.joker import okey_value_for_indicator
from okey101.engine.melds import Meld, MeldKind, MeldTile
from okey101.engine.player import PlayerState
from okey101.engine.state import (
    DiscardRecord,
    GameState,
    TerminalReason,
    TurnPhase,
)
from okey101.engine.table import TableState
from okey101.engine.tiles import TileValue, build_tile_set


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


def test_discard_history_is_a_nonowning_but_validated_reference() -> None:
    state = conserved_state()
    valid = replace(
        state,
        discard_history=(
            DiscardRecord(
                tile=state.stock[0],
                player_id=1,
                turn_number=0,
                taken_by=2,
            ),
        ),
    )
    validate_invariants(valid)

    broken = replace(
        state,
        discard_history=(
            DiscardRecord(
                tile=replace(state.stock[0], id=999),
                player_id=8,
                turn_number=2,
            ),
        ),
    )
    codes = {
        violation.code for violation in find_invariant_violations(broken)
    }

    assert {
        "UNKNOWN_DISCARD_HISTORY_TILE",
        "INVALID_DISCARD_HISTORY_PLAYER",
        "FUTURE_DISCARD_HISTORY_TURN",
    } <= codes


def test_physical_id_cannot_be_rebound_to_a_different_tile_value() -> None:
    state = conserved_state()
    canonical = state.stock[0]
    assert canonical.number is not None
    forged = replace(
        canonical,
        number=canonical.number % 13 + 1,
    )
    broken = replace(state, stock=(forged, *state.stock[1:]))

    codes = {
        violation.code for violation in find_invariant_violations(broken)
    }

    assert "PHYSICAL_TILE_IDENTITY" in codes


def test_table_meld_shape_and_stored_assignments_are_invariants() -> None:
    state = conserved_state()
    first, second, third = state.stock[:3]
    assert first.value is not None
    assert second.value is not None
    assert third.value is not None
    invalid = Meld(
        MeldKind.RUN,
        (
            MeldTile(first, first.value),
            MeldTile(second, second.value),
            MeldTile(
                third,
                TileValue(third.value.color, 13),
            ),
        ),
    )
    table, _ids = TableState().add_melds((invalid,))
    relocated = {first.id, second.id, third.id}
    broken = replace(
        state,
        stock=tuple(tile for tile in state.stock if tile.id not in relocated),
        table=table,
    )

    codes = {
        violation.code for violation in find_invariant_violations(broken)
    }

    assert "INVALID_TABLE_MELD" in codes
