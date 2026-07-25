from __future__ import annotations

from okey101.engine.actions import (
    AddToMeld,
    Discard,
    OpenMelds,
    TakePreviousDiscard,
)
from okey101.engine.config import RulesConfig
from okey101.engine.legal_actions import (
    get_legal_actions,
    is_playable_discard,
)
from okey101.engine.melds import build_meld
from okey101.engine.player import OpenedMode, PlayerState
from okey101.engine.state import (
    DrawSource,
    GameState,
    TurnContext,
    TurnPhase,
)
from okey101.engine.table import TableMeld, TableState
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue


def tile(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def state_for(
    hand: tuple[PhysicalTile, ...],
    *,
    phase: TurnPhase,
    opened_mode: OpenedMode = OpenedMode.NONE,
    discard_pile: tuple[PhysicalTile, ...] = (),
    table: TableState | None = None,
    context: TurnContext | None = None,
) -> GameState:
    return GameState(
        round_id=1,
        turn_number=1,
        current_player=0,
        starting_player=0,
        indicator=tile(900, Color.YELLOW, 13),
        okey_value=TileValue(Color.YELLOW, 1),
        stock=(tile(901, Color.BLACK, 13),),
        discard_pile=discard_pile,
        players=(PlayerState(hand=hand, opened_mode=opened_mode),),
        table=table or TableState(),
        progressive_series_threshold=9,
        phase=phase,
        turn_context=context or TurnContext(opened_mode_at_start=opened_mode),
    )


def test_discard_phase_keeps_every_discard_legal_even_when_penalized() -> None:
    hand = (
        tile(1, Color.RED, 5),
        tile(2, Color.BLUE, 8),
        tile(3, Color.YELLOW, 1),
    )

    actions = get_legal_actions(
        state_for(hand, phase=TurnPhase.DISCARD),
        RulesConfig(),
    )

    assert actions == tuple(Discard(candidate.id) for candidate in hand)


def test_opening_generator_meets_threshold_and_keeps_final_discard() -> None:
    hand = (
        tile(10, Color.RED, 2),
        tile(11, Color.RED, 3),
        tile(12, Color.RED, 4),
        tile(13, Color.BLACK, 9),
    )

    actions = get_legal_actions(
        state_for(hand, phase=TurnPhase.TABLE_ACTIONS),
        RulesConfig(opening_min_score=9),
    )
    openings = [action for action in actions if isinstance(action, OpenMelds)]

    assert openings
    assert all(sum(meld.score for meld in action.melds) >= 9 for action in openings)
    assert all(
        sum(len(meld.tiles) for meld in action.melds) < len(hand)
        for action in openings
    )


def test_previous_discard_is_legal_only_when_it_can_be_used_this_turn() -> None:
    table_meld = build_meld(
        (
            tile(20, Color.RED, 2),
            tile(21, Color.RED, 3),
            tile(22, Color.RED, 4),
        ),
        TileValue(Color.YELLOW, 1),
    )
    table = TableState(melds=(TableMeld(0, table_meld),), next_meld_id=1)
    hand = (tile(23, Color.BLACK, 9),)
    playable = tile(24, Color.RED, 5)
    dead = tile(25, Color.BLUE, 11)

    playable_actions = get_legal_actions(
        state_for(
            hand,
            phase=TurnPhase.DRAW_DECISION,
            opened_mode=OpenedMode.SERIES,
            discard_pile=(playable,),
            table=table,
        ),
        RulesConfig(),
    )
    dead_actions = get_legal_actions(
        state_for(
            hand,
            phase=TurnPhase.DRAW_DECISION,
            opened_mode=OpenedMode.SERIES,
            discard_pile=(dead,),
            table=table,
        ),
        RulesConfig(),
    )

    assert TakePreviousDiscard() in playable_actions
    assert TakePreviousDiscard() not in dead_actions


def test_taken_discard_constraint_exposes_only_actions_that_can_progress_usage() -> None:
    table_meld = build_meld(
        (
            tile(30, Color.RED, 2),
            tile(31, Color.RED, 3),
            tile(32, Color.RED, 4),
        ),
        TileValue(Color.YELLOW, 1),
    )
    taken = tile(33, Color.RED, 5)
    context = TurnContext(
        draw_source=DrawSource.PREVIOUS_DISCARD,
        drawn_tile_id=taken.id,
        taken_discard_tile_id=taken.id,
        opened_mode_at_start=OpenedMode.SERIES,
    )
    state = state_for(
        (taken, tile(34, Color.BLACK, 9)),
        phase=TurnPhase.TABLE_ACTIONS,
        opened_mode=OpenedMode.SERIES,
        table=TableState(melds=(TableMeld(0, table_meld),), next_meld_id=1),
        context=context,
    )

    actions = get_legal_actions(state, RulesConfig())

    assert any(
        isinstance(action, AddToMeld)
        and any(candidate.physical_tile.id == taken.id for candidate in action.tiles)
        for action in actions
    )
    assert all(action.type.value != "end_table_actions" for action in actions)
    assert is_playable_discard(state, taken)
