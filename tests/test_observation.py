from __future__ import annotations

from dataclasses import replace

from okey101.engine.joker import okey_value_for_indicator
from okey101.engine.player import OpenedMode, PlayerState
from okey101.engine.state import DiscardRecord, GameState
from okey101.engine.tiles import Color, PhysicalTile, TileKind
from okey101.rl.observation import get_observation


def normal(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def state_with_hidden_hands() -> GameState:
    indicator = normal(0, Color.RED, 4)
    return GameState(
        round_id=1,
        turn_number=3,
        current_player=0,
        starting_player=0,
        indicator=indicator,
        okey_value=okey_value_for_indicator(indicator),
        stock=(normal(90, Color.BLACK, 13),),
        discard_pile=(normal(80, Color.YELLOW, 7),),
        players=(
            PlayerState(hand=(normal(1, Color.RED, 1),)),
            PlayerState(hand=(normal(2, Color.BLUE, 2),)),
            PlayerState(hand=(normal(3, Color.BLACK, 3),)),
            PlayerState(hand=(normal(4, Color.YELLOW, 4),)),
        ),
    )


def test_observation_excludes_opponent_hands_and_stock_identities() -> None:
    state = state_with_hidden_hands()
    observation = get_observation(state, 0)

    assert observation.own_tile_ids == (1,)
    assert observation.stock_count == 1
    assert not hasattr(observation, "stock")
    assert not hasattr(observation, "player_id")
    assert not hasattr(observation.player_statuses[1], "hand")


def test_opponent_hidden_hand_changes_do_not_change_observation() -> None:
    state = state_with_hidden_hands()
    changed_opponent = replace(
        state.players[1],
        hand=(normal(72, Color.RED, 13), normal(73, Color.BLUE, 13)),
    )
    changed = state.replace_player(1, changed_opponent)

    assert get_observation(state, 0) == get_observation(changed, 0)


def test_statuses_are_relative_and_public() -> None:
    state = state_with_hidden_hands()
    opened = replace(
        state.players[0],
        opened_mode=OpenedMode.SERIES,
        immediate_penalty=101,
        score=303,
    )
    state = state.replace_player(0, opened)

    observation = get_observation(state, 3)

    assert tuple(status.relative_seat for status in observation.player_statuses) == (
        0,
        1,
        2,
        3,
    )
    self_status, left_status = observation.player_statuses[:2]
    assert self_status.opened_mode is OpenedMode.NONE
    assert left_status.opened_mode is OpenedMode.SERIES
    assert left_status.immediate_penalty == 101
    assert left_status.score == 303


def test_discard_history_retains_discarder_and_taker_relative_to_viewer() -> None:
    state = state_with_hidden_hands()
    discarded = normal(81, Color.BLUE, 8)
    state = replace(
        state,
        discard_history=(
            DiscardRecord(
                tile=discarded,
                player_id=2,
                turn_number=1,
                taken_by=3,
            ),
        ),
    )

    record = get_observation(state, 1).discard_history[0]

    assert record.tile.tile_id == discarded.id
    assert record.player_relative == 1
    assert record.turn_number == 1
    assert record.taken_by_relative == 2
