from __future__ import annotations

from dataclasses import fields, replace

from okey101.engine.config import GameConfig, ScoringConfig
from okey101.engine.joker import okey_value_for_indicator
from okey101.engine.melds import build_meld
from okey101.engine.player import PlayerState
from okey101.engine.state import (
    AttachmentUsage,
    DiscardRecord,
    DrawSource,
    GameState,
    TurnContext,
)
from okey101.engine.table import AttachmentSide, TableMeld, TableState
from okey101.engine.tiles import Color, PhysicalTile, TileKind
from okey101.rl.encoder import (
    GLOBAL_CATEGORICAL_NAMES,
    GLOBAL_CONTINUOUS_NAMES,
    HAND_VALUE_COUNT,
    MAX_PLAYERS,
    MAX_RECENT_DISCARD_EVENTS,
    MAX_TABLE_GROUPS,
    MAX_TABLE_TILES,
    OBS_SCHEMA_VERSION,
    RULE_CONTINUOUS_NAMES,
    RULE_FLAG_NAMES,
    TURN_FLAG_NAMES,
    encode_observation,
    tile_token,
)
from okey101.rl.observation import get_observation


def normal(tile_id: int, color: Color, number: int) -> PhysicalTile:
    return PhysicalTile(tile_id, TileKind.NORMAL, color, number)


def base_state() -> GameState:
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


def test_v1_shapes_padding_and_tokens_are_stable() -> None:
    encoded = encode_observation(get_observation(base_state(), 0), GameConfig())

    assert encoded.schema_version == OBS_SCHEMA_VERSION
    assert len(encoded.hand_counts) == HAND_VALUE_COUNT
    assert len(encoded.global_continuous) == len(GLOBAL_CONTINUOUS_NAMES)
    assert len(encoded.global_categorical) == len(GLOBAL_CATEGORICAL_NAMES)
    assert len(encoded.turn_flags) == len(TURN_FLAG_NAMES)
    assert len(encoded.rule_continuous) == len(RULE_CONTINUOUS_NAMES)
    assert len(encoded.rule_flags) == len(RULE_FLAG_NAMES)
    assert len(encoded.player_modes) == MAX_PLAYERS
    assert len(encoded.table_group_kinds) == MAX_TABLE_GROUPS
    assert len(encoded.table_tile_tokens) == MAX_TABLE_TILES
    assert len(encoded.recent_discard_tile_tokens) == MAX_RECENT_DISCARD_EVENTS
    assert encoded.player_mask == (1, 1, 1, 1)
    assert not any(encoded.table_group_mask)
    assert not any(encoded.table_tile_mask)
    assert encoded.hand_counts[0] == 0.5
    assert encoded.global_categorical[3] == 4
    assert encoded.global_categorical[4] == 5


def test_hidden_tile_identity_and_value_changes_do_not_leak() -> None:
    state = base_state()
    changed = state.replace_player(
        1,
        replace(state.players[1], hand=(normal(72, Color.RED, 13),)),
    )
    changed = replace(
        changed,
        stock=(normal(91, Color.YELLOW, 12),),
    )

    before = encode_observation(get_observation(state, 0), GameConfig())
    after = encode_observation(get_observation(changed, 0), GameConfig())

    assert before == after


def test_physical_copy_ids_are_not_neural_features() -> None:
    state = base_state()
    changed = state.replace_player(
        0,
        replace(state.players[0], hand=(normal(71, Color.RED, 1),)),
    )

    before_observation = get_observation(state, 0)
    after_observation = get_observation(changed, 0)
    assert before_observation.own_tile_ids != after_observation.own_tile_ids
    assert encode_observation(before_observation, GameConfig()) == encode_observation(
        after_observation,
        GameConfig(),
    )

    assert all(
        "id" not in field.name
        for field in fields(type(encode_observation(before_observation, GameConfig())))
    )


def test_public_hand_count_is_encoded_relative_to_self() -> None:
    state = base_state()
    changed = state.replace_player(
        1,
        replace(
            state.players[1],
            hand=(normal(72, Color.RED, 13), normal(73, Color.BLUE, 13)),
        ),
    )

    before = encode_observation(get_observation(state, 0), GameConfig())
    after = encode_observation(get_observation(changed, 0), GameConfig())

    assert before.player_continuous[1][0] == 1 / 22
    assert after.player_continuous[1][0] == 2 / 22
    assert before != after


def test_absolute_seat_rotation_preserves_relative_encoding() -> None:
    state = base_state()
    state = replace(
        state,
        current_player=2,
        starting_player=3,
        players=tuple(
            replace(player, score=index * 101)
            for index, player in enumerate(state.players)
        ),
        discard_history=(
            DiscardRecord(
                normal(81, Color.BLUE, 8),
                player_id=1,
                turn_number=2,
                taken_by=2,
            ),
        ),
    )
    # Move every absolute seat one step clockwise.  Old seat 1, whose
    # perspective is encoded, becomes new seat 2.
    rotated = replace(
        state,
        current_player=3,
        starting_player=0,
        players=(
            state.players[3],
            state.players[0],
            state.players[1],
            state.players[2],
        ),
        discard_history=(
            DiscardRecord(
                state.discard_history[0].tile,
                player_id=2,
                turn_number=2,
                taken_by=3,
            ),
        ),
    )

    assert encode_observation(
        get_observation(state, 1),
        GameConfig(),
    ) == encode_observation(
        get_observation(rotated, 2),
        GameConfig(),
    )


def test_table_assignments_groups_and_attachment_usage_are_preserved() -> None:
    state = base_state()
    okey_a = normal(8, Color.RED, 5)
    okey_b = normal(9, Color.RED, 5)
    red_three = normal(6, Color.RED, 3)
    meld = build_meld((red_three, okey_a, okey_b), state.okey_value)
    state = replace(
        state,
        table=TableState(melds=(TableMeld(7, meld),), next_meld_id=8),
        turn_context=TurnContext(
            draw_source=DrawSource.STOCK,
            attachment_usage=(
                AttachmentUsage(7, AttachmentSide.RIGHT, 2),
            ),
        ),
    )

    encoded = encode_observation(get_observation(state, 0), GameConfig())

    assert encoded.table_group_kinds[0] == 1
    assert encoded.table_group_sizes[0] == 3 / 13
    assert encoded.table_group_attachment_usage[0] == (0.0, 1.0, 0.0)
    assert encoded.table_tile_mask[:3] == (1, 1, 1)
    assert encoded.table_group_slots[:3] == (1, 1, 1)
    assert encoded.table_positions[:3] == (1, 2, 3)
    assert encoded.table_tile_tokens[:3] == (5, 5, 3)
    assert encoded.table_represented_tokens[:3] == (1, 2, 3)
    assert encoded.table_is_real_okey[:3] == (1, 1, 0)


def test_turn_context_and_taken_discard_are_encoded_without_copy_id() -> None:
    state = base_state()
    taken = normal(81, Color.BLUE, 8)
    state = replace(
        state,
        discard_history=(
            DiscardRecord(taken, player_id=3, turn_number=2, taken_by=0),
        ),
        turn_context=TurnContext(
            draw_source=DrawSource.PREVIOUS_DISCARD,
            drawn_tile_id=taken.id,
            taken_discard_tile_id=taken.id,
            opened_this_turn=True,
            stock_exhausted_after_draw=True,
        ),
    )

    encoded = encode_observation(get_observation(state, 0), GameConfig())

    assert encoded.global_categorical[-1] == 3
    assert encoded.turn_flags == (1, 1, 1)
    assert encoded.taken_discard_token == tile_token(
        get_observation(state, 0).taken_discard  # type: ignore[arg-type]
    )


def test_long_discard_history_uses_recent_window_and_full_aggregates() -> None:
    state = base_state()
    history = tuple(
        DiscardRecord(
            normal(1000 + index, Color.RED, index % 13 + 1),
            player_id=index % 4,
            turn_number=index,
            taken_by=(index + 1) % 4 if index % 2 else None,
        )
        for index in range(70)
    )
    state = replace(state, turn_number=80, discard_history=history)

    encoded = encode_observation(get_observation(state, 0), GameConfig())

    assert encoded.history_truncated == 1
    assert sum(encoded.recent_discard_mask) == 64
    assert encoded.recent_discard_tile_tokens[0] == 7
    assert encoded.recent_discard_ages[0] == (80 - 6) / 106
    assert sum(sum(row) for row in encoded.discarded_value_counts) == 35.0
    assert sum(sum(row) for row in encoded.taken_value_counts) == 17.5


def test_all_config_fields_have_explicit_slots_and_normalization() -> None:
    scoring = ScoringConfig(
        normal_finish_reward=-101,
        same_turn_open_finish_reward=-202,
        same_turn_open_okey_finish_reward=-303,
        okey_finish_reward=-404,
        pair_finish_reward=-505,
        elden_finish_reward=-606,
        elden_okey_finish_reward=-707,
        pair_okey_finish_reward=-808,
        unopened_end_penalty=909,
        pair_remaining_multiplier=3,
        same_turn_open_finish_opponent_multiplier=4,
        okey_finish_opponent_multiplier=5,
        elden_finish_opponent_multiplier=6,
        pair_finish_opponent_multiplier=7,
        playable_discard_penalty=1010,
        normal_okey_discard_penalty=1111,
        opened_player_okey_in_hand_surcharge=1212,
        multiply_okey_in_hand_surcharge_by_pair=True,
        multiply_okey_in_hand_surcharge_by_finish=True,
    )
    config = GameConfig(
        player_count=4,
        initial_hand_size=42,
        starting_player_extra_tile=False,
        opening_min_score=202,
        opening_min_pairs=10,
        progressive_opening=True,
        max_contiguous_attach=4,
        require_final_discard=False,
        rounds=200,
        void_round_counts_toward_match=True,
        scoring=scoring,
    )

    encoded = encode_observation(get_observation(base_state(), 0), config)

    assert encoded.rule_continuous == (
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        9.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        10.0,
        11.0,
        12.0,
    )
    assert encoded.rule_flags == (0, 1, 0, 1, 1, 1)
    assert set(RULE_CONTINUOUS_NAMES) == {
        "player_count",
        "initial_hand_size",
        "opening_min_score",
        "opening_min_pairs",
        "max_contiguous_attach",
        "rounds",
        *{
            f"scoring.{field.name}"
            for field in fields(ScoringConfig)
            if not isinstance(getattr(scoring, field.name), bool)
        },
    }
    assert set(RULE_FLAG_NAMES) == {
        "starting_player_extra_tile",
        "progressive_opening",
        "require_final_discard",
        "void_round_counts_toward_match",
        *{
            f"scoring.{field.name}"
            for field in fields(ScoringConfig)
            if isinstance(getattr(scoring, field.name), bool)
        },
    }
