from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace

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
from okey101.engine.melds import Meld, MeldKind, MeldTile
from okey101.engine.config import GameConfig
from okey101.engine.pairs import Pair
from okey101.engine.player import PlayerState
from okey101.engine.state import (
    AttachmentUsage,
    DiscardRecord,
    GameState,
    TurnContext,
    TurnPhase,
)
from okey101.engine.round import RoundEngine
from okey101.engine.table import AttachmentSide, TableMeld, TableState
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue, build_tile_set
from okey101.engine.transition import apply_action
from okey101.rl.action_codec import catalog_from_actions
from okey101.rl.candidate_encoder import (
    ACTION_TYPE_SIZE,
    CANDIDATE_FEATURE_SIZE,
    LAID_GROUP_FEATURE_SIZE,
    MAX_LAID_GROUPS,
    SCALAR_SIZE,
    CandidateEncodingError,
    CandidateFeatures,
    encode_candidate,
    encode_catalog,
)
from okey101.rl.encoder import encode_observation
from okey101.rl.observation import get_observation


TILES = build_tile_set()
CONFIG = GameConfig()


def tile(color: Color, number: int, copy: int = 0) -> PhysicalTile:
    color_index = tuple(Color).index(color)
    return TILES[color_index * 26 + (number - 1) * 2 + copy]


def fixture_state(
    *,
    player_id: int = 0,
    meld_id: int = 77,
) -> tuple[GameState, dict[str, object]]:
    okey = TileValue(Color.RED, 1)
    real_okey = tile(Color.RED, 1, 1)
    table_meld = Meld(
        MeldKind.RUN,
        (
            MeldTile(tile(Color.BLUE, 2), TileValue(Color.BLUE, 2)),
            MeldTile(real_okey, TileValue(Color.BLUE, 3)),
            MeldTile(tile(Color.BLUE, 4), TileValue(Color.BLUE, 4)),
        ),
    )
    pair = Pair(
        (
            MeldTile(tile(Color.YELLOW, 7), TileValue(Color.YELLOW, 7)),
            MeldTile(tile(Color.YELLOW, 7, 1), TileValue(Color.YELLOW, 7)),
        )
    )
    hand = (
        tile(Color.RED, 1),
        tile(Color.RED, 2),
        tile(Color.RED, 3),
        tile(Color.RED, 4),
        tile(Color.RED, 5),
        tile(Color.RED, 5, 1),
        tile(Color.BLUE, 5),
        tile(Color.BLUE, 3),
        tile(Color.BLACK, 13),
        TILES[104],
    )
    players = [PlayerState() for _ in range(4)]
    players[player_id] = PlayerState(hand=hand)
    discarded = tile(Color.YELLOW, 8)
    state = GameState(
        round_id=1,
        turn_number=4,
        current_player=player_id,
        starting_player=player_id,
        indicator=tile(Color.RED, 13),
        okey_value=okey,
        stock=(tile(Color.BLACK, 12),),
        discard_pile=(discarded,),
        discard_history=(
            DiscardRecord(
                tile=discarded,
                player_id=(player_id - 1) % 4,
                turn_number=3,
            ),
        ),
        players=tuple(players),
        table=TableState(
            melds=(TableMeld(meld_id, table_meld),),
            pairs=(pair,),
            next_meld_id=meld_id + 1,
        ),
        phase=TurnPhase.TABLE_ACTIONS,
        turn_context=TurnContext(
            attachment_usage=(
                AttachmentUsage(meld_id, AttachmentSide.RIGHT, 1),
            ),
        ),
    )
    opening_meld = Meld(
        MeldKind.RUN,
        (
            MeldTile(hand[1], TileValue(Color.RED, 2)),
            MeldTile(hand[2], TileValue(Color.RED, 3)),
            MeldTile(hand[0], TileValue(Color.RED, 4)),
        ),
    )
    hand_pair = Pair(
        (
            MeldTile(hand[4], TileValue(Color.RED, 5)),
            MeldTile(hand[5], TileValue(Color.RED, 5)),
        )
    )
    actions = {
        "draw": DrawFromStock(),
        "take": TakePreviousDiscard(),
        "open_melds": OpenMelds((opening_meld,)),
        "open_pairs": OpenPairs((hand_pair,)),
        "add_to_meld": AddToMeld(
            meld_id,
            (MeldTile(hand[6], TileValue(Color.BLUE, 5)),),
            AttachmentSide.RIGHT,
        ),
        "add_pair": AddPair(hand_pair),
        "replace_joker": ReplaceJoker(
            meld_id,
            real_okey.id,
            hand[7].id,
        ),
        "end": EndTableActions(),
        "discard": Discard(hand[8].id),
    }
    return state, actions


def test_all_nine_action_types_have_distinct_fixed_shape_features() -> None:
    state, actions = fixture_state()
    observation = get_observation(state, state.current_player)
    catalog = catalog_from_actions(tuple(actions.values()))

    encoded = encode_catalog(observation, catalog, CONFIG)

    assert len(encoded) == 9
    assert all(len(item.action_type) == ACTION_TYPE_SIZE for item in encoded)
    assert all(len(item.scalars) == SCALAR_SIZE for item in encoded)
    assert all(len(item.laid_groups) == MAX_LAID_GROUPS for item in encoded)
    assert all(
        len(group) == LAID_GROUP_FEATURE_SIZE
        for item in encoded
        for group in item.laid_groups
    )
    assert all(len(item.as_vector()) == CANDIDATE_FEATURE_SIZE for item in encoded)
    assert {item.action_type for item in encoded} == {
        tuple(1.0 if index == action_index else 0.0 for index in range(9))
        for action_index in range(9)
    }


def test_real_okey_assignment_and_target_assignment_are_encoded() -> None:
    state, actions = fixture_state()
    observation = get_observation(state, 0)

    opening = encode_candidate(observation, actions["open_melds"], CONFIG)
    replacement = encode_candidate(
        observation,
        actions["replace_joker"],
        CONFIG,
    )
    represented_red_four = tuple(Color).index(Color.RED) * 13 + 3
    represented_blue_three = tuple(Color).index(Color.BLUE) * 13 + 2

    assert opening.selected_tile_roles[53] == 1.0
    assert opening.joker_assignments[represented_red_four] == 1.0
    assert replacement.target_joker_assignments[represented_blue_three] == 1.0
    assert replacement.represented_values[represented_blue_three] == 1.0
    assert replacement.scalars[13] == 1.0


def test_same_value_physical_copies_get_same_features_but_stay_two_candidates() -> None:
    state, _actions = fixture_state()
    observation = get_observation(state, 0)
    first = Discard(tile(Color.RED, 5).id)
    second = Discard(tile(Color.RED, 5, 1).id)
    catalog = catalog_from_actions((first, second))

    encoded = encode_catalog(observation, catalog, CONFIG)

    assert len(catalog) == 2
    assert catalog.decode(0) != catalog.decode(1)
    assert encoded[0] == encoded[1]


def test_table_and_physical_id_changes_do_not_change_neural_features() -> None:
    first_state, first_actions = fixture_state(meld_id=77)
    second_state, second_actions = fixture_state(meld_id=901)
    first_observation = get_observation(first_state, 0)
    second_observation = get_observation(second_state, 0)

    first = encode_candidate(
        first_observation,
        first_actions["add_to_meld"],
        CONFIG,
    )
    second = encode_candidate(
        second_observation,
        second_actions["add_to_meld"],
        CONFIG,
    )

    assert first == second
    assert first.scalars[11] == 1.0


def test_absolute_seat_rotation_does_not_change_candidate_features() -> None:
    first_state, first_actions = fixture_state(player_id=0)
    second_state, second_actions = fixture_state(player_id=2)

    first = encode_candidate(
        get_observation(first_state, 0),
        first_actions["open_melds"],
        CONFIG,
    )
    second = encode_candidate(
        get_observation(second_state, 2),
        second_actions["open_melds"],
        CONFIG,
    )

    assert first == second


def test_feature_schema_contains_no_identifier_field_or_non_float_payload() -> None:
    state, actions = fixture_state()
    feature = encode_candidate(
        get_observation(state, 0),
        actions["replace_joker"],
        CONFIG,
    )

    field_names = {field.name for field in fields(CandidateFeatures)}
    assert not any(
        name.endswith("_id") or "physical" in name or "identifier" in name
        for name in field_names
    )
    assert all(isinstance(value, float) for value in feature.as_vector())


def test_take_discard_uses_public_visible_tile_without_exposing_its_id() -> None:
    state, actions = fixture_state()
    feature = encode_candidate(
        get_observation(state, 0),
        actions["take"],
        CONFIG,
    )
    yellow_eight_index = tuple(Color).index(Color.YELLOW) * 13 + 7

    assert feature.selected_tile_roles[yellow_eight_index] == 1.0
    assert feature.scalars[0] == 1.0


def test_discard_penalty_signals_distinguish_playable_and_real_okey() -> None:
    state, _actions = fixture_state()
    observation = get_observation(state, 0)

    playable = encode_candidate(
        observation,
        Discard(tile(Color.BLUE, 5).id),
        CONFIG,
    )
    real_okey = encode_candidate(
        observation,
        Discard(tile(Color.RED, 1).id),
        CONFIG,
    )
    ordinary = encode_candidate(
        observation,
        Discard(tile(Color.BLACK, 13).id),
        CONFIG,
    )

    assert playable.scalars[-2:] == (1.0, 0.0)
    assert real_okey.scalars[-2:] == (0.0, 1.0)
    assert ordinary.scalars[-2:] == (0.0, 0.0)


def test_encoder_rejects_tiles_or_targets_not_visible_to_actor() -> None:
    state, actions = fixture_state()
    observation = get_observation(state, 0)

    with pytest.raises(CandidateEncodingError, match="acting hand"):
        encode_candidate(
            observation,
            Discard(tile(Color.YELLOW, 9).id),
            CONFIG,
        )
    with pytest.raises(CandidateEncodingError, match="target meld"):
        encode_candidate(
            observation,
            replace(actions["add_to_meld"], meld_id=999),
            CONFIG,
        )


def test_take_uses_only_current_discard_record() -> None:
    state, actions = fixture_state()
    old_available = state.discard_history[-1]
    current_taken = DiscardRecord(
        tile=tile(Color.YELLOW, 9),
        player_id=2,
        turn_number=4,
        taken_by=0,
    )
    observation = get_observation(
        replace(
            state,
            discard_history=(old_available, current_taken),
        ),
        0,
    )

    with pytest.raises(CandidateEncodingError, match="already been taken"):
        encode_candidate(observation, actions["take"], CONFIG)


def test_playable_discard_respects_attachment_limit_and_final_immunity() -> None:
    state, _actions = fixture_state()
    observation = get_observation(state, 0)
    discard = Discard(tile(Color.BLUE, 5).id)

    assert encode_candidate(observation, discard, CONFIG).scalars[-2] == 1.0
    assert (
        encode_candidate(
            observation,
            discard,
            GameConfig(max_contiguous_attach=1),
        ).scalars[-2]
        == 0.0
    )

    final_observation = replace(
        observation,
        player_statuses=(
            replace(observation.player_statuses[0], hand_count=1),
            *observation.player_statuses[1:],
        ),
    )
    assert (
        encode_candidate(final_observation, discard, CONFIG).scalars[-2]
        == 0.0
    )


def seed_one_collision_fixture():
    engine = RoundEngine(CONFIG)
    engine.reset(seed=1)
    for episode_step in range(54):
        assert engine.state is not None
        state = engine.state
        observation = engine.get_observation(state.current_player)
        catalog = catalog_from_actions(engine.get_legal_actions())
        if episode_step == 53:
            return state, observation, catalog
        candidate_id = (
            episode_step * 7 + state.current_player
        ) % len(catalog)
        engine.step(catalog.decode(candidate_id))
    raise AssertionError("seed-1 collision step was not reached")


def test_laid_group_partition_breaks_seed_one_step_53_collision() -> None:
    state, observation, catalog = seed_one_collision_fixture()
    assert len(catalog) == 19
    first_action = catalog.decode(8)
    second_action = catalog.decode(10)

    first_features = encode_candidate(
        observation,
        catalog.candidates[8],
        CONFIG,
    )
    second_features = encode_candidate(
        observation,
        catalog.candidates[10],
        CONFIG,
    )
    first_state, _events = apply_action(state, first_action, CONFIG)
    second_state, _events = apply_action(state, second_action, CONFIG)
    first_result = encode_observation(
        get_observation(first_state, first_state.current_player),
        CONFIG,
    )
    second_result = encode_observation(
        get_observation(second_state, second_state.current_player),
        CONFIG,
    )

    assert first_result != second_result
    assert first_features != second_features
    assert first_features.laid_groups != second_features.laid_groups


def test_equal_features_at_collision_fixture_imply_equal_result_encoding() -> None:
    state, observation, catalog = seed_one_collision_fixture()
    by_features: dict[tuple[float, ...], list[int]] = defaultdict(list)
    result_encodings = []
    for index, candidate in enumerate(catalog.candidates):
        features = encode_candidate(observation, candidate, CONFIG).as_vector()
        by_features[features].append(index)
        resulting_state, _events = apply_action(
            state,
            candidate.action,
            CONFIG,
        )
        result_encodings.append(
            encode_observation(
                get_observation(
                    resulting_state,
                    resulting_state.current_player,
                ),
                CONFIG,
            )
        )

    assert any(len(indexes) > 1 for indexes in by_features.values())
    for indexes in by_features.values():
        expected = result_encodings[indexes[0]]
        assert all(result_encodings[index] == expected for index in indexes[1:])


def test_laid_group_order_permutations_have_identical_features() -> None:
    _state, observation, catalog = seed_one_collision_fixture()
    opening = next(
        candidate.action
        for candidate in catalog.candidates
        if isinstance(candidate.action, OpenMelds)
        and len(candidate.action.melds) > 1
    )
    reversed_opening = OpenMelds(tuple(reversed(opening.melds)))

    assert encode_candidate(observation, opening, CONFIG) == encode_candidate(
        observation,
        reversed_opening,
        CONFIG,
    )


def test_laid_group_capacity_is_explicit_and_overflow_fails() -> None:
    engine = RoundEngine(CONFIG)
    engine.reset(seed=1)
    assert engine.state is not None
    observation = engine.get_observation(engine.state.current_player)
    hand = engine.state.current_player_state.hand
    pairs = tuple(
        Pair(
            (
                MeldTile(hand[index], observation.okey_value),
                MeldTile(hand[index + 1], observation.okey_value),
            )
        )
        for index in range(0, 22, 2)
    )

    assert MAX_LAID_GROUPS == 10
    with pytest.raises(CandidateEncodingError, match="V1 capacity"):
        encode_candidate(observation, OpenPairs(pairs), CONFIG)
