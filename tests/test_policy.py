from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest

from okey101.engine.actions import Action, EndTableActions
from okey101.engine.config import GameConfig
from okey101.rl.action_codec import (
    ActionCandidate,
    ActionCatalog,
    catalog_from_actions,
)
from okey101.rl.candidate_encoder import CANDIDATE_FEATURE_SIZE, encode_catalog
from okey101.rl.encoder import EncodedObservationV1
from okey101.rl.env import Decision, SingleRoundEnv
from okey101.rl.masks import CandidateCapacityError
from okey101.rl.observation import VisibleMeld, VisibleTile
from okey101.rl.policy import ModelInput, prepare_model_input


def discard_decision() -> tuple[Decision, GameConfig]:
    config = GameConfig()
    env = SingleRoundEnv(config)
    decision = env.reset(seed=51)
    end = next(
        action
        for action in decision.legal_actions
        if isinstance(action, EndTableActions)
    )
    discarded = env.step(end).next_decision
    assert discarded is not None
    assert len(discarded.legal_actions) > 1
    return discarded, config


def contains_forbidden_runtime_object(value: object) -> bool:
    if isinstance(
        value,
        (
            ActionCandidate,
            ActionCatalog,
            VisibleTile,
            VisibleMeld,
        ),
    ):
        return True
    if is_dataclass(value):
        return any(
            contains_forbidden_runtime_object(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(contains_forbidden_runtime_object(item) for item in value)
    if isinstance(value, dict):
        return any(contains_forbidden_runtime_object(item) for item in value.values())
    return False


def test_model_input_contains_only_encoded_observation_features_and_mask() -> None:
    decision, config = discard_decision()

    model_input = prepare_model_input(decision, config)

    assert isinstance(model_input.observation, EncodedObservationV1)
    assert len(model_input.candidate_features) == len(model_input.action_mask)
    assert all(model_input.action_mask)
    assert all(
        len(row) == CANDIDATE_FEATURE_SIZE
        and all(isinstance(value, float) for value in row)
        for row in model_input.candidate_features
    )
    assert not contains_forbidden_runtime_object(model_input)
    assert {field.name for field in fields(ModelInput)} == {
        "observation",
        "candidate_features",
        "action_mask",
    }


def test_candidate_rows_follow_canonical_catalog_not_decision_input_order() -> None:
    decision, config = discard_decision()
    reversed_decision = replace(
        decision,
        legal_actions=tuple(reversed(decision.legal_actions)),
    )
    catalog = catalog_from_actions(reversed_decision.legal_actions)

    model_input = prepare_model_input(reversed_decision, config)
    expected = tuple(
        feature.as_vector()
        for feature in encode_catalog(
            reversed_decision.observation,
            catalog,
            config,
        )
    )

    assert model_input.candidate_features == expected
    for candidate_id, expected_features in enumerate(expected):
        action = catalog.decode(candidate_id)
        assert action in reversed_decision.legal_actions
        assert model_input.candidate_features[candidate_id] == expected_features


def test_padding_adds_zero_rows_and_false_mask_only_at_end() -> None:
    decision, config = discard_decision()
    legal_count = len(decision.legal_actions)

    model_input = prepare_model_input(
        decision,
        config,
        capacity=legal_count + 3,
    )

    assert model_input.action_mask == (True,) * legal_count + (False,) * 3
    assert model_input.candidate_features[-3:] == (
        (0.0,) * CANDIDATE_FEATURE_SIZE,
    ) * 3
    with pytest.raises(CandidateCapacityError):
        prepare_model_input(decision, config, capacity=legal_count - 1)


def test_runner_seed_and_absolute_seat_are_not_model_inputs() -> None:
    decision, config = discard_decision()
    changed_runner_metadata = replace(
        decision,
        seat=decision.seat + 100,
    )

    assert prepare_model_input(decision, config) == prepare_model_input(
        changed_runner_metadata,
        config,
    )
    assert all(
        forbidden not in {field.name for field in fields(ModelInput)}
        for forbidden in (
            "seed",
            "seat",
            "action",
            "catalog",
            "key",
            "candidate_id",
            "physical_id",
            "meld_id",
        )
    )
    assert not hasattr(decision, "episode_seed")


def test_model_input_validates_padding_and_feature_width() -> None:
    decision, config = discard_decision()
    valid = prepare_model_input(decision, config)

    with pytest.raises(ValueError, match="equal length"):
        replace(valid, action_mask=(*valid.action_mask, False))
    with pytest.raises(ValueError, match="invalid width"):
        replace(
            valid,
            candidate_features=((0.0,), *valid.candidate_features[1:]),
        )
    with pytest.raises(ValueError, match="padding candidate"):
        ModelInput(
            observation=valid.observation,
            candidate_features=(valid.candidate_features[0],),
            action_mask=(False,),
        )


def test_empty_decision_is_rejected() -> None:
    decision, config = discard_decision()

    with pytest.raises(ValueError, match="at least one legal"):
        prepare_model_input(
            replace(decision, legal_actions=()),
            config,
        )
