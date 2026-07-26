from __future__ import annotations

import json

import numpy as np

from okey101.engine.config import GameConfig
from okey101.rl.action_codec import catalog_from_actions
from okey101.rl.env import SingleRoundEnv
from okey101.rl.policy import prepare_model_input
from okey101.training import (
    SelfPlayTrainer,
    TrainingConfig,
    evaluate_against_random,
    load_checkpoint,
    save_checkpoint,
)


def _fast_config() -> GameConfig:
    return GameConfig(initial_hand_size=3, opening_min_score=3)


def _parameter_copies(trainer: SelfPlayTrainer) -> dict[str, np.ndarray]:
    return {
        name: value.copy()
        for name, value in trainer.model.parameters.items()
    }


def test_shared_policy_scores_only_engine_legal_candidates() -> None:
    trainer = SelfPlayTrainer(seed=2, game_config=_fast_config())
    env = SingleRoundEnv(_fast_config())
    decision = env.reset(seed=8)
    model_input = prepare_model_input(decision, env.config, capacity=4)

    probabilities, value = trainer.model.forward(model_input)
    selected, selected_value = trainer.model.select(
        model_input,
        rng=np.random.default_rng(4),
    )
    catalog = catalog_from_actions(decision.legal_actions)

    assert probabilities.shape == (len(catalog),)
    np.testing.assert_allclose(
        probabilities.sum(),
        1.0,
    )
    assert np.isfinite(value)
    assert value == selected_value
    assert catalog.decode(selected) in decision.legal_actions


def test_real_self_play_updates_parameters() -> None:
    trainer = SelfPlayTrainer(
        seed=7,
        game_config=_fast_config(),
        training_config=TrainingConfig(
            hidden_size=4,
            max_actions_per_episode=500,
        ),
    )
    before = _parameter_copies(trainer)

    result = trainer.train_episode()

    assert result.actions > 0
    assert result.episode == 1
    assert trainer.optimizer.step == 1
    assert not np.array_equal(
        before["candidate_score_weight"],
        trainer.model.candidate_score_weight,
    )
    assert any(
        not np.array_equal(before[name], after)
        for name, after in trainer.model.parameters.items()
    )


def test_checkpoint_resume_is_bitwise_deterministic(tmp_path) -> None:
    config = TrainingConfig(hidden_size=4, max_actions_per_episode=500)
    uninterrupted = SelfPlayTrainer(
        seed=19,
        game_config=_fast_config(),
        training_config=config,
    )
    first = uninterrupted.train_episode()
    checkpoint = save_checkpoint(uninterrupted, tmp_path / "policy.npz")
    expected_second = uninterrupted.train_episode()

    resumed = load_checkpoint(checkpoint)
    actual_second = resumed.train_episode()

    assert first.episode == 1
    assert actual_second == expected_second
    assert resumed.episodes_completed == uninterrupted.episodes_completed
    assert resumed.actions_completed == uninterrupted.actions_completed
    for name, expected in uninterrupted.model.parameters.items():
        np.testing.assert_array_equal(resumed.model.parameters[name], expected)
        np.testing.assert_array_equal(
            resumed.optimizer.first_moment[name],
            uninterrupted.optimizer.first_moment[name],
        )
        np.testing.assert_array_equal(
            resumed.optimizer.second_moment[name],
            uninterrupted.optimizer.second_moment[name],
        )

    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
    assert metadata["checkpoint_schema_version"] == 1
    assert metadata["feature_schema"] == {
        "action_codec": 1,
        "candidate": 1,
        "observation": 1,
    }
    assert metadata["git_hash"]
    assert metadata["progress"]["optimizer_step"] == 1
    assert metadata["rng_state"]


def test_evaluation_rotates_policy_against_random_agents() -> None:
    trainer = SelfPlayTrainer(
        seed=23,
        game_config=_fast_config(),
        training_config=TrainingConfig(hidden_size=4),
    )

    result = evaluate_against_random(
        trainer.model,
        episodes=4,
        start_seed=40,
        game_config=_fast_config(),
        max_actions_per_episode=500,
    )

    assert result.episodes == 4
    assert result.actions > 0
    assert np.isfinite(result.mean_relative_reward)
    assert np.isfinite(result.mean_score)
    assert 0.0 <= result.finish_rate <= 1.0
    assert 0.0 <= result.real_okey_discard_rate <= 1.0
    assert 0.0 <= result.playable_discard_rate <= 1.0
