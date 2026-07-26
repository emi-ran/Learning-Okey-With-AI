from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from okey101.engine.actions import Discard
from okey101.rl.env import InvalidEnvironmentAction, SingleRoundEnv
from okey101.rl.vector_env import VectorRoundEnv


def _contains_seed_field(value: object) -> bool:
    if is_dataclass(value):
        for field in fields(value):
            if field.name in {"seed", "episode_seed"}:
                return True
            if _contains_seed_field(getattr(value, field.name)):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_seed_field(item) for item in value)
    elif isinstance(value, dict):
        return any(_contains_seed_field(item) for item in value.values())
    return False


def test_vector_reset_is_deterministic_and_matches_standalone_envs() -> None:
    first = VectorRoundEnv(3)
    second = VectorRoundEnv(3)

    first_decisions = first.reset(seed=808, starting_players=(0, 1, 2))
    second_decisions = second.reset(seed=808, starting_players=(0, 1, 2))

    assert first.base_seed == second.base_seed == 808
    assert first.episode_seeds == second.episode_seeds
    assert first_decisions == second_decisions
    assert len(set(first.episode_seeds)) == 3
    assert not _contains_seed_field(first_decisions)

    standalone = tuple(
        SingleRoundEnv().reset(seed, starting_player=starter)
        for seed, starter in zip(
            first.episode_seeds,
            (0, 1, 2),
            strict=True,
        )
    )
    assert first_decisions == standalone


def test_vector_step_prevalidates_every_slot_before_any_mutation() -> None:
    env = VectorRoundEnv(2)
    decisions = env.reset(seed=44)
    before = env.current_decisions

    with pytest.raises(InvalidEnvironmentAction, match="legal candidates"):
        env.step((decisions[0].legal_actions[0], Discard(-1)))

    assert env.current_decisions == before
    assert env.terminated == (False, False)

    with pytest.raises(ValueError, match="length"):
        env.step((decisions[0].legal_actions[0],))
    assert env.current_decisions == before


def test_vector_trajectory_matches_independent_envs_across_resets() -> None:
    vector = VectorRoundEnv(3)
    decisions = vector.reset(seed=2026)
    standalone_envs = tuple(SingleRoundEnv() for _ in range(3))
    standalone_decisions = tuple(
        env.reset(seed)
        for env, seed in zip(
            standalone_envs,
            vector.episode_seeds,
            strict=True,
        )
    )
    assert decisions == standalone_decisions

    reset_counter = 0
    for step_index in range(180):
        current = vector.current_decisions
        assert all(decision is not None for decision in current)
        actions = tuple(
            decision.legal_actions[
                (step_index + env_index * 5) % len(decision.legal_actions)
            ]
            for env_index, maybe_decision in enumerate(current)
            for decision in (maybe_decision,)
            if decision is not None
        )
        vector_results = vector.step(actions)
        standalone_results = tuple(
            env.step(action)
            for env, action in zip(standalone_envs, actions, strict=True)
        )
        assert vector_results == standalone_results

        for env_index, result in enumerate(vector_results):
            if not result.terminated:
                continue
            reset_counter += 1
            new_seed = 50_000 + reset_counter * 10 + env_index
            vector_decision = vector.reset_at(env_index, new_seed)
            standalone_decision = standalone_envs[env_index].reset(new_seed)
            assert vector_decision == standalone_decision

    assert reset_counter > 0


def test_reset_at_preserves_other_slots_and_generated_seed_is_exposed() -> None:
    env = VectorRoundEnv(3)
    env.reset(seed=17)
    before = env.current_decisions

    replacement = env.reset_at(0)

    assert isinstance(env.episode_seeds[0], int)
    assert not hasattr(replacement, "episode_seed")
    assert env.current_decisions[1:] == before[1:]


def test_vector_validates_constructor_reset_and_indices_before_mutation() -> None:
    with pytest.raises(ValueError, match="positive"):
        VectorRoundEnv(0)
    with pytest.raises(TypeError, match="integer"):
        VectorRoundEnv(True)

    env = VectorRoundEnv(2)
    with pytest.raises(ValueError, match="length"):
        env.reset(seed=1, starting_players=(0,))
    assert env.current_decisions == (None, None)

    with pytest.raises(ValueError, match="seat range"):
        env.reset(seed=1, starting_players=(0, 4))
    assert env.current_decisions == (None, None)

    with pytest.raises(IndexError, match="out of range"):
        env.reset_at(2, seed=1)
