from __future__ import annotations

from dataclasses import fields, is_dataclass

import pytest

from okey101.engine.actions import Discard
from okey101.engine.config import GameConfig
from okey101.engine.state import EngineEvent, GameState
from okey101.rl.env import (
    InvalidEnvironmentAction,
    SingleRoundEnv,
    StepResult,
)
from okey101.rl.rewards import relative_terminal_rewards


def _contains_instance(value: object, forbidden: tuple[type[object], ...]) -> bool:
    if isinstance(value, forbidden):
        return True
    if is_dataclass(value):
        return any(
            _contains_instance(getattr(value, field.name), forbidden)
            for field in fields(value)
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_instance(item, forbidden) for item in value)
    if isinstance(value, dict):
        return any(_contains_instance(item, forbidden) for item in value.values())
    return False


def _play_by_candidate_ordinal(
    env: SingleRoundEnv,
    *,
    limit: int = 1000,
) -> tuple[StepResult, ...]:
    results: list[StepResult] = []
    for step_index in range(limit):
        decision = env.current_decision
        assert decision is not None
        action = decision.legal_actions[
            (step_index * 7 + decision.seat) % len(decision.legal_actions)
        ]
        result = env.step(action)
        results.append(result)
        if result.terminated:
            return tuple(results)
    raise AssertionError("round did not terminate within the action limit")


def test_relative_terminal_rewards_are_normalized_and_zero_sum() -> None:
    rewards = relative_terminal_rewards((-101, 202, 202, 202))

    assert rewards == (3.0, -1.0, -1.0, -1.0)
    assert sum(rewards) == pytest.approx(0.0)
    assert relative_terminal_rewards((0, 0, 0, 0)) == (0.0,) * 4

    with pytest.raises(ValueError, match="at least two"):
        relative_terminal_rewards((0,))
    with pytest.raises(ValueError, match="finite and positive"):
        relative_terminal_rewards((0, 1), scale=0)
    with pytest.raises(TypeError, match="integers"):
        relative_terminal_rewards((0, True))


def test_same_seed_and_candidate_policy_reproduce_full_episode() -> None:
    first = SingleRoundEnv()
    second = SingleRoundEnv()

    assert first.reset(seed=-73, starting_player=2) == second.reset(
        seed=-73,
        starting_player=2,
    )
    assert _play_by_candidate_ordinal(first) == _play_by_candidate_ordinal(second)


def test_generated_episode_seed_is_exposed_and_replays_initial_decision() -> None:
    env = SingleRoundEnv()
    initial = env.reset()

    assert isinstance(initial.episode_seed, int)
    assert env.episode_seed == initial.episode_seed
    assert SingleRoundEnv().reset(initial.episode_seed) == initial


def test_policy_contract_has_zero_intermediate_reward_and_no_raw_state() -> None:
    env = SingleRoundEnv()
    decision = env.reset(seed=91)

    assert 0 <= decision.seat < env.num_players
    assert decision.observation.current_player_relative == 0
    assert decision.legal_actions
    assert not _contains_instance(decision, (GameState, EngineEvent))

    first = env.step(decision.legal_actions[0])
    assert not first.terminated
    assert first.rewards == (0.0,) * env.num_players
    assert first.final_scores is None
    assert first.terminal_reason is None
    assert first.next_decision is not None
    assert not _contains_instance(first, (GameState, EngineEvent))

    remaining = _play_by_candidate_ordinal(env)
    terminal = remaining[-1]
    assert terminal.terminated
    assert terminal.next_decision is None
    assert terminal.final_scores is not None
    assert terminal.rewards == pytest.approx(
        relative_terminal_rewards(terminal.final_scores)
    )
    assert sum(terminal.rewards) == pytest.approx(0.0)
    assert not _contains_instance(terminal, (GameState, EngineEvent))


def test_invalid_or_post_terminal_action_is_rejected_without_advancing() -> None:
    env = SingleRoundEnv()
    decision = env.reset(seed=12)

    with pytest.raises(InvalidEnvironmentAction, match="legal candidates"):
        env.step(Discard(-1))
    assert env.current_decision == decision

    terminal = _play_by_candidate_ordinal(env)[-1]
    assert terminal.terminated
    with pytest.raises(RuntimeError, match="Terminal round"):
        env.step(terminal)


def test_random_seed_sample_has_no_nonterminal_legal_action_dead_end() -> None:
    for seed in range(10):
        env = SingleRoundEnv()
        env.reset(seed)
        results = _play_by_candidate_ordinal(env)
        assert results[-1].terminated
        assert all(
            result.next_decision is None
            or result.next_decision.legal_actions
            for result in results
        )


def test_single_round_environment_rejects_match_config_and_bad_rewards() -> None:
    with pytest.raises(ValueError, match=r"rounds=1"):
        SingleRoundEnv(GameConfig(rounds=2))

    env = SingleRoundEnv(reward_fn=lambda _scores: (0.0,))
    env.reset(seed=3)
    with pytest.raises(ValueError, match="reward count"):
        _play_by_candidate_ordinal(env)
