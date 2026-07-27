"""Shared-policy stochastic self-play and RandomAgent evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from okey101.agents.random_agent import RandomAgent
from okey101.engine.actions import Action, Discard
from okey101.engine.config import GameConfig
from okey101.engine.state import TerminalReason
from okey101.rl.action_codec import catalog_from_actions
from okey101.rl.candidate_encoder import SCALAR_NAMES, encode_candidate
from okey101.rl.env import Decision, SingleRoundEnv
from okey101.rl.policy import ModelInput, prepare_model_input

from .model import (
    FloatArray,
    NumpyActorCritic,
    candidate_matrix,
    observation_vector,
)
from .optimizer import Adam

_PLAYABLE_INDEX = SCALAR_NAMES.index("nonfinal_playable_discard")
_OKEY_INDEX = SCALAR_NAMES.index("real_okey_discard")
_NO_WINNER_REASONS = {
    TerminalReason.STOCK_EXHAUSTED,
    TerminalReason.ALL_PLAYERS_OPENED_PAIRS,
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    hidden_size: int = 16
    learning_rate: float = 0.003
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 5.0
    max_actions_per_episode: int = 1_000

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        for name in (
            "learning_rate",
            "value_coefficient",
            "max_gradient_norm",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isfinite(self.entropy_coefficient)
            or self.entropy_coefficient < 0
        ):
            raise ValueError("entropy_coefficient must be finite and non-negative")
        if self.max_actions_per_episode < 1:
            raise ValueError("max_actions_per_episode must be positive")


@dataclass(frozen=True, slots=True)
class EpisodeTrainingResult:
    episode: int
    episode_seed: int
    actions: int
    loss: float
    gradient_norm: float
    rewards: tuple[float, ...]
    final_scores: tuple[int, ...]
    terminal_reason: str
    winner_seat: int | None
    actions_by_seat: tuple[int, ...]
    discard_actions: int
    discards_by_seat: tuple[int, ...]
    real_okey_discards: int
    real_okey_discards_by_seat: tuple[int, ...]
    playable_discards: int
    playable_discards_by_seat: tuple[int, ...]
    immediate_penalties: tuple[int, ...]
    opened_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    mean_relative_reward: float
    mean_score: float
    finish_rate: float
    actions: int
    discard_actions: int
    real_okey_discards: int
    playable_discards: int
    immediate_penalty_points: int
    penalized_episodes: int
    opened_series: int
    opened_pairs: int
    best_score: int
    worst_score: int
    terminal_reasons: dict[str, int]

    @property
    def real_okey_discard_rate(self) -> float:
        return (
            self.real_okey_discards / self.discard_actions
            if self.discard_actions
            else 0.0
        )

    @property
    def playable_discard_rate(self) -> float:
        return (
            self.playable_discards / self.discard_actions
            if self.discard_actions
            else 0.0
        )

    @property
    def mean_immediate_penalty(self) -> float:
        return self.immediate_penalty_points / self.episodes

    @property
    def penalized_episode_rate(self) -> float:
        return self.penalized_episodes / self.episodes


def _selected_action(
    decision: Decision,
    candidate_index: int,
) -> Action:
    return catalog_from_actions(decision.legal_actions).decode(candidate_index)


def _discard_flags(
    decision: Decision,
    action: Action,
    game_config: GameConfig,
) -> tuple[int, int, int]:
    if not isinstance(action, Discard):
        return 0, 0, 0
    features = encode_candidate(
        decision.observation,
        action,
        game_config,
    )
    return (
        1,
        int(features.scalars[_OKEY_INDEX]),
        int(features.scalars[_PLAYABLE_INDEX]),
    )


class SelfPlayTrainer:
    """Owns one policy shared by all seats, its optimizer, and its RNG."""

    def __init__(
        self,
        *,
        seed: int = 0,
        training_config: TrainingConfig | None = None,
        game_config: GameConfig | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.training_config = training_config or TrainingConfig()
        self.game_config = game_config or GameConfig()
        if self.game_config.rounds != 1:
            raise ValueError("training requires GameConfig(rounds=1)")
        self.rng = np.random.default_rng(seed)

        probe_env = SingleRoundEnv(self.game_config)
        probe = prepare_model_input(
            probe_env.reset(seed=0, starting_player=0),
            self.game_config,
        )
        self.model = NumpyActorCritic(
            observation_size=len(observation_vector(probe)),
            hidden_size=self.training_config.hidden_size,
            rng=self.rng,
        )
        self.optimizer = Adam(
            self.model.parameters,
            learning_rate=self.training_config.learning_rate,
            max_gradient_norm=self.training_config.max_gradient_norm,
        )
        self.episodes_completed = 0
        self.actions_completed = 0

    def train_episode(self) -> EpisodeTrainingResult:
        episode_seed = int(
            self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        starting_player = int(self.rng.integers(0, self.game_config.player_count))
        env = SingleRoundEnv(self.game_config)
        decision = env.reset(
            seed=episode_seed,
            starting_player=starting_player,
        )
        trajectory: list[
            tuple[FloatArray, FloatArray, int, int]
        ] = []
        actions_by_seat = [0] * self.game_config.player_count
        discards_by_seat = [0] * self.game_config.player_count
        real_okey_by_seat = [0] * self.game_config.player_count
        playable_by_seat = [0] * self.game_config.player_count

        for action_index in range(
            1,
            self.training_config.max_actions_per_episode + 1,
        ):
            model_input = prepare_model_input(decision, self.game_config)
            observation = observation_vector(model_input)
            candidates = candidate_matrix(model_input)
            selected, _value = self.model.select_prepared(
                observation,
                candidates,
                rng=self.rng,
            )
            action = _selected_action(decision, selected)
            discard, okey, playable = _discard_flags(
                decision,
                action,
                self.game_config,
            )
            actions_by_seat[decision.seat] += 1
            discards_by_seat[decision.seat] += discard
            real_okey_by_seat[decision.seat] += okey
            playable_by_seat[decision.seat] += playable
            trajectory.append(
                (observation, candidates, selected, decision.seat)
            )
            result = env.step(action)
            if not result.terminated:
                assert result.next_decision is not None
                decision = result.next_decision
                continue

            if (
                result.final_scores is None
                or result.terminal_reason is None
            ):
                raise RuntimeError("terminal environment result is incomplete")
            if (
                result.immediate_penalties is None
                or result.opened_modes is None
            ):
                raise RuntimeError("terminal telemetry is incomplete")
            samples = [
                (observation, candidates, choice, result.rewards[seat])
                for observation, candidates, choice, seat in trajectory
            ]
            loss, gradients = self.model.loss_and_gradients_prepared(
                samples,
                value_coefficient=self.training_config.value_coefficient,
                entropy_coefficient=self.training_config.entropy_coefficient,
            )
            gradient_norm = self.optimizer.update(
                self.model.parameters,
                gradients,
            )
            self.episodes_completed += 1
            self.actions_completed += action_index
            return EpisodeTrainingResult(
                episode=self.episodes_completed,
                episode_seed=episode_seed,
                actions=action_index,
                loss=loss,
                gradient_norm=gradient_norm,
                rewards=result.rewards,
                final_scores=result.final_scores,
                terminal_reason=result.terminal_reason.value,
                winner_seat=(
                    None
                    if result.terminal_reason in _NO_WINNER_REASONS
                    else result.acting_seat
                ),
                actions_by_seat=tuple(actions_by_seat),
                discard_actions=sum(discards_by_seat),
                discards_by_seat=tuple(discards_by_seat),
                real_okey_discards=sum(real_okey_by_seat),
                real_okey_discards_by_seat=tuple(real_okey_by_seat),
                playable_discards=sum(playable_by_seat),
                playable_discards_by_seat=tuple(playable_by_seat),
                immediate_penalties=result.immediate_penalties,
                opened_modes=result.opened_modes,
            )
        raise RuntimeError(
            f"episode seed {episode_seed} exceeded "
            f"{self.training_config.max_actions_per_episode} actions"
        )

    def train(self, episodes: int) -> tuple[EpisodeTrainingResult, ...]:
        if isinstance(episodes, bool) or not isinstance(episodes, int):
            raise TypeError("episodes must be an integer")
        if episodes < 1:
            raise ValueError("episodes must be positive")
        return tuple(self.train_episode() for _ in range(episodes))

    def config_payload(self) -> dict[str, object]:
        return {
            "training": asdict(self.training_config),
            "game": asdict(self.game_config),
        }


def evaluate_against_random(
    model: NumpyActorCritic,
    *,
    episodes: int,
    start_seed: int = 10_000,
    game_config: GameConfig | None = None,
    max_actions_per_episode: int = 1_000,
) -> EvaluationResult:
    """Rotate one greedy learned seat against three reproducible random seats."""

    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = game_config or GameConfig()
    reward_total = 0.0
    score_total = 0
    finishes = 0
    actions = 0
    discard_actions = 0
    real_okey_discards = 0
    playable_discards = 0
    immediate_penalty_points = 0
    penalized_episodes = 0
    opened_series = 0
    opened_pairs = 0
    scores: list[int] = []
    terminal_reasons: dict[str, int] = {}

    for episode_offset in range(episodes):
        episode_seed = start_seed + episode_offset
        learned_seat = episode_offset % config.player_count
        env = SingleRoundEnv(config)
        decision = env.reset(seed=episode_seed)
        random_agents = tuple(
            RandomAgent(seed=episode_seed * config.player_count + seat)
            for seat in range(config.player_count)
        )
        for _ in range(max_actions_per_episode):
            if decision.seat == learned_seat:
                model_input = prepare_model_input(decision, config)
                selected, _value = model.select(
                    model_input,
                    deterministic=True,
                )
                action = _selected_action(decision, selected)
                discard, okey, playable = _discard_flags(
                    decision,
                    action,
                    config,
                )
                discard_actions += discard
                real_okey_discards += okey
                playable_discards += playable
            else:
                action = random_agents[decision.seat].select_action(
                    decision.observation,
                    decision.legal_actions,
                )
            result = env.step(action)
            actions += 1
            if not result.terminated:
                assert result.next_decision is not None
                decision = result.next_decision
                continue
            assert result.final_scores is not None
            assert result.terminal_reason is not None
            assert result.immediate_penalties is not None
            assert result.opened_modes is not None
            reward_total += result.rewards[learned_seat]
            learned_score = result.final_scores[learned_seat]
            score_total += learned_score
            scores.append(learned_score)
            learned_penalty = result.immediate_penalties[learned_seat]
            immediate_penalty_points += learned_penalty
            penalized_episodes += int(learned_penalty > 0)
            opened_series += int(
                result.opened_modes[learned_seat] == "series"
            )
            opened_pairs += int(
                result.opened_modes[learned_seat] == "pairs"
            )
            reason = result.terminal_reason.value
            terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
            if (
                result.terminal_reason not in _NO_WINNER_REASONS
                and result.acting_seat == learned_seat
            ):
                finishes += 1
            break
        else:
            raise RuntimeError(
                f"evaluation seed {episode_seed} exceeded "
                f"{max_actions_per_episode} actions"
            )

    return EvaluationResult(
        episodes=episodes,
        mean_relative_reward=reward_total / episodes,
        mean_score=score_total / episodes,
        finish_rate=finishes / episodes,
        actions=actions,
        discard_actions=discard_actions,
        real_okey_discards=real_okey_discards,
        playable_discards=playable_discards,
        immediate_penalty_points=immediate_penalty_points,
        penalized_episodes=penalized_episodes,
        opened_series=opened_series,
        opened_pairs=opened_pairs,
        best_score=min(scores),
        worst_score=max(scores),
        terminal_reasons=terminal_reasons,
    )
