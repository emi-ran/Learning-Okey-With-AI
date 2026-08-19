"""High-throughput multi-core parallel rollout collector for 101 Okey self-play."""

from __future__ import annotations

import concurrent.futures
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from okey101.engine.config import GameConfig
from okey101.engine.state import TerminalReason
from okey101.rl.action_codec import catalog_from_actions
from okey101.rl.candidate_encoder import SCALAR_NAMES
from okey101.rl.env import Decision, SingleRoundEnv
from okey101.rl.policy import prepare_model_input

from .model import (
    FloatArray,
    NumpyActorCritic,
    candidate_matrix,
    observation_vector,
)
from .trainer import (
    EpisodeTrainingResult,
    SelfPlayTrainer,
    TrainingConfig,
    _discard_flags,
    _NO_WINNER_REASONS,
    _selected_action,
)

if TYPE_CHECKING:
    pass


def _worker_rollout(
    model_params: dict[str, np.ndarray],
    observation_size: int,
    hidden_size: int,
    episode_seed: int,
    starting_player: int,
    training_config: TrainingConfig,
    game_config: GameConfig,
) -> tuple[
    list[tuple[FloatArray, FloatArray, int, float]],
    int,
    tuple[float, ...],
    tuple[int, ...],
    str,
    int | None,
    tuple[int, ...],
    int,
    tuple[int, ...],
    int,
    tuple[int, ...],
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[str, ...],
]:
    rng = np.random.default_rng(episode_seed)
    model = NumpyActorCritic(
        observation_size=observation_size,
        hidden_size=hidden_size,
        rng=rng,
    )
    model.load_parameters(model_params)

    env = SingleRoundEnv(game_config)
    decision = env.reset(seed=episode_seed, starting_player=starting_player)

    trajectory: list[tuple[FloatArray, FloatArray, int, int]] = []
    actions_by_seat = [0] * game_config.player_count
    discards_by_seat = [0] * game_config.player_count
    real_okey_by_seat = [0] * game_config.player_count
    playable_by_seat = [0] * game_config.player_count

    for action_index in range(1, training_config.max_actions_per_episode + 1):
        model_input = prepare_model_input(decision, game_config)
        obs = observation_vector(model_input)
        cands = candidate_matrix(model_input)
        selected, _ = model.select_prepared(obs, cands, rng=rng)
        action = _selected_action(decision, selected)
        discard, okey, playable = _discard_flags(decision, action, game_config)

        actions_by_seat[decision.seat] += 1
        discards_by_seat[decision.seat] += discard
        real_okey_by_seat[decision.seat] += okey
        playable_by_seat[decision.seat] += playable
        trajectory.append((obs, cands, selected, decision.seat))

        result = env.step(action)
        if not result.terminated:
            assert result.next_decision is not None
            decision = result.next_decision
            continue

        samples = [
            (obs, cands, choice, result.rewards[seat])
            for obs, cands, choice, seat in trajectory
        ]
        winner_seat = (
            None
            if result.terminal_reason in _NO_WINNER_REASONS
            else result.acting_seat
        )
        return (
            samples,
            action_index,
            result.rewards,
            result.final_scores or (0, 0, 0, 0),
            result.terminal_reason.value if result.terminal_reason else "unknown",
            winner_seat,
            tuple(actions_by_seat),
            sum(discards_by_seat),
            tuple(discards_by_seat),
            sum(real_okey_by_seat),
            tuple(real_okey_by_seat),
            sum(playable_by_seat),
            tuple(playable_by_seat),
            result.immediate_penalties or (0, 0, 0, 0),
            result.opened_modes or ("none", "none", "none", "none"),
        )

    raise RuntimeError(
        f"episode seed {episode_seed} exceeded max actions"
    )


class ParallelSelfPlayTrainer:
    """Multi-core self-play trainer utilizing CPU worker pools."""

    def __init__(
        self,
        trainer: SelfPlayTrainer,
        max_workers: int | None = None,
    ) -> None:
        self.trainer = trainer
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1))
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )

    def train_batch(self, batch_size: int) -> tuple[EpisodeTrainingResult, ...]:
        """Collect a batch of episodes in parallel across all CPU cores."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        param_copies = {
            name: arr.copy()
            for name, arr in self.trainer.model.parameters.items()
        }
        tasks = []

        for _ in range(batch_size):
            ep_seed = int(
                self.trainer.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
            )
            starting_player = int(
                self.trainer.rng.integers(0, self.trainer.game_config.player_count)
            )
            tasks.append(
                (
                    param_copies,
                    self.trainer.model.observation_size,
                    self.trainer.model.hidden_size,
                    ep_seed,
                    starting_player,
                    self.trainer.training_config,
                    self.trainer.game_config,
                )
            )

        futures = [
            self._executor.submit(_worker_rollout, *task)
            for task in tasks
        ]

        results: list[EpisodeTrainingResult] = []
        for index, future in enumerate(futures):
            (
                samples,
                action_count,
                rewards,
                final_scores,
                terminal_reason,
                winner_seat,
                actions_by_seat,
                discard_actions,
                discards_by_seat,
                real_okey_discards,
                real_okey_discards_by_seat,
                playable_discards,
                playable_discards_by_seat,
                immediate_penalties,
                opened_modes,
            ) = future.result()

            loss, gradients = self.trainer.model.loss_and_gradients_prepared(
                samples,
                value_coefficient=self.trainer.training_config.value_coefficient,
                entropy_coefficient=self.trainer.training_config.entropy_coefficient,
            )
            gradient_norm = self.trainer.optimizer.update(
                self.trainer.model.parameters,
                gradients,
            )
            self.trainer.episodes_completed += 1
            self.trainer.actions_completed += action_count

            results.append(
                EpisodeTrainingResult(
                    episode=self.trainer.episodes_completed,
                    episode_seed=tasks[index][3],
                    actions=action_count,
                    loss=loss,
                    gradient_norm=gradient_norm,
                    rewards=rewards,
                    final_scores=final_scores,
                    terminal_reason=terminal_reason,
                    winner_seat=winner_seat,
                    actions_by_seat=actions_by_seat,
                    discard_actions=discard_actions,
                    discards_by_seat=discards_by_seat,
                    real_okey_discards=real_okey_discards,
                    real_okey_discards_by_seat=real_okey_discards_by_seat,
                    playable_discards=playable_discards,
                    playable_discards_by_seat=playable_discards_by_seat,
                    immediate_penalties=immediate_penalties,
                    opened_modes=opened_modes,
                )
            )

        return tuple(results)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> ParallelSelfPlayTrainer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()
