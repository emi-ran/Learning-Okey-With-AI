"""Sequential batching for independent single-round environments."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from okey101.engine.actions import Action
from okey101.engine.config import GameConfig

from .env import Decision, SingleRoundEnv, StepResult, _validated_seed
from .rewards import RewardFn, relative_terminal_rewards


class VectorRoundEnv:
    """A deterministic batch API without claiming parallel execution."""

    def __init__(
        self,
        num_envs: int,
        config: GameConfig | None = None,
        *,
        reward_fn: RewardFn = relative_terminal_rewards,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int):
            raise TypeError("num_envs must be an integer")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")

        self.num_envs = num_envs
        self.config = config or GameConfig()
        self._envs = tuple(
            SingleRoundEnv(self.config, reward_fn=reward_fn)
            for _ in range(num_envs)
        )
        self._base_seed: int | None = None

    @property
    def base_seed(self) -> int | None:
        return self._base_seed

    @property
    def episode_seeds(self) -> tuple[int | None, ...]:
        return tuple(env.episode_seed for env in self._envs)

    @property
    def terminated(self) -> tuple[bool, ...]:
        return tuple(env.terminated for env in self._envs)

    @property
    def current_decisions(self) -> tuple[Decision | None, ...]:
        return tuple(env.current_decision for env in self._envs)

    def reset(
        self,
        seed: int | None = None,
        *,
        starting_players: Sequence[int | None] | None = None,
    ) -> tuple[Decision, ...]:
        actual_seed = _validated_seed(seed)
        if starting_players is None:
            starters: tuple[int | None, ...] = (None,) * self.num_envs
        else:
            starters = tuple(starting_players)
            if len(starters) != self.num_envs:
                raise ValueError(
                    "starting_players length must match num_envs"
                )
            for starter in starters:
                if starter is not None and (
                    isinstance(starter, bool) or not isinstance(starter, int)
                ):
                    raise TypeError(
                        "starting_players must contain integers or None"
                    )
                if starter is not None and not (
                    0 <= starter < self.config.player_count
                ):
                    raise ValueError(
                        "starting player is outside the configured seat range"
                    )

        seed_stream = Random(actual_seed)
        episode_seeds = tuple(
            seed_stream.getrandbits(64)
            for _ in range(self.num_envs)
        )
        self._base_seed = actual_seed
        return tuple(
            env.reset(episode_seed, starting_player=starter)
            for env, episode_seed, starter in zip(
                self._envs,
                episode_seeds,
                starters,
                strict=True,
            )
        )

    def reset_at(
        self,
        index: int,
        seed: int | None = None,
        *,
        starting_player: int | None = None,
    ) -> Decision:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.num_envs:
            raise IndexError("environment index is out of range")
        return self._envs[index].reset(
            seed,
            starting_player=starting_player,
        )

    def step(
        self,
        actions: Sequence[Action],
    ) -> tuple[StepResult, ...]:
        candidates = tuple(actions)
        if len(candidates) != self.num_envs:
            raise ValueError("actions length must match num_envs")

        # Complete every fallible policy-level check before any engine mutates.
        for env, action in zip(self._envs, candidates, strict=True):
            env.validate_action(action)

        return tuple(
            env.step(action)
            for env, action in zip(self._envs, candidates, strict=True)
        )
