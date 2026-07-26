"""A dependency-free, policy-safe single-round self-play environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import SystemRandom

from okey101.engine.actions import Action
from okey101.engine.config import GameConfig
from okey101.engine.round import RoundEngine
from okey101.engine.state import TerminalReason

from .observation import PlayerObservation
from .rewards import RewardFn, relative_terminal_rewards


class InvalidEnvironmentAction(ValueError):
    """Raised when a policy submits an action outside its current candidates."""


@dataclass(frozen=True, slots=True)
class Decision:
    """The complete policy-facing input for one engine decision."""

    seat: int
    observation: PlayerObservation
    legal_actions: tuple[Action, ...]
    episode_seed: int


@dataclass(frozen=True, slots=True)
class StepResult:
    """One multi-agent transition without exposing hidden engine state."""

    acting_seat: int
    next_decision: Decision | None
    rewards: tuple[float, ...]
    terminated: bool
    terminal_reason: TerminalReason | None
    final_scores: tuple[int, ...] | None


def _validated_seed(seed: int | None) -> int:
    if seed is None:
        return SystemRandom().getrandbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    return seed


class SingleRoundEnv:
    """Turn-based four-seat environment over one deterministic ``RoundEngine``."""

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        reward_fn: RewardFn = relative_terminal_rewards,
    ) -> None:
        self.config = config or GameConfig()
        if self.config.rounds != 1:
            raise ValueError("SingleRoundEnv requires GameConfig(rounds=1)")
        if not callable(reward_fn):
            raise TypeError("reward_fn must be callable")

        self._reward_fn = reward_fn
        self._engine = RoundEngine(self.config)
        self._episode_seed: int | None = None
        self._decision: Decision | None = None

    @property
    def episode_seed(self) -> int | None:
        return self._episode_seed

    @property
    def terminated(self) -> bool:
        return self._engine.is_terminal()

    @property
    def num_players(self) -> int:
        return self.config.player_count

    @property
    def current_decision(self) -> Decision | None:
        return self._decision

    def _make_decision(self) -> Decision:
        state = self._engine.state
        if state is None:
            raise RuntimeError("reset() must be called before requesting a decision")
        if state.terminal:
            raise RuntimeError("Terminal round has no active decision")
        assert self._episode_seed is not None
        seat = state.current_player
        actions = self._engine.get_legal_actions(seat)
        if not actions:
            raise RuntimeError("Non-terminal round has no legal actions")
        return Decision(
            seat=seat,
            observation=self._engine.get_observation(seat),
            legal_actions=actions,
            episode_seed=self._episode_seed,
        )

    def reset(
        self,
        seed: int | None = None,
        *,
        starting_player: int | None = None,
    ) -> Decision:
        actual_seed = _validated_seed(seed)
        if starting_player is not None and (
            isinstance(starting_player, bool)
            or not isinstance(starting_player, int)
        ):
            raise TypeError("starting_player must be an integer or None")
        if starting_player is not None and not (
            0 <= starting_player < self.num_players
        ):
            raise ValueError(
                "starting_player is outside the configured seat range"
            )

        self._engine.reset(
            seed=actual_seed,
            starting_player=starting_player,
        )
        self._episode_seed = actual_seed
        self._decision = self._make_decision()
        return self._decision

    def validate_action(self, action: Action) -> None:
        """Validate without mutating, allowing vector-wide atomic prechecks."""

        if self._decision is None:
            if self.terminated:
                raise RuntimeError("Terminal round cannot accept actions")
            raise RuntimeError("reset() must be called before step()")
        if action not in self._decision.legal_actions:
            raise InvalidEnvironmentAction(
                "Action is not one of the current legal candidates"
            )

    def _terminal_rewards(self, scores: tuple[int, ...]) -> tuple[float, ...]:
        raw_rewards = self._reward_fn(scores)
        try:
            rewards = tuple(float(reward) for reward in raw_rewards)
        except (TypeError, ValueError) as error:
            raise ValueError("reward_fn must return numeric rewards") from error
        if len(rewards) != self.num_players:
            raise ValueError(
                "reward_fn returned a reward count that does not match player_count"
            )
        if not all(isfinite(reward) for reward in rewards):
            raise ValueError("reward_fn returned a non-finite reward")
        return rewards

    def step(self, action: Action) -> StepResult:
        self.validate_action(action)
        assert self._decision is not None
        acting_seat = self._decision.seat

        state, _events = self._engine.step(action)
        if not state.terminal:
            self._decision = self._make_decision()
            return StepResult(
                acting_seat=acting_seat,
                next_decision=self._decision,
                rewards=(0.0,) * self.num_players,
                terminated=False,
                terminal_reason=None,
                final_scores=None,
            )

        scores = self._engine.get_scores()
        self._decision = None
        rewards = self._terminal_rewards(scores)
        return StepResult(
            acting_seat=acting_seat,
            next_decision=None,
            rewards=rewards,
            terminated=True,
            terminal_reason=state.terminal_reason,
            final_scores=scores,
        )
