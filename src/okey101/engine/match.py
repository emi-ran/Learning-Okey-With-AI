"""Deterministic multi-round match orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import SystemRandom

from .actions import Action
from .config import GameConfig
from .round import RoundEngine
from .scoring import calculate_round_scores
from .state import EngineEvent, GameState, TerminalReason

_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15


def derive_round_seed(match_seed: int, round_index: int) -> int:
    """Derive a stable 64-bit seed for one zero-based round attempt."""

    if isinstance(match_seed, bool) or not isinstance(match_seed, int):
        raise TypeError("match_seed must be an integer")
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        raise TypeError("round_index must be an integer")
    if round_index < 0:
        raise ValueError("round_index cannot be negative")

    value = (
        (match_seed & _MASK_64)
        + _GOLDEN_GAMMA * (round_index + 1)
    ) & _MASK_64
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & _MASK_64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & _MASK_64
    return value ^ (value >> 31)


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """Immutable result of one dealt round, including void attempts."""

    round_id: int
    seed: int
    starting_player: int
    terminal_reason: TerminalReason
    scores: tuple[int, ...]
    counts_toward_match: bool
    terminal_state: GameState


class MatchEngine:
    """Manage deterministic rounds while preserving ``RoundEngine`` semantics."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.round_engine = RoundEngine(self.config)
        self.seed: int | None = None
        self.initial_starting_player: int | None = None
        self.current_round_seed: int | None = None
        self.completed_rounds = 0
        self.round_records: list[RoundRecord] = []
        self.action_history: list[tuple[int, Action]] = []
        self.event_log: list[EngineEvent] = []
        self._scores = tuple(0 for _ in range(self.config.player_count))
        self._terminal = False
        self._has_reset = False

    @property
    def current_round(self) -> GameState:
        if self.round_engine.state is None:
            raise RuntimeError("reset() must be called before accessing current_round")
        return self.round_engine.state

    @property
    def rounds_played(self) -> int:
        """Number of dealt attempts, including void rounds."""

        return len(self.round_records)

    def _starting_player_for_attempt(self, attempt_index: int) -> int:
        assert self.initial_starting_player is not None
        return (
            self.initial_starting_player + attempt_index
        ) % self.config.player_count

    def _start_round(self, attempt_index: int) -> tuple[GameState, tuple[EngineEvent, ...]]:
        assert self.seed is not None
        round_seed = derive_round_seed(self.seed, attempt_index)
        starting_player = self._starting_player_for_attempt(attempt_index)
        state = self.round_engine.reset(
            seed=round_seed,
            round_id=attempt_index + 1,
            starting_player=starting_player,
        )
        state = replace(
            state,
            players=tuple(
                replace(player, score=self._scores[player_id])
                for player_id, player in enumerate(state.players)
            ),
        )
        self.round_engine.state = state
        events = tuple(self.round_engine.event_log)
        self.current_round_seed = round_seed
        self.event_log.extend(events)
        return state, events

    def reset(
        self,
        seed: int | None = None,
        *,
        starting_player: int = 0,
    ) -> GameState:
        """Reset the match and deal its first round.

        When ``seed`` is omitted, the generated seed is retained on ``self.seed``
        so the exact match can still be replayed.
        """

        if not 0 <= starting_player < self.config.player_count:
            raise ValueError("starting_player is outside the configured seat range")
        self.seed = (
            SystemRandom().getrandbits(64)
            if seed is None
            else seed
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer or None")
        self.initial_starting_player = starting_player
        self.current_round_seed = None
        self.completed_rounds = 0
        self.round_records.clear()
        self.action_history.clear()
        self.event_log.clear()
        self._scores = tuple(0 for _ in range(self.config.player_count))
        self._terminal = False
        self._has_reset = True
        state, _ = self._start_round(0)
        return state

    def _record_terminal_round(self, state: GameState) -> RoundRecord:
        reason = state.terminal_reason
        if not state.terminal or reason is None:
            raise RuntimeError("cannot record a non-terminal round")
        assert self.current_round_seed is not None

        scores = calculate_round_scores(state, self.config.scoring)
        is_void = reason is TerminalReason.ALL_PLAYERS_OPENED_PAIRS
        counts_toward_match = (
            not is_void or self.config.void_round_counts_toward_match
        )
        if counts_toward_match:
            self.completed_rounds += 1
        self._scores = tuple(
            total + round_score
            for total, round_score in zip(self._scores, scores, strict=True)
        )
        scored_state = replace(
            state,
            players=tuple(
                replace(player, score=self._scores[player_id])
                for player_id, player in enumerate(state.players)
            ),
        )
        self.round_engine.state = scored_state
        record = RoundRecord(
            round_id=scored_state.round_id,
            seed=self.current_round_seed,
            starting_player=scored_state.starting_player,
            terminal_reason=reason,
            scores=scores,
            counts_toward_match=counts_toward_match,
            terminal_state=scored_state,
        )
        self.round_records.append(record)
        return record

    def step(self, action: Action) -> tuple[GameState, tuple[EngineEvent, ...]]:
        if not self._has_reset:
            raise RuntimeError("reset() must be called before step()")
        if self._terminal:
            raise RuntimeError("Terminal match cannot accept actions")

        round_id = self.current_round.round_id
        state, events = self.round_engine.step(action)
        self.action_history.append((round_id, action))
        self.event_log.extend(events)
        if not state.terminal:
            return state, events

        record = self._record_terminal_round(state)
        state = record.terminal_state
        if self.completed_rounds >= self.config.rounds:
            self._terminal = True
            return state, events

        next_state, deal_events = self._start_round(len(self.round_records))
        return next_state, (*events, *deal_events)

    def get_legal_actions(self, player_id: int | None = None) -> tuple[Action, ...]:
        if self._terminal:
            return ()
        return self.round_engine.get_legal_actions(player_id)

    def get_observation(self, player_id: int):
        if self._terminal:
            raise RuntimeError("Terminal match has no active observation")
        return self.round_engine.get_observation(player_id)

    def is_terminal(self) -> bool:
        return self._terminal

    def get_scores(self) -> tuple[int, ...]:
        if not self._has_reset:
            raise RuntimeError("reset() must be called before get_scores()")
        return self._scores
