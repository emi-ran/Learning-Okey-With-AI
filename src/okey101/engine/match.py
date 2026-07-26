"""Deterministic multi-round match orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from random import SystemRandom
from typing import Any, NoReturn

from .actions import Action
from .config import GameConfig, ScoringConfig
from .invariants import InvariantError, validate_invariants
from .round import (
    RoundEngine,
    _to_primitive,
    deserialize_action,
    deserialize_state,
    serialize_action,
)
from .scoring import calculate_round_scores
from .state import EngineEvent, GameState, TerminalReason

_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_SNAPSHOT_VERSION = 1


def _invalid(path: str, message: str) -> NoReturn:
    raise ValueError(f"Invalid match state at {path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid(path, "expected an object")
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        _invalid(path, "expected an array")
    return list(value)


def _required(data: Mapping[str, object], key: str, path: str) -> object:
    if key not in data:
        _invalid(path, f"missing field {key!r}")
    return data[key]


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(path, "expected an integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _invalid(path, "expected a boolean")
    return value


def _enum(enum_type: type[Enum], value: object, path: str) -> Enum:
    if not isinstance(value, str):
        _invalid(path, "expected an enum string")
    try:
        return enum_type(value)
    except ValueError as error:
        _invalid(path, str(error))


def _strict_fields(
    data: Mapping[str, object],
    allowed: set[str],
    path: str,
) -> None:
    unexpected = sorted(set(data) - allowed)
    if unexpected:
        _invalid(path, f"unexpected fields: {unexpected}")


def _dataclass_config_values(
    payload: object,
    default: object,
    path: str,
    *,
    excluded: set[str] = frozenset(),
) -> dict[str, object]:
    data = _mapping(payload, path)
    names = {field.name for field in fields(default)} - excluded
    _strict_fields(data, names, path)
    result: dict[str, object] = {}
    for field_name in names:
        raw = _required(data, field_name, path)
        field_path = f"{path}.{field_name}"
        default_value = getattr(default, field_name)
        if isinstance(default_value, bool):
            result[field_name] = _boolean(raw, field_path)
        elif isinstance(default_value, int):
            result[field_name] = _integer(raw, field_path)
        else:
            _invalid(field_path, "unsupported config field type")
    return result


def _deserialize_config(payload: object, path: str) -> GameConfig:
    data = _mapping(payload, path)
    game_field_names = {field.name for field in fields(GameConfig())}
    _strict_fields(data, game_field_names, path)
    game_values = _dataclass_config_values(
        {key: value for key, value in data.items() if key != "scoring"},
        GameConfig(),
        path,
        excluded={"scoring"},
    )
    scoring_values = _dataclass_config_values(
        _required(data, "scoring", path),
        ScoringConfig(),
        f"{path}.scoring",
    )
    try:
        return GameConfig(
            **game_values,
            scoring=ScoringConfig(**scoring_values),
        )
    except (TypeError, ValueError) as error:
        _invalid(path, str(error))


def _deserialize_game_state(payload: object, path: str) -> GameState:
    try:
        state = deserialize_state(_mapping(payload, path))
        validate_invariants(state)
    except (InvariantError, TypeError, ValueError) as error:
        message = str(error).replace("$", path, 1)
        _invalid(path, message)
    return state


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


class MatchReplayError(RuntimeError):
    def __init__(
        self,
        action_index: int,
        action: Action,
        round_id: int,
        reason: str,
    ) -> None:
        self.action_index = action_index
        self.action = action
        self.round_id = round_id
        super().__init__(
            f"Match replay failed at action {action_index} in round "
            f"{round_id} ({type(action).__name__}): {reason}"
        )


def _deserialize_round_record(payload: object, path: str) -> RoundRecord:
    data = _mapping(payload, path)
    allowed = {
        "round_id",
        "seed",
        "starting_player",
        "terminal_reason",
        "scores",
        "counts_toward_match",
        "terminal_state",
    }
    _strict_fields(data, allowed, path)
    reason = _enum(
        TerminalReason,
        _required(data, "terminal_reason", path),
        f"{path}.terminal_reason",
    )
    state = _deserialize_game_state(
        _required(data, "terminal_state", path),
        f"{path}.terminal_state",
    )
    scores = tuple(
        _integer(value, f"{path}.scores[{index}]")
        for index, value in enumerate(
            _sequence(_required(data, "scores", path), f"{path}.scores")
        )
    )
    return RoundRecord(
        round_id=_integer(
            _required(data, "round_id", path),
            f"{path}.round_id",
        ),
        seed=_integer(_required(data, "seed", path), f"{path}.seed"),
        starting_player=_integer(
            _required(data, "starting_player", path),
            f"{path}.starting_player",
        ),
        terminal_reason=reason,
        scores=scores,
        counts_toward_match=_boolean(
            _required(data, "counts_toward_match", path),
            f"{path}.counts_toward_match",
        ),
        terminal_state=state,
    )


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

    def serialize_state(self) -> dict[str, Any]:
        """Return a self-contained JSON-compatible match snapshot."""

        if not self._has_reset:
            raise RuntimeError("reset() must be called before serialize_state()")
        assert self.seed is not None
        assert self.initial_starting_player is not None
        assert self.current_round_seed is not None
        return {
            "version": _SNAPSHOT_VERSION,
            "config": _to_primitive(self.config),
            "seed": self.seed,
            "initial_starting_player": self.initial_starting_player,
            "current_round_seed": self.current_round_seed,
            "completed_rounds": self.completed_rounds,
            "round_attempts": self.rounds_played,
            "scores": list(self._scores),
            "terminal": self._terminal,
            "round_records": [
                _to_primitive(record)
                for record in self.round_records
            ],
            "current_round": self.round_engine.serialize_state(),
            "action_history": [
                {
                    "round_id": round_id,
                    "action": serialize_action(action),
                }
                for round_id, action in self.action_history
            ],
        }

    def load_state(self, payload: Mapping[str, object]) -> GameState:
        """Load and replay-verify a JSON-compatible match snapshot."""

        data = _mapping(payload, "$")
        allowed = {
            "version",
            "config",
            "seed",
            "initial_starting_player",
            "current_round_seed",
            "completed_rounds",
            "round_attempts",
            "scores",
            "terminal",
            "round_records",
            "current_round",
            "action_history",
        }
        _strict_fields(data, allowed, "$")
        version = _integer(_required(data, "version", "$"), "$.version")
        if version != _SNAPSHOT_VERSION:
            _invalid("$.version", f"unsupported version {version}")

        config = _deserialize_config(_required(data, "config", "$"), "$.config")
        seed = _integer(_required(data, "seed", "$"), "$.seed")
        initial_starting_player = _integer(
            _required(data, "initial_starting_player", "$"),
            "$.initial_starting_player",
        )
        if not 0 <= initial_starting_player < config.player_count:
            _invalid(
                "$.initial_starting_player",
                "outside the configured seat range",
            )
        current_round_seed = _integer(
            _required(data, "current_round_seed", "$"),
            "$.current_round_seed",
        )
        completed_rounds = _integer(
            _required(data, "completed_rounds", "$"),
            "$.completed_rounds",
        )
        if completed_rounds < 0:
            _invalid("$.completed_rounds", "cannot be negative")
        if completed_rounds > config.rounds:
            _invalid(
                "$.completed_rounds",
                f"cannot exceed configured rounds ({config.rounds})",
            )
        round_attempts = _integer(
            _required(data, "round_attempts", "$"),
            "$.round_attempts",
        )
        if round_attempts < 0:
            _invalid("$.round_attempts", "cannot be negative")
        scores = tuple(
            _integer(value, f"$.scores[{index}]")
            for index, value in enumerate(
                _sequence(_required(data, "scores", "$"), "$.scores")
            )
        )
        if len(scores) != config.player_count:
            _invalid(
                "$.scores",
                f"expected {config.player_count} seat scores",
            )
        terminal = _boolean(_required(data, "terminal", "$"), "$.terminal")
        current_round = _deserialize_game_state(
            _required(data, "current_round", "$"),
            "$.current_round",
        )
        if len(current_round.players) != config.player_count:
            _invalid(
                "$.current_round.players",
                f"expected {config.player_count} players",
            )

        records = tuple(
            _deserialize_round_record(item, f"$.round_records[{index}]")
            for index, item in enumerate(
                _sequence(
                    _required(data, "round_records", "$"),
                    "$.round_records",
                )
            )
        )
        if round_attempts != len(records):
            _invalid(
                "$.round_attempts",
                f"expected {len(records)} from round_records",
            )

        parsed_actions: list[tuple[int, Action]] = []
        for index, item in enumerate(
            _sequence(
                _required(data, "action_history", "$"),
                "$.action_history",
            )
        ):
            item_path = f"$.action_history[{index}]"
            item_data = _mapping(item, item_path)
            _strict_fields(item_data, {"round_id", "action"}, item_path)
            round_id = _integer(
                _required(item_data, "round_id", item_path),
                f"{item_path}.round_id",
            )
            try:
                action = deserialize_action(
                    _mapping(
                        _required(item_data, "action", item_path),
                        f"{item_path}.action",
                    )
                )
            except (TypeError, ValueError) as error:
                _invalid(f"{item_path}.action", str(error))
            parsed_actions.append((round_id, action))

        running_scores = tuple(0 for _ in range(config.player_count))
        expected_completed = 0
        for index, record in enumerate(records):
            path = f"$.round_records[{index}]"
            expected_round_id = index + 1
            if record.round_id != expected_round_id:
                _invalid(
                    f"{path}.round_id",
                    f"expected {expected_round_id}",
                )
            expected_seed = derive_round_seed(seed, index)
            if record.seed != expected_seed:
                _invalid(f"{path}.seed", f"expected {expected_seed}")
            expected_starter = (
                initial_starting_player + index
            ) % config.player_count
            if record.starting_player != expected_starter:
                _invalid(
                    f"{path}.starting_player",
                    f"expected {expected_starter}",
                )
            if record.terminal_state.round_id != record.round_id:
                _invalid(
                    f"{path}.terminal_state.round_id",
                    "does not match record round_id",
                )
            if record.terminal_state.starting_player != record.starting_player:
                _invalid(
                    f"{path}.terminal_state.starting_player",
                    "does not match record starting_player",
                )
            if record.terminal_state.terminal_reason is not record.terminal_reason:
                _invalid(
                    f"{path}.terminal_state.terminal_reason",
                    "does not match record terminal_reason",
                )
            if len(record.terminal_state.players) != config.player_count:
                _invalid(
                    f"{path}.terminal_state.players",
                    f"expected {config.player_count} players",
                )
            expected_scores = calculate_round_scores(
                record.terminal_state,
                config.scoring,
            )
            if record.scores != expected_scores:
                _invalid(f"{path}.scores", f"expected {expected_scores}")
            is_void = (
                record.terminal_reason
                is TerminalReason.ALL_PLAYERS_OPENED_PAIRS
            )
            expected_counts = (
                not is_void or config.void_round_counts_toward_match
            )
            if record.counts_toward_match is not expected_counts:
                _invalid(
                    f"{path}.counts_toward_match",
                    f"expected {expected_counts}",
                )
            if expected_counts:
                expected_completed += 1
            running_scores = tuple(
                total + round_score
                for total, round_score in zip(
                    running_scores,
                    record.scores,
                    strict=True,
                )
            )
            if tuple(
                player.score for player in record.terminal_state.players
            ) != running_scores:
                _invalid(
                    f"{path}.terminal_state.players",
                    "player scores do not match cumulative totals",
                )

        if completed_rounds != expected_completed:
            _invalid(
                "$.completed_rounds",
                f"expected {expected_completed} from round_records",
            )
        if scores != running_scores:
            _invalid("$.scores", f"expected cumulative totals {running_scores}")
        if terminal != (completed_rounds >= config.rounds):
            _invalid(
                "$.terminal",
                "does not match completed round quota",
            )

        if terminal and not records:
            _invalid(
                "$.round_records",
                "terminal match requires at least one completed record",
            )
        expected_current_id = len(records) if terminal else len(records) + 1
        if current_round.round_id != expected_current_id:
            _invalid(
                "$.current_round.round_id",
                f"expected {expected_current_id}",
            )
        current_index = current_round.round_id - 1
        expected_current_seed = derive_round_seed(seed, current_index)
        if current_round_seed != expected_current_seed:
            _invalid(
                "$.current_round_seed",
                f"expected {expected_current_seed}",
            )
        expected_current_starter = (
            initial_starting_player + current_index
        ) % config.player_count
        if current_round.starting_player != expected_current_starter:
            _invalid(
                "$.current_round.starting_player",
                f"expected {expected_current_starter}",
            )
        if tuple(player.score for player in current_round.players) != scores:
            _invalid(
                "$.current_round.players",
                "player scores do not match cumulative totals",
            )
        if terminal:
            if not records or current_round != records[-1].terminal_state:
                _invalid(
                    "$.current_round",
                    "terminal match must retain its last terminal record state",
                )
        elif current_round.terminal:
            _invalid(
                "$.current_round.terminal",
                "active match round cannot be terminal",
            )

        try:
            replayed = replay_match_from_seed_and_actions(
                seed,
                (action for _round_id, action in parsed_actions),
                config=config,
                starting_player=initial_starting_player,
            )
        except MatchReplayError as error:
            _invalid("$.action_history", str(error))
        if tuple(replayed.action_history) != tuple(parsed_actions):
            _invalid(
                "$.action_history",
                "round boundaries do not match deterministic replay",
            )
        if replayed.current_round != current_round:
            _invalid(
                "$.current_round",
                "does not match deterministic action replay",
            )
        if tuple(replayed.round_records) != records:
            _invalid(
                "$.round_records",
                "do not match deterministic action replay",
            )
        if replayed.get_scores() != scores:
            _invalid("$.scores", "do not match deterministic action replay")
        if replayed.completed_rounds != completed_rounds:
            _invalid(
                "$.completed_rounds",
                "does not match deterministic action replay",
            )
        if replayed.is_terminal() is not terminal:
            _invalid("$.terminal", "does not match deterministic action replay")

        self.config = replayed.config
        self.round_engine = replayed.round_engine
        self.seed = replayed.seed
        self.initial_starting_player = replayed.initial_starting_player
        self.current_round_seed = replayed.current_round_seed
        self.completed_rounds = replayed.completed_rounds
        self.round_records = list(replayed.round_records)
        self.action_history = list(replayed.action_history)
        self.event_log = list(replayed.event_log)
        self._scores = replayed._scores
        self._terminal = replayed._terminal
        self._has_reset = True
        return self.current_round

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


def replay_match_from_seed_and_actions(
    seed: int,
    actions: Iterable[Action],
    *,
    config: GameConfig | None = None,
    starting_player: int = 0,
) -> MatchEngine:
    """Replay one flat typed action sequence across automatic round boundaries.

    Callers pass actions in the exact order originally selected; no explicit
    round separator is needed because ``MatchEngine.step`` deals the next round
    after each non-final terminal action. The returned engine represents the
    complete supplied prefix, whether it ends mid-round or at match terminal.
    """

    engine = MatchEngine(config)
    engine.reset(seed=seed, starting_player=starting_player)
    for index, action in enumerate(actions):
        round_id = engine.current_round.round_id
        try:
            engine.step(action)
        except (RuntimeError, TypeError, ValueError) as error:
            raise MatchReplayError(
                index,
                action,
                round_id,
                str(error),
            ) from error
    return engine
