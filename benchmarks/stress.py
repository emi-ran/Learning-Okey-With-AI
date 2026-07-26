"""Parallel invariant stress runner with reproducible failure artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.agents.random_agent import RandomAgent
from okey101.engine.invariants import validate_invariants
from okey101.engine.round import RoundEngine


@dataclass(frozen=True, slots=True)
class StressFailure:
    seed: int
    action_index: int
    error_type: str
    error_message: str
    traceback: str
    actions: list[Any]
    failed_action: Any | None
    state: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class WorkerResult:
    rounds: int
    actions: int
    terminal_reasons: dict[str, int]
    failures: tuple[StressFailure, ...]


@dataclass(frozen=True, slots=True)
class StressResult:
    requested_rounds: int
    completed_rounds: int
    actions: int
    elapsed_seconds: float
    games_per_second: float
    actions_per_second: float
    workers: int
    terminal_reasons: dict[str, int]
    failure_count: int


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    return value


def _run_seed_range(
    start_seed: int,
    count: int,
    *,
    validate_every_step: bool,
    max_actions_per_round: int,
) -> WorkerResult:
    engine = RoundEngine()
    actions = 0
    completed = 0
    terminal_reasons: dict[str, int] = {}
    failures: list[StressFailure] = []

    for offset in range(count):
        seed = start_seed + offset
        action_index = 0
        attempted_action: object | None = None
        pre_action_state: dict[str, Any] | None = None
        history_length_before = 0
        agents = tuple(
            RandomAgent(seed * 17 + 100_000 + seat) for seat in range(4)
        )
        try:
            state = engine.reset(seed, round_id=seed + 1)
            validate_invariants(state)
            while not state.terminal:
                if action_index >= max_actions_per_round:
                    raise RuntimeError(
                        f"Round exceeded {max_actions_per_round} actions"
                    )
                legal_actions = engine.get_legal_actions()
                if not legal_actions:
                    raise RuntimeError(
                        f"No legal actions in phase {state.phase.value}"
                    )
                player_id = state.current_player
                observation = engine.get_observation(player_id)
                attempted_action = agents[player_id].select_action(
                    observation,
                    legal_actions,
                )
                pre_action_state = engine.serialize_state()
                history_length_before = len(engine.action_history)
                state, _events = engine.step(attempted_action)
                actions += 1
                if validate_every_step:
                    validate_invariants(state)
                action_index += 1
                attempted_action = None
                pre_action_state = None

            validate_invariants(state)
            engine.get_scores()
            assert state.terminal_reason is not None
            reason = state.terminal_reason.value
            terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
            completed += 1
        except Exception as error:  # failure artifact is the point of this runner
            recorded_actions = list(engine.action_history)
            if (
                attempted_action is not None
                and len(recorded_actions) == history_length_before
            ):
                recorded_actions.append(attempted_action)
            failures.append(
                StressFailure(
                    seed=seed,
                    action_index=action_index,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback=traceback.format_exc(),
                    actions=_primitive(recorded_actions),
                    failed_action=_primitive(attempted_action),
                    state=(
                        pre_action_state
                        if pre_action_state is not None
                        else (
                            engine.serialize_state()
                            if engine.state is not None
                            else None
                        )
                    ),
                )
            )

    return WorkerResult(
        rounds=completed,
        actions=actions,
        terminal_reasons=terminal_reasons,
        failures=tuple(failures),
    )


def _chunks(rounds: int, workers: int, start_seed: int) -> list[tuple[int, int]]:
    base, remainder = divmod(rounds, workers)
    chunks: list[tuple[int, int]] = []
    next_seed = start_seed
    for worker_index in range(workers):
        count = base + int(worker_index < remainder)
        if count:
            chunks.append((next_seed, count))
            next_seed += count
    return chunks


def run_stress(
    *,
    rounds: int,
    start_seed: int = 0,
    workers: int = 1,
    validate_every_step: bool = True,
    max_actions_per_round: int = 10_000,
    failure_directory: Path | None = None,
) -> StressResult:
    """Run independent seed ranges and persist every reproducible failure."""

    if rounds < 1:
        raise ValueError("rounds must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    workers = min(workers, rounds)
    failure_directory = (
        failure_directory
        or _PROJECT_ROOT / "artifacts" / "stress-failures"
    )
    started = perf_counter()
    work = _chunks(rounds, workers, start_seed)

    if workers == 1:
        results = (
            _run_seed_range(
                work[0][0],
                work[0][1],
                validate_every_step=validate_every_step,
                max_actions_per_round=max_actions_per_round,
            ),
        )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _run_seed_range,
                    seed,
                    count,
                    validate_every_step=validate_every_step,
                    max_actions_per_round=max_actions_per_round,
                )
                for seed, count in work
            ]
            results = tuple(future.result() for future in futures)

    elapsed = perf_counter() - started
    completed = sum(result.rounds for result in results)
    actions = sum(result.actions for result in results)
    failures = tuple(
        failure for result in results for failure in result.failures
    )
    terminal_reasons: dict[str, int] = {}
    for result in results:
        for reason, count in result.terminal_reasons.items():
            terminal_reasons[reason] = terminal_reasons.get(reason, 0) + count

    if failures:
        failure_directory.mkdir(parents=True, exist_ok=True)
        for failure in failures:
            path = failure_directory / f"seed-{failure.seed}.json"
            path.write_text(
                json.dumps(asdict(failure), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    return StressResult(
        requested_rounds=rounds,
        completed_rounds=completed,
        actions=actions,
        elapsed_seconds=elapsed,
        games_per_second=completed / elapsed,
        actions_per_second=actions / elapsed,
        workers=workers,
        terminal_reasons=terminal_reasons,
        failure_count=len(failures),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--no-step-validation", action="store_true")
    parser.add_argument("--max-actions-per-round", type=int, default=10_000)
    parser.add_argument("--failure-directory", type=Path)
    args = parser.parse_args()

    result = run_stress(
        rounds=args.rounds,
        start_seed=args.start_seed,
        workers=args.workers,
        validate_every_step=not args.no_step_validation,
        max_actions_per_round=args.max_actions_per_round,
        failure_directory=args.failure_directory,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    if result.failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
