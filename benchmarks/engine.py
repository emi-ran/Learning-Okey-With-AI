"""Headless RandomAgent benchmark for the deterministic round engine."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.agents.random_agent import RandomAgent
from okey101.engine.invariants import validate_invariants
from okey101.engine.round import RoundEngine


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    rounds: int
    actions: int
    elapsed_seconds: float
    games_per_second: float
    actions_per_second: float
    legal_action_median_ms: float
    legal_action_p95_ms: float
    terminal_reasons: dict[str, int]


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index]


def run_benchmark(
    *,
    rounds: int,
    seed: int = 0,
    validate_every_step: bool = False,
    max_actions_per_round: int = 10_000,
) -> BenchmarkResult:
    """Run reproducible legal-random rounds and return timing statistics."""

    if rounds < 1:
        raise ValueError("rounds must be positive")
    if max_actions_per_round < 1:
        raise ValueError("max_actions_per_round must be positive")

    engine = RoundEngine()
    agents = tuple(RandomAgent(seed + 10_000 + seat) for seat in range(4))
    legal_action_times: list[float] = []
    terminal_reasons: dict[str, int] = {}
    action_count = 0
    started = perf_counter()

    for round_index in range(rounds):
        state = engine.reset(seed + round_index, round_id=round_index + 1)
        validate_invariants(state)
        round_actions = 0
        while not state.terminal:
            if round_actions >= max_actions_per_round:
                raise RuntimeError(
                    f"Round {round_index + 1} exceeded "
                    f"{max_actions_per_round} actions"
                )

            legal_started = perf_counter()
            legal_actions = engine.get_legal_actions()
            legal_action_times.append(perf_counter() - legal_started)
            if not legal_actions:
                raise RuntimeError(
                    f"No legal action at round {round_index + 1}, "
                    f"action {round_actions}, phase {state.phase.value}"
                )

            player_id = state.current_player
            observation = engine.get_observation(player_id)
            action = agents[player_id].select_action(observation, legal_actions)
            state, _events = engine.step(action)
            action_count += 1
            round_actions += 1
            if validate_every_step:
                validate_invariants(state)

        validate_invariants(state)
        assert state.terminal_reason is not None
        reason = state.terminal_reason.value
        terminal_reasons[reason] = terminal_reasons.get(reason, 0) + 1
        engine.get_scores()

    elapsed = perf_counter() - started
    legal_ms = [duration * 1_000 for duration in legal_action_times]
    return BenchmarkResult(
        rounds=rounds,
        actions=action_count,
        elapsed_seconds=elapsed,
        games_per_second=rounds / elapsed,
        actions_per_second=action_count / elapsed,
        legal_action_median_ms=median(legal_ms) if legal_ms else 0.0,
        legal_action_p95_ms=_percentile_95(legal_ms),
        terminal_reasons=terminal_reasons,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-every-step", action="store_true")
    parser.add_argument("--max-actions-per-round", type=int, default=10_000)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    result = run_benchmark(
        rounds=args.rounds,
        seed=args.seed,
        validate_every_step=args.validate_every_step,
        max_actions_per_round=args.max_actions_per_round,
    )
    if args.as_json:
        print(json.dumps(asdict(result), sort_keys=True))
        return

    print(f"rounds: {result.rounds}")
    print(f"actions: {result.actions}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"games_per_second: {result.games_per_second:.3f}")
    print(f"actions_per_second: {result.actions_per_second:.3f}")
    print(f"legal_action_median_ms: {result.legal_action_median_ms:.3f}")
    print(f"legal_action_p95_ms: {result.legal_action_p95_ms:.3f}")
    print(f"terminal_reasons: {result.terminal_reasons}")


if __name__ == "__main__":
    main()
