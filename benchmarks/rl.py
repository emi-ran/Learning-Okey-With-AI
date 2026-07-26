"""RL-interface throughput and legal-candidate distribution benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.engine.config import GameConfig
from okey101.rl.action_codec import catalog_from_actions
from okey101.rl.encoder import encode_observation
from okey101.rl.env import SingleRoundEnv
from okey101.rl.policy import prepare_model_input


@dataclass(frozen=True, slots=True)
class RlBenchmarkResult:
    episodes: int
    actions: int
    elapsed_seconds: float
    episodes_per_second: float
    actions_per_second: float
    candidate_count_median: float
    candidate_count_p95: int
    candidate_count_max: int
    observation_encode_median_ms: float
    observation_encode_p95_ms: float
    action_catalog_median_ms: float
    action_catalog_p95_ms: float
    model_input_median_ms: float
    model_input_p95_ms: float
    model_input_max_ms: float


def _p95(values: list[float] | list[int]) -> float | int:
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * 0.95) - 1]


def run_rl_benchmark(
    *,
    episodes: int,
    start_seed: int = 0,
    max_actions_per_episode: int = 10_000,
) -> RlBenchmarkResult:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if max_actions_per_episode < 1:
        raise ValueError("max_actions_per_episode must be positive")

    config = GameConfig()
    env = SingleRoundEnv(config)
    candidate_counts: list[int] = []
    observation_times: list[float] = []
    catalog_times: list[float] = []
    model_input_times: list[float] = []
    action_count = 0
    started = perf_counter()

    for seed in range(start_seed, start_seed + episodes):
        decision = env.reset(seed=seed)
        for episode_step in range(max_actions_per_episode):
            encode_started = perf_counter()
            encode_observation(decision.observation, config)
            observation_times.append((perf_counter() - encode_started) * 1_000)

            catalog_started = perf_counter()
            catalog = catalog_from_actions(decision.legal_actions)
            catalog_times.append((perf_counter() - catalog_started) * 1_000)
            candidate_counts.append(len(catalog))

            model_started = perf_counter()
            model_input = prepare_model_input(decision, config)
            model_input_times.append((perf_counter() - model_started) * 1_000)
            if sum(model_input.action_mask) != len(catalog):
                raise RuntimeError("model input lost legal action candidates")

            candidate_id = (
                episode_step * 7 + decision.seat
            ) % len(catalog)
            result = env.step(catalog.decode(candidate_id))
            action_count += 1
            if result.terminated:
                break
            assert result.next_decision is not None
            decision = result.next_decision
        else:
            raise RuntimeError(
                f"episode seed {seed} exceeded {max_actions_per_episode} actions"
            )

    elapsed = perf_counter() - started
    return RlBenchmarkResult(
        episodes=episodes,
        actions=action_count,
        elapsed_seconds=elapsed,
        episodes_per_second=episodes / elapsed,
        actions_per_second=action_count / elapsed,
        candidate_count_median=median(candidate_counts),
        candidate_count_p95=int(_p95(candidate_counts)),
        candidate_count_max=max(candidate_counts),
        observation_encode_median_ms=median(observation_times),
        observation_encode_p95_ms=float(_p95(observation_times)),
        action_catalog_median_ms=median(catalog_times),
        action_catalog_p95_ms=float(_p95(catalog_times)),
        model_input_median_ms=median(model_input_times),
        model_input_p95_ms=float(_p95(model_input_times)),
        model_input_max_ms=max(model_input_times),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-actions-per-episode", type=int, default=10_000)
    args = parser.parse_args()
    result = run_rl_benchmark(
        episodes=args.episodes,
        start_seed=args.start_seed,
        max_actions_per_episode=args.max_actions_per_episode,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
