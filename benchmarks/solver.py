"""Opening-solver timing and optional peak-memory benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from statistics import median
from time import perf_counter

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.engine.round import RoundEngine
from okey101.solver.meld_generator import generate_melds
from okey101.solver.opening_solver import find_legal_openings
from okey101.solver.pair_solver import find_pair_openings, generate_pairs


@dataclass(frozen=True, slots=True)
class SolverBenchmarkResult:
    seeds: int
    meld_generation_median_ms: float
    meld_generation_p95_ms: float
    opening_solver_median_ms: float
    opening_solver_p95_ms: float
    pair_generation_median_ms: float
    pair_generation_p95_ms: float
    pair_opening_median_ms: float
    pair_opening_p95_ms: float
    median_meld_candidates: float
    median_opening_candidates: float
    median_pair_candidates: float
    median_pair_opening_candidates: float
    peak_memory_bytes: int | None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[ceil(len(ordered) * 0.95) - 1]


def run_solver_benchmark(
    *,
    seeds: int,
    start_seed: int = 0,
    measure_memory: bool = False,
) -> SolverBenchmarkResult:
    if seeds < 1:
        raise ValueError("seeds must be positive")

    timings: dict[str, list[float]] = {
        "meld": [],
        "opening": [],
        "pair": [],
        "pair_opening": [],
    }
    counts: dict[str, list[int]] = {name: [] for name in timings}
    engine = RoundEngine()
    for seed in range(start_seed, start_seed + seeds):
        state = engine.reset(seed)
        hand = state.current_player_state.hand

        started = perf_counter()
        melds = generate_melds(hand, state.okey_value)
        timings["meld"].append((perf_counter() - started) * 1_000)
        counts["meld"].append(len(melds))

        started = perf_counter()
        openings = find_legal_openings(
            hand,
            state.okey_value,
            threshold=state.progressive_series_threshold,
            preserve_final_discard=True,
        )
        timings["opening"].append((perf_counter() - started) * 1_000)
        counts["opening"].append(len(openings))

        started = perf_counter()
        pairs = generate_pairs(hand, state.okey_value)
        timings["pair"].append((perf_counter() - started) * 1_000)
        counts["pair"].append(len(pairs))

        started = perf_counter()
        pair_openings = find_pair_openings(
            hand,
            state.okey_value,
            threshold=state.progressive_pair_threshold,
            preserve_final_discard=True,
        )
        timings["pair_opening"].append((perf_counter() - started) * 1_000)
        counts["pair_opening"].append(len(pair_openings))

    peak_memory: int | None = None
    if measure_memory:
        tracemalloc.start()
        try:
            for seed in range(start_seed, start_seed + seeds):
                state = engine.reset(seed)
                hand = state.current_player_state.hand
                generate_melds(hand, state.okey_value)
                find_legal_openings(
                    hand,
                    state.okey_value,
                    threshold=state.progressive_series_threshold,
                    preserve_final_discard=True,
                )
                generate_pairs(hand, state.okey_value)
                find_pair_openings(
                    hand,
                    state.okey_value,
                    threshold=state.progressive_pair_threshold,
                    preserve_final_discard=True,
                )
            peak_memory = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    return SolverBenchmarkResult(
        seeds=seeds,
        meld_generation_median_ms=median(timings["meld"]),
        meld_generation_p95_ms=_p95(timings["meld"]),
        opening_solver_median_ms=median(timings["opening"]),
        opening_solver_p95_ms=_p95(timings["opening"]),
        pair_generation_median_ms=median(timings["pair"]),
        pair_generation_p95_ms=_p95(timings["pair"]),
        pair_opening_median_ms=median(timings["pair_opening"]),
        pair_opening_p95_ms=_p95(timings["pair_opening"]),
        median_meld_candidates=median(counts["meld"]),
        median_opening_candidates=median(counts["opening"]),
        median_pair_candidates=median(counts["pair"]),
        median_pair_opening_candidates=median(counts["pair_opening"]),
        peak_memory_bytes=peak_memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--measure-memory", action="store_true")
    args = parser.parse_args()

    result = run_solver_benchmark(
        seeds=args.seeds,
        start_seed=args.start_seed,
        measure_memory=args.measure_memory,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
