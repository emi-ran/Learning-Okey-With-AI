from __future__ import annotations

from benchmarks.rl import run_rl_benchmark
from benchmarks.solver import run_solver_benchmark


def test_solver_benchmark_smoke_reports_all_metrics() -> None:
    result = run_solver_benchmark(seeds=1, measure_memory=True)

    assert result.seeds == 1
    assert result.meld_generation_median_ms >= 0
    assert result.opening_solver_median_ms >= 0
    assert result.pair_generation_median_ms >= 0
    assert result.pair_opening_median_ms >= 0
    assert result.median_meld_candidates >= 0
    assert result.peak_memory_bytes is not None
    assert result.peak_memory_bytes > 0


def test_rl_benchmark_smoke_reports_candidate_and_encoder_metrics() -> None:
    result = run_rl_benchmark(episodes=1)

    assert result.episodes == 1
    assert result.actions > 0
    assert result.candidate_count_median >= 1
    assert result.candidate_count_p95 >= 1
    assert result.candidate_count_max >= result.candidate_count_p95
    assert result.observation_encode_median_ms >= 0
    assert result.action_catalog_median_ms >= 0
    assert result.model_input_median_ms >= 0
    assert result.model_input_max_ms >= result.model_input_p95_ms
