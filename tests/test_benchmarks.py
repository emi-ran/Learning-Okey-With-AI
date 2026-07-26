from __future__ import annotations

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
