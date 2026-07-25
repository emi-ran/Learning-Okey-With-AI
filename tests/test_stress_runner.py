from __future__ import annotations

from benchmarks.stress import _chunks, run_stress


def test_seed_chunks_are_contiguous_and_cover_every_round() -> None:
    assert _chunks(10, 3, 20) == [(20, 4), (24, 3), (27, 3)]


def test_single_worker_stress_smoke_passes() -> None:
    result = run_stress(rounds=1, workers=1, validate_every_step=True)

    assert result.requested_rounds == 1
    assert result.completed_rounds == 1
    assert result.actions > 0
    assert result.failure_count == 0
    assert sum(result.terminal_reasons.values()) == 1
