from __future__ import annotations

import json
import subprocess
import sys

import pytest

import benchmarks.random_baseline as random_baseline
from benchmarks.random_baseline import run_random_baseline


def _deterministic_fields(result) -> tuple[object, ...]:
    return (
        result.episodes,
        result.start_seed,
        result.end_seed_exclusive,
        result.actions,
        result.terminal_reason_counts,
        result.seat_score_means,
        result.overall_finish_rate,
        result.seat_finish_rates,
        result.overall_unopened_rate,
        result.seat_unopened_rates,
        result.penalties_per_game,
        result.unavailable_metrics,
    )


def test_random_baseline_smoke_reports_policy_safe_evaluation_metrics() -> None:
    result = run_random_baseline(episodes=3, start_seed=30)

    assert result.episodes == 3
    assert result.start_seed == 30
    assert result.end_seed_exclusive == 33
    assert result.actions > 0
    assert result.elapsed_seconds > 0
    assert result.episodes_per_second > 0
    assert result.actions_per_second > 0
    assert sum(result.terminal_reason_counts.values()) == result.episodes
    assert len(result.seat_score_means) == 4
    assert len(result.seat_finish_rates) == 4
    assert len(result.seat_unopened_rates) == 4
    assert 0 <= result.overall_finish_rate <= 1
    assert all(0 <= rate <= 1 for rate in result.seat_finish_rates)
    assert 0 <= result.overall_unopened_rate <= 1
    assert all(0 <= rate <= 1 for rate in result.seat_unopened_rates)
    assert result.penalties_per_game is None
    assert "terminal stock-exhaustion discard" in result.unavailable_metrics[0]


def test_random_baseline_is_reproducible_except_for_wall_clock_metrics() -> None:
    first = run_random_baseline(episodes=4, start_seed=-4)
    second = run_random_baseline(episodes=4, start_seed=-4)

    assert _deterministic_fields(first) == _deterministic_fields(second)


def test_benchmark_module_has_no_raw_engine_state_dependency() -> None:
    assert "RoundEngine" not in random_baseline.__dict__
    assert "GameState" not in random_baseline.__dict__
    assert "EngineEvent" not in random_baseline.__dict__


def test_random_baseline_json_cli_reports_seed_range() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.random_baseline",
            "--episodes",
            "1",
            "--start-seed",
            "17",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["episodes"] == 1
    assert payload["start_seed"] == 17
    assert payload["end_seed_exclusive"] == 18
    assert payload["actions"] > 0
    assert sum(payload["terminal_reason_counts"].values()) == 1
    assert payload["penalties_per_game"] is None


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"episodes": 0}, ValueError, "positive"),
        ({"episodes": True}, TypeError, "integer"),
        ({"episodes": 1, "start_seed": True}, TypeError, "start_seed"),
        (
            {"episodes": 1, "max_actions_per_episode": 0},
            ValueError,
            "positive",
        ),
    ),
)
def test_random_baseline_validates_inputs(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        run_random_baseline(**kwargs)
