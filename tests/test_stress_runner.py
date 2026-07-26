from __future__ import annotations

import json

import benchmarks.replay_failure as replay_failure_module
import benchmarks.stress as stress_module
from benchmarks.replay_failure import reproduce_failure
from benchmarks.stress import _chunks, run_stress
from okey101.engine.invariants import InvariantError, InvariantViolation
from okey101.engine.round import RoundEngine
from okey101.engine.transition import IllegalAction


def test_seed_chunks_are_contiguous_and_cover_every_round() -> None:
    assert _chunks(10, 3, 20) == [(20, 4), (24, 3), (27, 3)]


def test_single_worker_stress_smoke_passes() -> None:
    result = run_stress(rounds=1, workers=1, validate_every_step=True)

    assert result.requested_rounds == 1
    assert result.completed_rounds == 1
    assert result.actions > 0
    assert result.failure_count == 0
    assert sum(result.terminal_reasons.values()) == 1


def test_failure_artifact_replays_the_failed_action(tmp_path) -> None:
    engine = RoundEngine()
    engine.reset(seed=4, starting_player=0)
    artifact = tmp_path / "failure.json"
    artifact.write_text(
        json.dumps(
            {
                "state": engine.serialize_state(),
                "failed_action": {"type": "draw_from_stock"},
            }
        ),
        encoding="utf-8",
    )

    error = reproduce_failure(artifact)

    assert isinstance(error, IllegalAction)
    assert "not allowed in this phase" in str(error)


def test_post_step_invariant_failure_keeps_action_and_pre_action_state(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0

    def fail_after_initial_validation(_state) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InvariantError(
                (InvariantViolation("SYNTHETIC", "after step"),)
            )

    monkeypatch.setattr(
        stress_module,
        "validate_invariants",
        fail_after_initial_validation,
    )
    result = run_stress(
        rounds=1,
        workers=1,
        failure_directory=tmp_path,
    )
    artifact = tmp_path / "seed-0.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert result.failure_count == 1
    assert payload["failed_action"] is not None
    assert payload["actions"][-1] == payload["failed_action"]

    def reproduce_invariant(_state) -> None:
        raise InvariantError(
            (InvariantViolation("SYNTHETIC", "after step"),)
        )

    monkeypatch.setattr(
        replay_failure_module,
        "validate_invariants",
        reproduce_invariant,
    )
    error = reproduce_failure(artifact)

    assert isinstance(error, InvariantError)
    assert "SYNTHETIC" in str(error)
