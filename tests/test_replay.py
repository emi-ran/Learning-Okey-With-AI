from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.replay import record_random_replays
from okey101.engine.actions import Action
from okey101.replay import (
    ActionSelection,
    CandidateProbability,
    CheckpointMetadata,
    ReplayValidationError,
    ReplayVerificationError,
    SelectionContext,
    build_comparison_manifest,
    load_comparison_manifest,
    load_replay,
    record_episode,
    record_random_episode,
    validate_replay_document,
    verify_replay,
    write_replay,
)
from okey101.replay.schema import content_digest
from okey101.rl.observation import PlayerObservation


def test_random_replay_is_deterministic_and_renderer_ready() -> None:
    first = record_random_episode(seed=12, spectator_mode=True)
    second = record_random_episode(seed=12, spectator_mode=True)

    assert first == second
    assert first["schema_version"] == "okey101.replay.v1"
    assert first["visibility"] == {
        "mode": "spectator",
        "reveal_all_hands": True,
    }
    assert first["frames"][0]["action"] is None
    assert first["frames"][-1]["terminal"]["is_terminal"]
    assert first["summary"]["frame_count"] == first["summary"]["action_count"] + 1
    assert all(
        player["hand_count"] == len(player["hand"])
        for frame in first["frames"]
        for player in frame["view"]["players"]
    )
    verify_replay(first)


def test_recording_requires_explicit_spectator_mode() -> None:
    with pytest.raises(ValueError, match="explicitly enabled"):
        record_random_episode(seed=0)


def test_generic_selector_can_attach_policy_probability_and_value() -> None:
    def first_legal(
        observation: PlayerObservation,
        legal_actions: list[Action] | tuple[Action, ...],
        context: SelectionContext,
    ) -> ActionSelection:
        del observation, context
        return ActionSelection(
            action=legal_actions[0],
            selected_probability=1.0,
            value=0.25,
            candidates=(CandidateProbability(legal_actions[0], 1.0),),
        )

    replay = record_episode(
        seed=8,
        selector=first_legal,
        checkpoint=CheckpointMetadata("first", "İlk legal", 25),
        policy_name="FixturePolicy",
        spectator_mode=True,
    )

    policy_step = replay["frames"][1]["policy_step"]
    assert policy_step["selected_probability"] == 1.0
    assert policy_step["value"] == 0.25
    assert policy_step["candidates"][0]["probability"] == 1.0
    verify_replay(replay)


def test_replay_load_validates_json_and_integrity(tmp_path: Path) -> None:
    replay = record_random_episode(seed=4, spectator_mode=True)
    path = write_replay(replay, tmp_path / "replay.json")
    assert load_replay(path) == replay

    tampered = copy.deepcopy(replay)
    tampered["visibility"]["mode"] = "player"
    unsigned = dict(tampered)
    unsigned.pop("integrity")
    tampered["integrity"]["document_digest"] = content_digest(unsigned)
    with pytest.raises(ReplayValidationError, match="spectator"):
        validate_replay_document(tampered)

    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ReplayValidationError, match="JSON"):
        load_replay(tmp_path / "broken.json")


def test_deterministic_verification_detects_engine_frame_tampering() -> None:
    replay = record_random_episode(seed=2, spectator_mode=True)
    tampered = copy.deepcopy(replay)
    frame = tampered["frames"][1]
    frame["view"]["stock_count"] += 1
    frame["state_digest"] = content_digest(
        {
            key: value
            for key, value in frame.items()
            if key not in {"narration", "policy_step", "state_digest"}
        }
    )
    unsigned = dict(tampered)
    unsigned.pop("integrity")
    tampered["integrity"]["document_digest"] = content_digest(unsigned)

    with pytest.raises(ReplayVerificationError, match=r"frame 1\.view"):
        verify_replay(tampered)


def test_comparison_manifest_requires_same_fixed_seeds() -> None:
    checkpoint = CheckpointMetadata("random", "Random", 0)
    first = record_random_episode(
        seed=5,
        checkpoint=checkpoint,
        spectator_mode=True,
    )
    second = record_random_episode(
        seed=6,
        checkpoint=checkpoint,
        spectator_mode=True,
    )
    manifest = build_comparison_manifest(
        title="Gelişim",
        fixed_seeds=(5, 6),
        checkpoints=(
            (
                first["checkpoint"],
                (("seed-5.json", first), ("seed-6.json", second)),
            ),
        ),
    )
    assert manifest["fixed_seeds"] == [5, 6]
    assert [item["seed"] for item in manifest["checkpoints"][0]["replays"]] == [
        5,
        6,
    ]

    with pytest.raises(ValueError, match="fixed_seeds"):
        build_comparison_manifest(
            title="Hatalı",
            fixed_seeds=(6, 5),
            checkpoints=(
                (
                    first["checkpoint"],
                    (("seed-5.json", first), ("seed-6.json", second)),
                ),
            ),
        )


def test_replay_benchmark_writes_verified_files_and_manifest(tmp_path: Path) -> None:
    result = record_random_replays(
        seeds=(21,),
        output_dir=tmp_path,
    )

    assert result.replay_count == 1
    assert result.actions > 0
    assert result.actions_per_second > 0
    assert len(result.output_paths) == 1
    assert Path(result.output_paths[0]).is_file()
    assert result.manifest_path is not None
    assert Path(result.manifest_path).is_file()
    assert load_comparison_manifest(result.manifest_path)["fixed_seeds"] == [21]
    verify_replay(load_replay(result.output_paths[0]))


def test_replay_cli_outputs_machine_readable_summary(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.replay",
            "--seeds",
            "31",
            "--output-dir",
            str(tmp_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["replay_count"] == 1
    assert payload["seeds"] == [31]
    assert Path(payload["output_paths"][0]).is_file()
