from __future__ import annotations

import argparse
import json

import pytest

from benchmarks import train_progression
from okey101.engine.config import GameConfig
from okey101.replay import load_replay, verify_replay
from okey101.training import TrainingConfig


def _fast_config() -> GameConfig:
    return GameConfig(initial_hand_size=3, opening_min_score=3)


def test_checkpoint_parser_requires_zero_and_target() -> None:
    assert train_progression._parse_checkpoints(
        "0,2,5",
        5,
    ) == (0, 2, 5)

    with pytest.raises(
        argparse.ArgumentTypeError,
        match="start with 0",
    ):
        train_progression._parse_checkpoints("1,5", 5)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="final checkpoint",
    ):
        train_progression._parse_checkpoints("0,2", 5)


def test_progression_publishes_checkpoints_replays_and_history(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        train_progression,
        "_REPOSITORY_ROOT",
        tmp_path,
    )
    output_dir = tmp_path / "artifacts" / "training" / "test-live"

    result = train_progression.run_progression(
        episodes=2,
        checkpoints=(0, 1, 2),
        seed=7,
        replay_seed=11,
        evaluation_episodes=1,
        output_dir=output_dir,
        render_videos=False,
        game_config=_fast_config(),
        training_config=TrainingConfig(
            hidden_size=4,
            max_actions_per_episode=500,
        ),
        quiet=True,
    )

    status = json.loads(
        (output_dir / "status.json").read_text(encoding="utf-8")
    )
    assert result["state"]["phase"] == "complete"
    assert status["state"]["current_episode"] == 2
    assert len(status["history"]) == 2
    assert [item["episode"] for item in status["checkpoints"]] == [0, 1, 2]
    assert all(
        item["status"] == "ready"
        for item in status["checkpoints"]
    )

    for episode in (0, 1, 2):
        checkpoint_id = f"checkpoint-{episode:04d}"
        assert (
            output_dir / "checkpoints" / f"{checkpoint_id}.npz"
        ).is_file()
        replay_path = (
            output_dir
            / "replays"
            / f"{checkpoint_id}-seed-11.json"
        )
        replay = load_replay(replay_path)
        verify_replay(replay)
        assert replay["checkpoint"]["training_step"] == episode


def test_progression_refuses_nonempty_output(tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        train_progression.run_progression(
            episodes=1,
            checkpoints=(0, 1),
            seed=0,
            replay_seed=0,
            evaluation_episodes=1,
            output_dir=output_dir,
            render_videos=False,
            quiet=True,
        )

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_atomic_status_write_retries_windows_read_lock(
    tmp_path,
    monkeypatch,
) -> None:
    destination = tmp_path / "status.json"
    real_replace = train_progression.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated Windows sharing violation")
        return real_replace(source, target)

    monkeypatch.setattr(train_progression.os, "replace", flaky_replace)

    train_progression._atomic_write_json(
        destination,
        {"state": {"phase": "training"}},
    )

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "state": {"phase": "training"}
    }
