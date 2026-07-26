from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

from okey101.replay import record_random_episode
from okey101.visualization import render_contact_sheet, render_frame_to_path


@pytest.fixture(scope="module")
def replay():
    return record_random_episode(seed=41, spectator_mode=True)


def test_render_frame_writes_expected_dimensions(replay, tmp_path: Path) -> None:
    output = render_frame_to_path(
        replay,
        1,
        tmp_path / "frame.png",
        size=(960, 540),
    )
    with Image.open(output) as image:
        assert image.size == (960, 540)
        assert image.mode == "RGB"


def test_contact_sheet_covers_start_middle_and_terminal(
    replay,
    tmp_path: Path,
) -> None:
    output = render_contact_sheet(replay, tmp_path / "poster.png")
    with Image.open(output) as image:
        assert image.size == (1440, 270)


def test_poster_only_cli_does_not_require_ffmpeg(
    replay,
    tmp_path: Path,
) -> None:
    from okey101.replay import write_replay

    replay_path = write_replay(replay, tmp_path / "replay.json")
    poster_path = tmp_path / "poster.png"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.render_replay",
            str(replay_path),
            "--poster",
            str(poster_path),
            "--poster-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert poster_path.is_file()
