from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

from okey101.replay import record_random_episode
from okey101.visualization import render_contact_sheet, render_frame_to_path
from okey101.visualization.replay_video import _footer_narration, _group_hand


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


def test_terminal_footer_reports_round_end_reason(replay) -> None:
    assert _footer_narration(replay["frames"][1]) == replay["frames"][1]["narration"]
    assert _footer_narration(replay["frames"][-1]) == (
        "Stok bitti · el sona erdi."
    )


def test_display_hand_groups_preserve_every_physical_tile() -> None:
    tiles = [
        {
            "id": index,
            "display": {"color": color, "number": number},
            "is_real_okey": real_okey,
            "is_fake_okey": False,
        }
        for index, (color, number, real_okey) in enumerate(
            [
                ("red", 4, False),
                ("red", 5, False),
                ("red", 6, False),
                ("yellow", 9, False),
                ("blue", 9, False),
                ("black", 9, False),
                ("yellow", 2, False),
                ("yellow", 2, False),
                ("black", 3, True),
            ]
        )
    ]

    groups = _group_hand(tiles)
    grouped_ids = [tile["id"] for _kind, group in groups for tile in group]

    assert sorted(grouped_ids) == list(range(len(tiles)))
    assert [kind for kind, _group in groups] == ["run", "set", "pair", "okey"]
    assert groups == _group_hand(tiles)


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
