"""Render an okey101 replay JSON to a poster and/or MP4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.replay import load_replay, verify_replay
from okey101.visualization import render_contact_sheet, render_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poster", type=Path)
    parser.add_argument("--poster-only", action="store_true")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    document = load_replay(args.replay)
    if not args.skip_verify:
        verify_replay(document)
    stem = args.replay.with_suffix("")
    poster = args.poster or stem.with_name(f"{stem.name}-poster.png")
    render_contact_sheet(document, poster)
    video = None
    if not args.poster_only:
        video = args.output or stem.with_suffix(".mp4")
        render_mp4(
            document,
            video,
            fps=args.fps,
            size=(args.width, args.height),
        )
    payload = {
        "replay": str(args.replay),
        "poster": str(poster),
        "video": None if video is None else str(video),
        "frames": len(document["frames"]),
        "fps": args.fps,
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
