"""Pillow frame rendering and FFmpeg encoding for replay schema v1."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from okey101.replay.schema import validate_replay_document

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - exercised only without the optional extra
    Image = ImageDraw = ImageFont = None


class MissingVideoDependency(RuntimeError):
    """Raised when Pillow or FFmpeg is unavailable."""


_COLORS = {
    "red": "#e54b4b",
    "yellow": "#e0b735",
    "blue": "#3e82d7",
    "black": "#222831",
}


def _require_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise MissingVideoDependency(
            "Pillow is required; install the project with the 'video' extra"
        )


def _font(size: int, *, bold: bool = False):
    _require_pillow()
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _rounded(draw, box, *, radius: int, fill: str, outline: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def _tile(
    draw,
    tile: Mapping[str, object],
    x: int,
    y: int,
    *,
    width: int = 27,
    height: int = 39,
) -> None:
    _rounded(
        draw,
        (x, y, x + width, y + height),
        radius=max(3, width // 6),
        fill="#f4ead2",
        outline="#c9b995",
    )
    display = tile["display"]
    assert isinstance(display, Mapping)
    color = _COLORS.get(str(display["color"]), "#222831")
    number = str(display["number"])
    number_font = _font(max(10, int(height * 0.46)), bold=True)
    bbox = draw.textbbox((0, 0), number, font=number_font)
    draw.text(
        (x + (width - (bbox[2] - bbox[0])) / 2, y + 2),
        number,
        font=number_font,
        fill=color,
    )
    if tile.get("is_real_okey"):
        draw.ellipse(
            (x + width * 0.36, y + height * 0.72, x + width * 0.64, y + height * 0.9),
            fill="#d22f2f",
        )
    elif tile.get("is_fake_okey"):
        draw.text(
            (x + width * 0.36, y + height * 0.63),
            "★",
            font=_font(max(8, int(height * 0.22))),
            fill="#242424",
        )


def _hand_row(
    draw,
    tiles: Sequence[object],
    box: tuple[int, int, int, int],
    *,
    columns: int,
) -> None:
    x0, y0, x1, y1 = box
    rows = max(1, (len(tiles) + columns - 1) // columns)
    gap = 3
    tile_width = max(18, min(29, (x1 - x0 - (columns - 1) * gap) // columns))
    tile_height = max(27, min(41, (y1 - y0 - (rows - 1) * gap) // rows))
    for index, item in enumerate(tiles):
        if not isinstance(item, Mapping):
            continue
        column = index % columns
        row = index // columns
        _tile(
            draw,
            item,
            x0 + column * (tile_width + gap),
            y0 + row * (tile_height + gap),
            width=tile_width,
            height=tile_height,
        )


def _player_panel(
    draw,
    player: Mapping[str, object],
    box: tuple[int, int, int, int],
    *,
    active: bool,
    columns: int,
) -> None:
    x0, y0, x1, y1 = box
    _rounded(
        draw,
        box,
        radius=12,
        fill="#163d34" if active else "#17352f",
        outline="#f6ca56" if active else "#32685d",
    )
    mode = str(player["opened_mode"])
    mode_text = {"none": "Açmadı", "series": "Seri", "pairs": "Çift"}.get(mode, mode)
    draw.text(
        (x0 + 10, y0 + 7),
        f"{player['label']} · {mode_text}",
        font=_font(15, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (x1 - 145, y0 + 8),
        f"Ceza {player['immediate_penalty']}",
        font=_font(13),
        fill="#ffd77a",
    )
    hand = player["hand"]
    assert isinstance(hand, Sequence)
    _hand_row(draw, hand, (x0 + 10, y0 + 30, x1 - 10, y1 - 8), columns=columns)


def _table(draw, table: Mapping[str, object], box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0), "MASA", font=_font(16, bold=True), fill="#9bd6c4")
    cursor_x = x0
    cursor_y = y0 + 25
    for meld_value in table["melds"]:
        if not isinstance(meld_value, Mapping):
            continue
        tiles = meld_value["tiles"]
        if not isinstance(tiles, Sequence):
            continue
        width = len(tiles) * 29 + 10
        if cursor_x + width > x1:
            cursor_x = x0
            cursor_y += 51
        _rounded(
            draw,
            (cursor_x - 3, cursor_y - 3, cursor_x + width - 5, cursor_y + 43),
            radius=6,
            fill="#0d2f28",
            outline="#397d6e",
        )
        for index, meld_tile in enumerate(tiles):
            if isinstance(meld_tile, Mapping) and isinstance(meld_tile.get("tile"), Mapping):
                _tile(draw, meld_tile["tile"], cursor_x + index * 29, cursor_y)
        cursor_x += width

    pairs = table["pairs"]
    if isinstance(pairs, Sequence) and pairs:
        cursor_x = x0
        cursor_y = min(y1 - 45, cursor_y + 56)
        draw.text(
            (cursor_x, cursor_y - 19),
            "ÇİFTLER",
            font=_font(12, bold=True),
            fill="#9bd6c4",
        )
        for pair_value in pairs:
            if not isinstance(pair_value, Mapping):
                continue
            pair_tiles = pair_value["tiles"]
            if not isinstance(pair_tiles, Sequence):
                continue
            if cursor_x + 67 > x1:
                cursor_x = x0
                cursor_y += 44
            for index, pair_tile in enumerate(pair_tiles):
                if isinstance(pair_tile, Mapping) and isinstance(pair_tile.get("tile"), Mapping):
                    _tile(
                        draw,
                        pair_tile["tile"],
                        cursor_x + index * 29,
                        cursor_y,
                    )
            cursor_x += 67


def render_frame(
    document: Mapping[str, object],
    frame_index: int,
    *,
    size: tuple[int, int] = (1280, 720),
):
    """Render one replay frame to a Pillow image."""

    _require_pillow()
    replay = validate_replay_document(document)
    frames = replay["frames"]
    if not 0 <= frame_index < len(frames):
        raise IndexError("frame_index is outside the replay")
    frame = frames[frame_index]
    view = frame["view"]
    width, height = size
    if width < 960 or height < 540:
        raise ValueError("video size must be at least 960x540")

    image = Image.new("RGB", size, "#0a2c24")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 76), fill="#081f1a")
    checkpoint = replay["checkpoint"]
    draw.text(
        (28, 14),
        "101 OKEY · AI GELİŞİMİ",
        font=_font(25, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (28, 45),
        f"{checkpoint['label']} · Eğitim adımı {checkpoint['training_step']:,}",
        font=_font(16),
        fill="#8fd8c2",
    )
    draw.text(
        (width - 290, 18),
        f"Kare {frame_index + 1}/{len(frames)}",
        font=_font(16, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (width - 290, 43),
        f"Tur {view['turn_number']} · {view['phase']}",
        font=_font(14),
        fill="#9db9b1",
    )

    players = view["players"]
    current = int(view["current_player"])
    _player_panel(
        draw,
        players[2],
        (270, 88, width - 270, 178),
        active=current == 2,
        columns=22,
    )
    _player_panel(
        draw,
        players[0],
        (270, height - 178, width - 270, height - 88),
        active=current == 0,
        columns=22,
    )
    _player_panel(
        draw,
        players[3],
        (18, 116, 240, height - 116),
        active=current == 3,
        columns=5,
    )
    _player_panel(
        draw,
        players[1],
        (width - 240, 116, width - 18, height - 116),
        active=current == 1,
        columns=5,
    )

    _table(draw, view["table"], (270, 206, width - 270, height - 212))
    indicator = view["indicator"]
    draw.text((257, 187), "Gösterge", font=_font(12), fill="#9db9b1")
    _tile(draw, indicator, 315, 181, width=29, height=41)
    draw.text(
        (360, 187),
        f"Okey: {view['okey_value']['color']} {view['okey_value']['number']}",
        font=_font(13, bold=True),
        fill="#ffd77a",
    )
    draw.text(
        (width - 440, 187),
        f"Stok {view['stock_count']} · Atılan {len(view['discard_pile'])}",
        font=_font(13, bold=True),
        fill="#ffffff",
    )
    if isinstance(view.get("discard_top"), Mapping):
        _tile(draw, view["discard_top"], width - 315, 181, width=29, height=41)

    footer_y = height - 76
    draw.rectangle((0, footer_y, width, height), fill="#081f1a")
    draw.text(
        (28, footer_y + 12),
        str(frame["narration"]),
        font=_font(20, bold=True),
        fill="#ffffff",
    )
    final_totals = frame["scores"]["final_totals"]
    score_text = (
        " · ".join(f"O{i + 1}: {score}" for i, score in enumerate(final_totals))
        if isinstance(final_totals, Sequence)
        else " · ".join(
            f"O{i + 1}: +{score}"
            for i, score in enumerate(frame["scores"]["immediate_penalties"])
        )
    )
    draw.text(
        (28, footer_y + 43),
        score_text,
        font=_font(14),
        fill="#ffd77a",
    )
    policy = frame.get("policy_step")
    if isinstance(policy, Mapping):
        probability = policy.get("selected_probability")
        value = policy.get("value")
        metrics = []
        if probability is not None:
            metrics.append(f"Hamle olasılığı %{float(probability) * 100:.1f}")
        if value is not None:
            metrics.append(f"Durum değeri {float(value):+.3f}")
        if metrics:
            draw.text(
                (width - 390, footer_y + 43),
                " · ".join(metrics),
                font=_font(14),
                fill="#8fd8c2",
            )
    return image


def render_frame_to_path(
    document: Mapping[str, object],
    frame_index: int,
    path: str | Path,
    *,
    size: tuple[int, int] = (1280, 720),
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_frame(document, frame_index, size=size).save(output, format="PNG")
    return output


def render_contact_sheet(
    document: Mapping[str, object],
    path: str | Path,
    *,
    frame_indices: Sequence[int] | None = None,
    frame_size: tuple[int, int] = (960, 540),
) -> Path:
    replay = validate_replay_document(document)
    frames = replay["frames"]
    indices = tuple(frame_indices or (0, len(frames) // 2, len(frames) - 1))
    images = [render_frame(replay, index, size=frame_size) for index in indices]
    thumb_width = 480
    thumb_height = round(frame_size[1] * thumb_width / frame_size[0])
    sheet = Image.new("RGB", (thumb_width * len(images), thumb_height), "#081f1a")
    for index, image in enumerate(images):
        image.thumbnail((thumb_width, thumb_height))
        sheet.paste(image, (index * thumb_width, 0))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    return output


def render_mp4(
    document: Mapping[str, object],
    path: str | Path,
    *,
    fps: float = 2.0,
    size: tuple[int, int] = (1280, 720),
    ffmpeg_path: str | None = None,
) -> Path:
    """Render every replay frame and encode an H.264/YUV420p MP4."""

    replay = validate_replay_document(document)
    if fps <= 0:
        raise ValueError("fps must be positive")
    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise MissingVideoDependency("FFmpeg was not found on PATH")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="okey101-video-") as temporary:
        frame_dir = Path(temporary)
        for index in range(len(replay["frames"])):
            render_frame(replay, index, size=size).save(
                frame_dir / f"frame-{index:06d}.png",
                format="PNG",
            )
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(frame_dir / "frame-%06d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {completed.stderr.strip()}")
    return output
