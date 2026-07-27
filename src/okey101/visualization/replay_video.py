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

_COLOR_LABELS = {
    "red": "Kırmızı",
    "yellow": "Sarı",
    "blue": "Mavi",
    "black": "Siyah",
}

_PHASE_LABELS = {
    "draw_decision": "ÇEKME KARARI",
    "table_actions": "MASA HAMLELERİ",
    "discard": "TAŞ ATMA",
    "terminal": "EL SONU",
}

_COLOR_ORDER = {"red": 0, "yellow": 1, "blue": 2, "black": 3}


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
    marker: str = "auto",
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
    if marker == "flower":
        center_x = x + width * 0.5
        center_y = y + height * 0.79
        petal = max(1.7, width * 0.075)
        offset = petal * 1.15
        for dx, dy in ((-offset, 0), (offset, 0), (0, -offset), (0, offset)):
            draw.ellipse(
                (
                    center_x + dx - petal,
                    center_y + dy - petal,
                    center_x + dx + petal,
                    center_y + dy + petal,
                ),
                fill="#92968e",
            )
        draw.ellipse(
            (
                center_x - petal,
                center_y - petal,
                center_x + petal,
                center_y + petal,
            ),
            fill="#e8ddc5",
        )
    elif marker == "auto" and tile.get("is_real_okey"):
        draw.ellipse(
            (x + width * 0.36, y + height * 0.72, x + width * 0.64, y + height * 0.9),
            fill="#d22f2f",
        )
    elif marker == "auto" and tile.get("is_fake_okey"):
        draw.text(
            (x + width * 0.36, y + height * 0.63),
            "★",
            font=_font(max(8, int(height * 0.22))),
            fill="#242424",
        )


def _tile_display(tile: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("display", "physical", "value"):
        display = tile.get(key)
        if isinstance(display, Mapping):
            return display
    return tile


def _tile_sort_key(tile: Mapping[str, object]) -> tuple[int, int, int]:
    display = _tile_display(tile)
    return (
        _COLOR_ORDER.get(str(display.get("color")), 9),
        int(display.get("number", 99)),
        int(tile.get("id", 0)),
    )


def _group_hand(
    tiles: Sequence[object],
    opened_mode: str = "none",
) -> list[tuple[str, list[Mapping[str, object]]]]:
    """Build a deterministic display-only rack layout without mutating game state."""

    valid = [tile for tile in tiles if isinstance(tile, Mapping)]
    remaining = [
        tile
        for tile in valid
        if not tile.get("is_real_okey") and not tile.get("is_fake_okey")
    ]
    special = [
        tile
        for tile in valid
        if tile.get("is_real_okey") or tile.get("is_fake_okey")
    ]
    groups: list[tuple[str, list[Mapping[str, object]]]] = []

    def take_pairs() -> None:
        buckets: dict[tuple[object, object], list[Mapping[str, object]]] = {}
        for tile in remaining:
            display = _tile_display(tile)
            key = (display.get("color"), display.get("number"))
            buckets.setdefault(key, []).append(tile)
        for bucket in buckets.values():
            while len(bucket) >= 2:
                pair = bucket[:2]
                del bucket[:2]
                for tile in pair:
                    remaining.remove(tile)
                groups.append(("pair", sorted(pair, key=_tile_sort_key)))

    if opened_mode == "pairs":
        take_pairs()

    for color in _COLOR_ORDER:
        while True:
            by_number: dict[int, list[Mapping[str, object]]] = {}
            for tile in sorted(remaining, key=_tile_sort_key):
                display = _tile_display(tile)
                if display.get("color") == color:
                    by_number.setdefault(int(display["number"]), []).append(tile)
            best: list[int] = []
            current: list[int] = []
            for number in sorted(by_number):
                current = (
                    [*current, number]
                    if not current or number == current[-1] + 1
                    else [number]
                )
                if len(current) > len(best):
                    best = current.copy()
            if len(best) < 3:
                break
            run = [by_number[number][0] for number in best]
            for tile in run:
                remaining.remove(tile)
            groups.append(("run", run))

    for number in range(1, 14):
        while True:
            set_tiles: list[Mapping[str, object]] = []
            for color in _COLOR_ORDER:
                tile = next(
                    (
                        item
                        for item in remaining
                        if _tile_display(item).get("number") == number
                        and _tile_display(item).get("color") == color
                    ),
                    None,
                )
                if tile is not None:
                    set_tiles.append(tile)
            if len(set_tiles) < 3:
                break
            for tile in set_tiles:
                remaining.remove(tile)
            groups.append(("set", sorted(set_tiles, key=_tile_sort_key)))

    if opened_mode != "pairs":
        take_pairs()

    for color in _COLOR_ORDER:
        loose = [
            tile
            for tile in remaining
            if _tile_display(tile).get("color") == color
        ]
        if loose:
            groups.append(("loose", sorted(loose, key=_tile_sort_key)))
            for tile in loose:
                remaining.remove(tile)
    if remaining:
        groups.append(("loose", sorted(remaining, key=_tile_sort_key)))
    if special:
        groups.append(("okey", sorted(special, key=_tile_sort_key)))
    return groups


def _hand_row(
    draw,
    tiles: Sequence[object],
    box: tuple[int, int, int, int],
    *,
    columns: int,
    opened_mode: str = "none",
) -> None:
    x0, y0, x1, y1 = box
    rows = max(1, (len(tiles) + columns - 1) // columns)
    gap = 2
    group_gap = 8
    tile_width = max(
        18,
        min(29, (x1 - x0 - (columns - 1) * gap - group_gap * 4) // columns),
    )
    tile_height = max(27, min(41, (y1 - y0 - (rows - 1) * gap) // rows))
    cursor_x = x0
    cursor_y = y0
    column = 0
    for kind, group in _group_hand(tiles, opened_mode):
        if column and column + len(group) > columns:
            column = 0
            cursor_x = x0
            cursor_y += tile_height + gap + 4
        elif column:
            cursor_x += group_gap
        group_start = cursor_x
        for item in group:
            if column >= columns:
                column = 0
                cursor_x = x0
                cursor_y += tile_height + gap + 4
                group_start = cursor_x
            _tile(
                draw,
                item,
                cursor_x,
                cursor_y,
                width=tile_width,
                height=tile_height,
            )
            cursor_x += tile_width + gap
            column += 1
        underline = {
            "run": "#62c59b",
            "set": "#eacb78",
            "pair": "#eacb78",
            "okey": "#d6a62e",
        }.get(kind)
        if underline and group:
            draw.line(
                (
                    group_start,
                    cursor_y + tile_height + 2,
                    cursor_x - gap,
                    cursor_y + tile_height + 2,
                ),
                fill=underline,
                width=2,
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
        str(player["label"]),
        font=_font(15, bold=True),
        fill="#ffffff",
    )
    score_text = f"Ceza {player['immediate_penalty']}"
    score_font = _font(13)
    score_box = draw.textbbox((0, 0), score_text, font=score_font)
    draw.text(
        (x1 - 10 - (score_box[2] - score_box[0]), y0 + 8),
        score_text,
        font=score_font,
        fill="#ffd77a",
    )
    draw.text(
        (x0 + 10, y0 + 25),
        mode_text,
        font=_font(11, bold=True),
        fill="#9bd6c4",
    )
    hand = player["hand"]
    assert isinstance(hand, Sequence)
    _hand_row(
        draw,
        hand,
        (x0 + 10, y0 + 40, x1 - 10, y1 - 8),
        columns=columns,
        opened_mode=mode,
    )


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


def _discard_panel(
    draw,
    history: Sequence[object],
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    _rounded(
        draw,
        box,
        radius=9,
        fill="#0d2f28",
        outline="#397d6e",
    )
    draw.text(
        (x0 + 8, y0 + 8),
        "ATILAN TAŞLAR",
        font=_font(11, bold=True),
        fill="#9bd6c4",
    )
    valid = [record for record in history if isinstance(record, Mapping)]
    row_height = max(12, (y1 - y0 - 31) // 4)
    tile_height = min(31, max(10, row_height - 4))
    tile_width = max(8, round(tile_height * 0.68))
    for player_id in range(4):
        row_y = y0 + 29 + player_id * row_height
        draw.text(
            (x0 + 8, row_y + 7),
            f"O{player_id + 1}",
            font=_font(10, bold=True),
            fill="#ffd77a",
        )
        records = [
            record
            for record in valid
            if int(record.get("player_id", -1)) == player_id
        ][-3:]
        if not records:
            draw.text(
                (x0 + 34, row_y + 8),
                "—",
                font=_font(13),
                fill="#597b71",
            )
            continue
        for index, record in enumerate(records):
            tile = record.get("tile")
            if not isinstance(tile, Mapping):
                continue
            tile_x = x0 + 32 + index * (tile_width + 3)
            _tile(
                draw,
                tile,
                tile_x,
                row_y,
                width=tile_width,
                height=tile_height,
            )
            if record.get("taken_by") is not None:
                draw.line(
                    (
                        tile_x + 2,
                        row_y + tile_height - 3,
                        tile_x + tile_width - 2,
                        row_y + 3,
                    ),
                    fill="#b96a5c",
                    width=2,
                )


def _discard_pocket(
    draw,
    history: Sequence[object],
    *,
    player_id: int,
    label: str,
    box: tuple[int, int, int, int],
) -> None:
    """Render one player's recent discards in a table-corner pocket."""

    x0, y0, x1, y1 = box
    _rounded(
        draw,
        box,
        radius=8,
        fill="#0b342a",
        outline="#6d9b81",
    )
    draw.text(
        (x0 + 8, y0 + 6),
        label,
        font=_font(10, bold=True),
        fill="#d9c47d",
    )
    records = [
        record
        for record in history
        if isinstance(record, Mapping)
        and int(record.get("player_id", -1)) == player_id
    ][-4:]
    if not records:
        draw.text(
            (x0 + 8, y0 + 28),
            "ATMA ALANI",
            font=_font(8, bold=True),
            fill="#56786d",
        )
        return
    tile_height = min(38, y1 - y0 - 12)
    tile_width = round(tile_height * 0.68)
    start_x = x1 - 8 - tile_width - (len(records) - 1) * (tile_width // 2)
    tile_y = y0 + (y1 - y0 - tile_height) // 2
    for index, record in enumerate(records):
        tile = record.get("tile")
        if not isinstance(tile, Mapping):
            continue
        tile_x = start_x + index * (tile_width // 2)
        _tile(
            draw,
            tile,
            tile_x,
            tile_y,
            width=tile_width,
            height=tile_height,
        )
        if record.get("taken_by") is not None:
            draw.line(
                (
                    tile_x + 2,
                    tile_y + tile_height - 3,
                    tile_x + tile_width - 2,
                    tile_y + 3,
                ),
                fill="#bd6558",
                width=2,
            )


def _status_tile(
    draw,
    *,
    label: str,
    tile: Mapping[str, object] | None,
    x: int,
    y: int,
    width: int = 88,
    marker: str = "auto",
) -> None:
    _rounded(
        draw,
        (x, y, x + width, y + 51),
        radius=7,
        fill="#102f28",
        outline="#315f55",
    )
    draw.text(
        (x + 7, y + 6),
        label,
        font=_font(9, bold=True),
        fill="#8fb5aa",
    )
    if tile is None:
        draw.text((x + width - 25, y + 22), "—", font=_font(14), fill="#597b71")
    else:
        _tile(
            draw,
            tile,
            x + width - 37,
            y + 7,
            width=27,
            height=38,
            marker=marker,
        )


def _footer_narration(frame: Mapping[str, object]) -> str:
    terminal = frame["terminal"]
    assert isinstance(terminal, Mapping)
    if not terminal["is_terminal"]:
        return str(frame["narration"])
    terminal_labels = {
        "stock_exhausted": "Stok bitti · el sona erdi.",
        "normal_finish": "Oyuncu eli bitirdi.",
        "okey_finish": "Okey ile bitiş.",
        "elden_finish": "Elden bitiş.",
        "elden_okey_finish": "Elden Okey ile bitiş.",
        "pair_finish": "Çift açarak bitiş.",
        "pair_okey_finish": "Çift açıp Okey ile bitiş.",
        "same_turn_open_finish": "Aynı tur açıp bitiş.",
        "same_turn_open_okey_finish": "Aynı tur açıp Okey ile bitiş.",
        "all_players_opened_pairs": "Tüm oyuncular çift açtı · el sona erdi.",
    }
    return terminal_labels.get(
        str(terminal["reason"]),
        "El sona erdi.",
    )


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

    image = Image.new("RGB", size, "#093226")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 70), fill="#071f19")
    draw.line((0, 69, width, 69), fill="#5f7e61", width=1)
    checkpoint = replay["checkpoint"]
    draw.text(
        (24, 12),
        "101 OKEY · AI GELİŞİMİ",
        font=_font(25, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (24, 42),
        f"{checkpoint['label']} · Eğitim adımı {checkpoint['training_step']:,}",
        font=_font(16),
        fill="#8fd8c2",
    )
    draw.text(
        (width - 265, 15),
        f"Kare {frame_index + 1}/{len(frames)}",
        font=_font(16, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (width - 265, 41),
        (
            f"Tur {view['turn_number']} · "
            f"{_PHASE_LABELS.get(str(view['phase']), str(view['phase']).upper())}"
        ),
        font=_font(14),
        fill="#9db9b1",
    )

    players = view["players"]
    current = int(view["current_player"])
    side_width = max(176, round(width * 0.165))
    center_left = side_width + 18
    center_right = width - side_width - 18
    footer_y = height - 68
    bottom_top = height - 168
    _player_panel(
        draw,
        players[2],
        (center_left, 78, center_right, 166),
        active=current == 2,
        columns=22,
    )
    _player_panel(
        draw,
        players[0],
        (center_left, bottom_top, center_right, footer_y - 8),
        active=current == 0,
        columns=22,
    )
    _player_panel(
        draw,
        players[3],
        (10, 104, side_width, height - 112),
        active=current == 3,
        columns=5,
    )
    _player_panel(
        draw,
        players[1],
        (width - side_width, 104, width - 10, height - 112),
        active=current == 1,
        columns=5,
    )

    arena_top = 174
    arena_bottom = bottom_top - 8
    _rounded(
        draw,
        (center_left, arena_top, center_right, arena_bottom),
        radius=11,
        fill="#164f39",
        outline="#6f9d7d",
    )
    draw.line(
        (center_left + 8, arena_top + 7, center_right - 8, arena_top + 7),
        fill="#2e6c50",
        width=1,
    )

    status_y = arena_top + 7
    status_gap = 8
    central_status_width = min(160, max(128, (center_right - center_left) // 5))
    status_total = central_status_width * 3 + status_gap * 2
    status_left = center_left + (center_right - center_left - status_total) // 2
    indicator = view["indicator"]
    _status_tile(
        draw,
        label="GÖSTERGE",
        tile=indicator,
        x=status_left,
        y=status_y,
        width=central_status_width,
        marker="flower",
    )
    okey_value = view["okey_value"]
    assert isinstance(okey_value, Mapping)
    okey_color = _COLOR_LABELS.get(
        str(okey_value.get("color")),
        str(okey_value.get("color", "")),
    )
    okey_tile: Mapping[str, object] = {"display": okey_value}
    second_x = status_left + central_status_width + status_gap
    _status_tile(
        draw,
        label=f"OKEY · {okey_color.upper()} {okey_value['number']}",
        tile=okey_tile,
        x=second_x,
        y=status_y,
        width=central_status_width,
        marker="flower",
    )
    stock_x = second_x + central_status_width + status_gap
    _rounded(
        draw,
        (
            stock_x,
            status_y,
            stock_x + central_status_width,
            status_y + 51,
        ),
        radius=7,
        fill="#382519",
        outline="#9a6c3d",
    )
    draw.text(
        (stock_x + 8, status_y + 6),
        "ORTADAKİ TAŞ",
        font=_font(9, bold=True),
        fill="#8fb5aa",
    )
    draw.text(
        (stock_x + 11, status_y + 21),
        str(view["stock_count"]),
        font=_font(22, bold=True),
        fill="#ffffff",
    )
    draw.text(
        (stock_x + 50, status_y + 28),
        f"Atılan {len(view['discard_pile'])}",
        font=_font(10),
        fill="#ffd77a",
    )
    table_top = status_y + 61
    table_bottom = arena_bottom - 61
    table_left = center_left + 103
    table_right = center_right - 103
    _rounded(
        draw,
        (table_left, table_top, table_right, table_bottom),
        radius=10,
        fill="#7a4d29",
        outline="#b7864f",
    )
    for wood_y in range(table_top + 14, table_bottom, 24):
        draw.line(
            (table_left + 5, wood_y, table_right - 5, wood_y),
            fill="#68401f",
            width=1,
        )
    draw.text(
        (
            (table_left + table_right) // 2 - 42,
            (table_top + table_bottom) // 2 - 19,
        ),
        "101",
        font=_font(40, bold=True),
        fill="#5e351c",
    )
    _table(
        draw,
        view["table"],
        (table_left + 12, table_top + 10, table_right - 12, table_bottom - 8),
    )
    discard_history = view.get("discard_history", [])
    assert isinstance(discard_history, Sequence)
    pocket_width = 98
    pocket_height = 50
    _discard_pocket(
        draw,
        discard_history,
        player_id=3,
        label="O4",
        box=(
            center_left + 8,
            table_top,
            center_left + 8 + pocket_width,
            table_top + pocket_height,
        ),
    )
    _discard_pocket(
        draw,
        discard_history,
        player_id=1,
        label="O2",
        box=(
            center_right - 8 - pocket_width,
            table_top,
            center_right - 8,
            table_top + pocket_height,
        ),
    )
    _discard_pocket(
        draw,
        discard_history,
        player_id=0,
        label="O1",
        box=(
            center_left + 8,
            table_bottom + 7,
            center_left + 8 + pocket_width,
            table_bottom + 7 + pocket_height,
        ),
    )
    _discard_pocket(
        draw,
        discard_history,
        player_id=2,
        label="O3",
        box=(
            center_right - 8 - pocket_width,
            table_bottom + 7,
            center_right - 8,
            table_bottom + 7 + pocket_height,
        ),
    )

    draw.rectangle((0, footer_y, width, height), fill="#081f1a")
    draw.text(
        (28, footer_y + 12),
        _footer_narration(frame),
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
