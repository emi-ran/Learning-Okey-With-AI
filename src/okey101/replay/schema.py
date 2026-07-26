"""Versioned, renderer-ready replay JSON contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

REPLAY_SCHEMA_VERSION = "okey101.replay.v1"
COMPARISON_SCHEMA_VERSION = "okey101.replay-comparison.v1"


class ReplayValidationError(ValueError):
    """Raised when replay JSON does not satisfy the public schema."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _invalid(path: str, message: str) -> NoReturn:
    raise ReplayValidationError(f"Invalid replay at {path}: {message}")


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid(path, "expected an object")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        _invalid(path, "expected an array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        _invalid(path, "expected a non-empty string")
    return value


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(path, "expected an integer")
    if minimum is not None and value < minimum:
        _invalid(path, f"must be at least {minimum}")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(path, "expected a number")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        _invalid(path, "must be between 0 and 1")
    return numeric


def _required(data: Mapping[str, object], key: str, path: str) -> object:
    if key not in data:
        _invalid(path, f"missing field {key!r}")
    return data[key]


def _validate_tile(value: object, path: str) -> None:
    tile = _object(value, path)
    _integer(_required(tile, "id", path), f"{path}.id", minimum=0)
    kind = _string(_required(tile, "kind", path), f"{path}.kind")
    if kind not in {"normal", "fake_okey"}:
        _invalid(f"{path}.kind", "unsupported tile kind")
    display = _object(_required(tile, "display", path), f"{path}.display")
    _string(_required(display, "color", f"{path}.display"), f"{path}.display.color")
    _integer(
        _required(display, "number", f"{path}.display"),
        f"{path}.display.number",
        minimum=1,
    )
    for flag in ("is_real_okey", "is_fake_okey", "is_indicator"):
        if not isinstance(_required(tile, flag, path), bool):
            _invalid(f"{path}.{flag}", "expected a boolean")


def _validate_policy_step(value: object, path: str) -> None:
    policy = _object(value, path)
    if policy.get("selected_probability") is not None:
        _number(policy["selected_probability"], f"{path}.selected_probability")
    if policy.get("value") is not None:
        raw = policy["value"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            _invalid(f"{path}.value", "expected a number or null")
    candidates = _array(policy.get("candidates", ()), f"{path}.candidates")
    for index, item in enumerate(candidates):
        candidate_path = f"{path}.candidates[{index}]"
        candidate = _object(item, candidate_path)
        _object(
            _required(candidate, "action", candidate_path),
            f"{candidate_path}.action",
        )
        _number(
            _required(candidate, "probability", candidate_path),
            f"{candidate_path}.probability",
        )


def validate_replay_document(value: object) -> dict[str, Any]:
    """Validate and return a plain replay document."""

    data = _object(value, "$")
    if data.get("schema_version") != REPLAY_SCHEMA_VERSION:
        _invalid("$.schema_version", f"expected {REPLAY_SCHEMA_VERSION!r}")
    if data.get("kind") != "episode_replay":
        _invalid("$.kind", "expected 'episode_replay'")

    visibility = _object(_required(data, "visibility", "$"), "$.visibility")
    if visibility.get("mode") != "spectator":
        _invalid("$.visibility.mode", "only explicit spectator replays are supported")
    if visibility.get("reveal_all_hands") is not True:
        _invalid(
            "$.visibility.reveal_all_hands",
            "spectator replay must explicitly reveal all hands",
        )

    episode = _object(_required(data, "episode", "$"), "$.episode")
    _integer(_required(episode, "seed", "$.episode"), "$.episode.seed")
    _integer(
        _required(episode, "round_id", "$.episode"),
        "$.episode.round_id",
        minimum=1,
    )
    _integer(
        _required(episode, "starting_player", "$.episode"),
        "$.episode.starting_player",
        minimum=0,
    )

    checkpoint = _object(_required(data, "checkpoint", "$"), "$.checkpoint")
    _string(_required(checkpoint, "id", "$.checkpoint"), "$.checkpoint.id")
    _string(_required(checkpoint, "label", "$.checkpoint"), "$.checkpoint.label")
    _integer(
        _required(checkpoint, "training_step", "$.checkpoint"),
        "$.checkpoint.training_step",
        minimum=0,
    )
    _object(_required(data, "policy", "$"), "$.policy")
    _object(_required(data, "config", "$"), "$.config")

    frames = _array(_required(data, "frames", "$"), "$.frames")
    if not frames:
        _invalid("$.frames", "must contain at least the initial frame")
    for index, item in enumerate(frames):
        path = f"$.frames[{index}]"
        frame = _object(item, path)
        if _integer(
            _required(frame, "frame_index", path),
            f"{path}.frame_index",
            minimum=0,
        ) != index:
            _invalid(f"{path}.frame_index", "must match array position")
        action_index = _required(frame, "action_index", path)
        action = _required(frame, "action", path)
        actor = _required(frame, "actor_seat", path)
        if index == 0:
            if action_index is not None or action is not None or actor is not None:
                _invalid(path, "initial frame cannot contain an action or actor")
        else:
            if _integer(action_index, f"{path}.action_index", minimum=0) != index - 1:
                _invalid(f"{path}.action_index", "must be frame_index - 1")
            _object(action, f"{path}.action")
            _integer(actor, f"{path}.actor_seat", minimum=0)
        _string(_required(frame, "narration", path), f"{path}.narration")
        events = _array(_required(frame, "events", path), f"{path}.events")
        for event_index, event_value in enumerate(events):
            event_path = f"{path}.events[{event_index}]"
            event = _object(event_value, event_path)
            _string(_required(event, "type", event_path), f"{event_path}.type")
            _string(
                _required(event, "narration", event_path),
                f"{event_path}.narration",
            )
            _object(_required(event, "details", event_path), f"{event_path}.details")

        view = _object(_required(frame, "view", path), f"{path}.view")
        _validate_tile(
            _required(view, "indicator", f"{path}.view"),
            f"{path}.view.indicator",
        )
        players = _array(
            _required(view, "players", f"{path}.view"),
            f"{path}.view.players",
        )
        if not players:
            _invalid(f"{path}.view.players", "must not be empty")
        for player_index, player_value in enumerate(players):
            player_path = f"{path}.view.players[{player_index}]"
            player = _object(player_value, player_path)
            if _integer(
                _required(player, "seat", player_path),
                f"{player_path}.seat",
                minimum=0,
            ) != player_index:
                _invalid(f"{player_path}.seat", "must match array position")
            hand = _array(
                _required(player, "hand", player_path),
                f"{player_path}.hand",
            )
            for tile_index, tile in enumerate(hand):
                _validate_tile(tile, f"{player_path}.hand[{tile_index}]")
            if _integer(
                _required(player, "hand_count", player_path),
                f"{player_path}.hand_count",
                minimum=0,
            ) != len(hand):
                _invalid(f"{player_path}.hand_count", "must equal visible hand length")

        _object(_required(frame, "scores", path), f"{path}.scores")
        terminal = _object(_required(frame, "terminal", path), f"{path}.terminal")
        if not isinstance(_required(terminal, "is_terminal", f"{path}.terminal"), bool):
            _invalid(f"{path}.terminal.is_terminal", "expected a boolean")
        _string(_required(frame, "state_digest", path), f"{path}.state_digest")
        if frame.get("policy_step") is not None:
            _validate_policy_step(frame["policy_step"], f"{path}.policy_step")

    summary = _object(_required(data, "summary", "$"), "$.summary")
    action_count = _integer(
        _required(summary, "action_count", "$.summary"),
        "$.summary.action_count",
        minimum=0,
    )
    if action_count != len(frames) - 1:
        _invalid("$.summary.action_count", "must equal len(frames) - 1")
    if not _object(frames[-1], "$.frames[-1]").get("terminal", {}).get(
        "is_terminal"
    ):
        _invalid("$.frames[-1].terminal", "last frame must be terminal")

    integrity = _object(_required(data, "integrity", "$"), "$.integrity")
    if integrity.get("algorithm") != "sha256":
        _invalid("$.integrity.algorithm", "expected 'sha256'")
    expected_digest = _string(
        _required(integrity, "document_digest", "$.integrity"),
        "$.integrity.document_digest",
    )
    unsigned = dict(data)
    unsigned.pop("integrity", None)
    if content_digest(unsigned) != expected_digest:
        _invalid("$.integrity.document_digest", "content digest mismatch")
    return dict(data)


def load_replay(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(f"Invalid replay JSON: {exc}") from exc
    return validate_replay_document(payload)


def write_replay(document: Mapping[str, object], path: str | Path) -> Path:
    validated = validate_replay_document(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def validate_comparison_manifest(value: object) -> dict[str, Any]:
    data = _object(value, "$")
    if data.get("schema_version") != COMPARISON_SCHEMA_VERSION:
        _invalid("$.schema_version", f"expected {COMPARISON_SCHEMA_VERSION!r}")
    if data.get("kind") != "checkpoint_comparison":
        _invalid("$.kind", "expected 'checkpoint_comparison'")
    _string(_required(data, "title", "$"), "$.title")
    seeds = [
        _integer(seed, f"$.fixed_seeds[{index}]")
        for index, seed in enumerate(
            _array(_required(data, "fixed_seeds", "$"), "$.fixed_seeds")
        )
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        _invalid("$.fixed_seeds", "must be a non-empty unique seed list")
    checkpoints = _array(_required(data, "checkpoints", "$"), "$.checkpoints")
    if not checkpoints:
        _invalid("$.checkpoints", "must not be empty")
    for index, item in enumerate(checkpoints):
        path = f"$.checkpoints[{index}]"
        checkpoint = _object(item, path)
        _string(_required(checkpoint, "id", path), f"{path}.id")
        _string(_required(checkpoint, "label", path), f"{path}.label")
        _integer(
            _required(checkpoint, "training_step", path),
            f"{path}.training_step",
            minimum=0,
        )
        replays = _array(_required(checkpoint, "replays", path), f"{path}.replays")
        replay_seeds = []
        for replay_index, replay_value in enumerate(replays):
            replay_path = f"{path}.replays[{replay_index}]"
            replay = _object(replay_value, replay_path)
            replay_seeds.append(
                _integer(_required(replay, "seed", replay_path), f"{replay_path}.seed")
            )
            _string(_required(replay, "path", replay_path), f"{replay_path}.path")
            _string(
                _required(replay, "document_digest", replay_path),
                f"{replay_path}.document_digest",
            )
            _object(_required(replay, "summary", replay_path), f"{replay_path}.summary")
        if replay_seeds != seeds:
            _invalid(f"{path}.replays", "must match fixed_seeds in order")
    return dict(data)


def load_comparison_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(
                f"Invalid comparison manifest JSON: {exc}"
            ) from exc
    return validate_comparison_manifest(payload)


def write_comparison_manifest(
    manifest: Mapping[str, object],
    path: str | Path,
) -> Path:
    validated = validate_comparison_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validated, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output
