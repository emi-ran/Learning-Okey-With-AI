"""Checkpoint comparison manifests over a fixed seed set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import (
    COMPARISON_SCHEMA_VERSION,
    validate_comparison_manifest,
    validate_replay_document,
)


def build_comparison_manifest(
    *,
    title: str,
    fixed_seeds: Sequence[int],
    checkpoints: Sequence[tuple[Mapping[str, object], Sequence[tuple[str, Mapping[str, object]]]]],
) -> dict[str, Any]:
    """Build a renderer playlist where every checkpoint uses identical seeds."""

    seeds = list(fixed_seeds)
    payload_checkpoints = []
    for checkpoint, replay_items in checkpoints:
        replays = []
        if len(replay_items) != len(seeds):
            raise ValueError("each checkpoint must provide one replay per fixed seed")
        for expected_seed, (path, document) in zip(seeds, replay_items):
            replay = validate_replay_document(document)
            episode = replay["episode"]
            if episode["seed"] != expected_seed:
                raise ValueError("checkpoint replay seeds must match fixed_seeds")
            if replay["checkpoint"] != checkpoint:
                raise ValueError("replay checkpoint metadata does not match manifest")
            replays.append(
                {
                    "seed": expected_seed,
                    "path": Path(path).as_posix(),
                    "document_digest": replay["integrity"]["document_digest"],
                    "summary": replay["summary"],
                }
            )
        payload_checkpoints.append(
            {
                "id": checkpoint["id"],
                "label": checkpoint["label"],
                "training_step": checkpoint["training_step"],
                "replays": replays,
            }
        )
    return validate_comparison_manifest(
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "kind": "checkpoint_comparison",
            "title": title,
            "fixed_seeds": seeds,
            "checkpoints": payload_checkpoints,
        }
    )
