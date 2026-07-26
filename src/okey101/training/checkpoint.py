"""Safe, deterministic NumPy checkpoint persistence."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from okey101.engine.config import GameConfig, ScoringConfig
from okey101.rl.action_codec import ACTION_CODEC_VERSION
from okey101.rl.candidate_encoder import CANDIDATE_ENCODER_VERSION
from okey101.rl.encoder import OBS_SCHEMA_VERSION

from .trainer import SelfPlayTrainer, TrainingConfig

CHECKPOINT_SCHEMA_VERSION = 1


def _git_hash() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    if not (repository_root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=repository_root,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _metadata(trainer: SelfPlayTrainer) -> dict[str, object]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema": {
            "observation": OBS_SCHEMA_VERSION,
            "candidate": CANDIDATE_ENCODER_VERSION,
            "action_codec": ACTION_CODEC_VERSION,
        },
        "git_hash": _git_hash(),
        "training_config": asdict(trainer.training_config),
        "game_config": asdict(trainer.game_config),
        "model": {
            "observation_size": trainer.model.observation_size,
            "hidden_size": trainer.model.hidden_size,
        },
        "progress": {
            "episodes_completed": trainer.episodes_completed,
            "actions_completed": trainer.actions_completed,
            "optimizer_step": trainer.optimizer.step,
        },
        "rng_state": trainer.rng.bit_generator.state,
    }


def save_checkpoint(trainer: SelfPlayTrainer, path: str | Path) -> Path:
    """Atomically save arrays and JSON metadata without pickle."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, object] = {
        "metadata": np.asarray(
            json.dumps(_metadata(trainer), sort_keys=True),
        )
    }
    for name, value in trainer.model.parameters.items():
        arrays[f"model.{name}"] = value
    for name, value in trainer.optimizer.first_moment.items():
        arrays[f"adam.first.{name}"] = value
    for name, value in trainer.optimizer.second_moment.items():
        arrays[f"adam.second.{name}"] = value

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {name} must be an object")
    return value


def _load_metadata(archive: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata" not in archive.files:
        raise ValueError("checkpoint is missing metadata")
    raw = archive["metadata"]
    if raw.shape != () or raw.dtype.kind not in {"U", "S"}:
        raise ValueError("checkpoint metadata must be a scalar string")
    try:
        return _mapping(json.loads(str(raw.item())), "metadata")
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("checkpoint metadata is invalid JSON") from error


def load_checkpoint(path: str | Path) -> SelfPlayTrainer:
    """Restore model, Adam moments, progress, configs, and exact RNG state."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = _load_metadata(archive)
        if metadata.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema version")
        expected_features = {
            "observation": OBS_SCHEMA_VERSION,
            "candidate": CANDIDATE_ENCODER_VERSION,
            "action_codec": ACTION_CODEC_VERSION,
        }
        if metadata.get("feature_schema") != expected_features:
            raise ValueError("checkpoint feature schema does not match runtime")

        training_data = _mapping(
            metadata.get("training_config"),
            "training_config",
        )
        game_data = _mapping(metadata.get("game_config"), "game_config")
        scoring_data = _mapping(game_data.pop("scoring", None), "game scoring")
        trainer = SelfPlayTrainer(
            seed=0,
            training_config=TrainingConfig(**training_data),
            game_config=GameConfig(
                **game_data,
                scoring=ScoringConfig(**scoring_data),
            ),
        )

        model_parameters = {}
        for name in trainer.model.PARAMETER_NAMES:
            key = f"model.{name}"
            if key not in archive.files:
                raise ValueError(f"checkpoint is missing array: {key}")
            model_parameters[name] = archive[key]
        trainer.model.load_parameters(model_parameters)

        for prefix, destination in (
            ("adam.first", trainer.optimizer.first_moment),
            ("adam.second", trainer.optimizer.second_moment),
        ):
            for name, current in destination.items():
                key = f"{prefix}.{name}"
                if key not in archive.files:
                    raise ValueError(f"checkpoint is missing array: {key}")
                incoming = np.asarray(archive[key], dtype=np.float64)
                if incoming.shape != current.shape:
                    raise ValueError(f"checkpoint optimizer shape mismatch: {key}")
                current[...] = incoming

        progress = _mapping(metadata.get("progress"), "progress")
        trainer.episodes_completed = int(progress["episodes_completed"])
        trainer.actions_completed = int(progress["actions_completed"])
        trainer.optimizer.step = int(progress["optimizer_step"])
        rng_state = _mapping(metadata.get("rng_state"), "rng_state")
        trainer.rng.bit_generator.state = rng_state
        return trainer
