"""Reinforcement-learning integration primitives."""

from .action_codec import (
    ACTION_CODEC_VERSION,
    ActionCandidate,
    ActionCatalog,
    build_action_catalog,
    catalog_from_actions,
)
from .encoder import (
    OBS_SCHEMA_VERSION,
    EncodedObservationV1,
    encode_observation,
)
from .env import Decision, SingleRoundEnv, StepResult
from .masks import build_action_mask
from .observation import PlayerObservation, get_observation
from .rewards import relative_terminal_rewards
from .vector_env import VectorRoundEnv

__all__ = [
    "ACTION_CODEC_VERSION",
    "OBS_SCHEMA_VERSION",
    "ActionCandidate",
    "ActionCatalog",
    "Decision",
    "EncodedObservationV1",
    "PlayerObservation",
    "SingleRoundEnv",
    "StepResult",
    "VectorRoundEnv",
    "build_action_catalog",
    "build_action_mask",
    "catalog_from_actions",
    "encode_observation",
    "get_observation",
    "relative_terminal_rewards",
]
