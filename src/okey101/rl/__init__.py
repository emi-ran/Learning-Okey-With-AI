"""Reinforcement-learning integration primitives."""

from .action_codec import (
    ACTION_CODEC_VERSION,
    ActionCandidate,
    ActionCatalog,
    build_action_catalog,
    catalog_from_actions,
)
from .candidate_encoder import (
    CANDIDATE_ENCODER_VERSION,
    CANDIDATE_FEATURE_SIZE,
    CandidateFeatures,
    encode_candidate,
    encode_catalog,
)
from .encoder import (
    OBS_SCHEMA_VERSION,
    EncodedObservationV1,
    encode_observation,
)
from .env import Decision, SingleRoundEnv, StepResult
from .masks import build_action_mask
from .observation import PlayerObservation, get_observation
from .policy import ModelInput, prepare_model_input
from .rewards import relative_terminal_rewards
from .vector_env import VectorRoundEnv

__all__ = [
    "ACTION_CODEC_VERSION",
    "CANDIDATE_ENCODER_VERSION",
    "CANDIDATE_FEATURE_SIZE",
    "OBS_SCHEMA_VERSION",
    "ActionCandidate",
    "ActionCatalog",
    "CandidateFeatures",
    "Decision",
    "EncodedObservationV1",
    "ModelInput",
    "PlayerObservation",
    "SingleRoundEnv",
    "StepResult",
    "VectorRoundEnv",
    "build_action_catalog",
    "build_action_mask",
    "catalog_from_actions",
    "encode_observation",
    "encode_candidate",
    "encode_catalog",
    "get_observation",
    "relative_terminal_rewards",
    "prepare_model_input",
]
