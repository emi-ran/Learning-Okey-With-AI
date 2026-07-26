"""Dependency-light neural self-play training."""

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from .model import NumpyActorCritic
from .replay import DeterministicPolicySelector
from .trainer import (
    EpisodeTrainingResult,
    EvaluationResult,
    SelfPlayTrainer,
    TrainingConfig,
    evaluate_against_random,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DeterministicPolicySelector",
    "EpisodeTrainingResult",
    "EvaluationResult",
    "NumpyActorCritic",
    "SelfPlayTrainer",
    "TrainingConfig",
    "evaluate_against_random",
    "load_checkpoint",
    "save_checkpoint",
]
