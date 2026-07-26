"""Public deterministic replay recording API."""

from .comparison import build_comparison_manifest
from .recorder import (
    ActionSelection,
    ActionSelector,
    CandidateProbability,
    CheckpointMetadata,
    ReplayVerificationError,
    SelectionContext,
    record_episode,
    record_random_episode,
    verify_replay,
)
from .schema import (
    COMPARISON_SCHEMA_VERSION,
    REPLAY_SCHEMA_VERSION,
    ReplayValidationError,
    load_comparison_manifest,
    load_replay,
    validate_comparison_manifest,
    validate_replay_document,
    write_comparison_manifest,
    write_replay,
)

__all__ = [
    "ActionSelection",
    "ActionSelector",
    "COMPARISON_SCHEMA_VERSION",
    "CandidateProbability",
    "CheckpointMetadata",
    "REPLAY_SCHEMA_VERSION",
    "ReplayValidationError",
    "ReplayVerificationError",
    "SelectionContext",
    "build_comparison_manifest",
    "load_comparison_manifest",
    "load_replay",
    "record_episode",
    "record_random_episode",
    "validate_comparison_manifest",
    "validate_replay_document",
    "verify_replay",
    "write_comparison_manifest",
    "write_replay",
]
