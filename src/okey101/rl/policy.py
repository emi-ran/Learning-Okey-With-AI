"""Strict model-facing boundary for variable-candidate policies."""

from __future__ import annotations

from dataclasses import dataclass

from okey101.engine.config import GameConfig

from .action_codec import catalog_from_actions
from .candidate_encoder import CANDIDATE_FEATURE_SIZE, encode_candidate
from .encoder import EncodedObservationV1, encode_observation
from .env import Decision
from .masks import build_action_mask


@dataclass(frozen=True, slots=True)
class ModelInput:
    """ID-free tensors-as-tuples supplied to a policy model."""

    observation: EncodedObservationV1
    candidate_features: tuple[tuple[float, ...], ...]
    action_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.candidate_features) != len(self.action_mask):
            raise ValueError(
                "candidate feature rows and action mask must have equal length"
            )
        for index, (features, enabled) in enumerate(
            zip(self.candidate_features, self.action_mask, strict=True)
        ):
            if not isinstance(enabled, bool):
                raise TypeError("action mask values must be booleans")
            if len(features) != CANDIDATE_FEATURE_SIZE:
                raise ValueError(
                    f"candidate feature row {index} has an invalid width"
                )
            if not all(isinstance(value, float) for value in features):
                raise TypeError("candidate features must contain only floats")
            if not enabled and any(features):
                raise ValueError("padding candidate features must contain only zeros")
        seen_padding = False
        for enabled in self.action_mask:
            if not enabled:
                seen_padding = True
            elif seen_padding:
                raise ValueError("action mask may disable only trailing padding")


def prepare_model_input(
    decision: Decision,
    config: GameConfig,
    *,
    capacity: int | None = None,
) -> ModelInput:
    """Encode a runner decision without returning actions or their resolver.

    The runner retains or deterministically rebuilds the same ``ActionCatalog``
    to decode the integer selected by the model. The model receives only
    encoded observation data, ID-free candidate rows and a padding-only mask.
    """

    if not decision.legal_actions:
        raise ValueError("a model decision requires at least one legal action")
    catalog = catalog_from_actions(decision.legal_actions)
    mask = build_action_mask(catalog, capacity=capacity)
    zero_row = (0.0,) * CANDIDATE_FEATURE_SIZE
    rows = tuple(
        encode_candidate(
            decision.observation,
            candidate,
            config,
        ).as_vector()
        for candidate in catalog.candidates
    )
    rows += (zero_row,) * (len(mask) - len(rows))
    return ModelInput(
        observation=encode_observation(decision.observation, config),
        candidate_features=rows,
        action_mask=mask,
    )
