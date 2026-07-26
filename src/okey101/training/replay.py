"""Replay selector adapter for deterministic trained-policy recordings."""

from __future__ import annotations

from collections.abc import Sequence

from okey101.engine.actions import Action
from okey101.engine.config import GameConfig
from okey101.replay import (
    ActionSelection,
    CandidateProbability,
    SelectionContext,
)
from okey101.rl.action_codec import catalog_from_actions
from okey101.rl.env import Decision
from okey101.rl.observation import PlayerObservation
from okey101.rl.policy import prepare_model_input

from .model import NumpyActorCritic


class DeterministicPolicySelector:
    """Adapt a greedy model to the recorder's exact public selector contract."""

    def __init__(
        self,
        model: NumpyActorCritic,
        config: GameConfig,
        *,
        top_candidates: int = 5,
    ) -> None:
        if (
            isinstance(top_candidates, bool)
            or not isinstance(top_candidates, int)
        ):
            raise TypeError("top_candidates must be an integer")
        if top_candidates < 1:
            raise ValueError("top_candidates must be positive")
        self.model = model
        self.config = config
        self.top_candidates = top_candidates

    def __call__(
        self,
        observation: PlayerObservation,
        legal_actions: Sequence[Action],
        context: SelectionContext,
    ) -> ActionSelection:
        if not legal_actions:
            raise ValueError("deterministic selector requires legal actions")
        decision = Decision(
            seat=context.player_id,
            observation=observation,
            legal_actions=tuple(legal_actions),
        )
        model_input = prepare_model_input(decision, self.config)
        probabilities, value = self.model.forward(model_input)
        selected_index = int(probabilities.argmax())
        catalog = catalog_from_actions(legal_actions)
        ranked_indices = sorted(
            range(len(probabilities)),
            key=lambda index: (-float(probabilities[index]), index),
        )[: self.top_candidates]
        return ActionSelection(
            action=catalog.decode(selected_index),
            selected_probability=float(probabilities[selected_index]),
            value=float(value),
            candidates=tuple(
                CandidateProbability(
                    action=catalog.decode(index),
                    probability=float(probabilities[index]),
                )
                for index in ranked_indices
            ),
        )
