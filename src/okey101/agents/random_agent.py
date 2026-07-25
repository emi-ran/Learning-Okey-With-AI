"""Uniform legal-action baseline used for engine stress tests."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Generic

from .base import ActionT, Agent, NoLegalActionsError, ObservationT


class RandomAgent(Agent[ObservationT, ActionT], Generic[ObservationT, ActionT]):
    """Select uniformly from the legal actions supplied by the engine."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)

    def select_action(
        self,
        observation: ObservationT,
        legal_actions: Sequence[ActionT],
    ) -> ActionT:
        del observation
        if not legal_actions:
            raise NoLegalActionsError("RandomAgent received no legal actions")
        return legal_actions[self._rng.randrange(len(legal_actions))]
