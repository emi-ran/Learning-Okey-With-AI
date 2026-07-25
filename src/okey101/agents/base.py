"""Agent interfaces shared by rule-based and learned policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


class NoLegalActionsError(RuntimeError):
    """Raised when an agent is asked to act without a legal candidate."""


class Agent(ABC, Generic[ObservationT, ActionT]):
    """Minimal policy interface.

    The engine owns legality. Agents receive an already filtered, non-empty
    candidate list and choose one action without inspecting hidden state.
    """

    @abstractmethod
    def select_action(
        self,
        observation: ObservationT,
        legal_actions: Sequence[ActionT],
    ) -> ActionT:
        """Choose exactly one action from ``legal_actions``."""

    def reset(self, seed: int | None = None) -> None:
        """Reset episode-local state, optionally reseeding the policy."""
