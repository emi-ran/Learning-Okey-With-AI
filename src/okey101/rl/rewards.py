"""Dependency-free reward functions for multi-seat self-play."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import TypeAlias

RewardFn: TypeAlias = Callable[[tuple[int, ...]], tuple[float, ...]]


def relative_terminal_rewards(
    scores: Sequence[int],
    *,
    scale: float = 101.0,
) -> tuple[float, ...]:
    """Convert lower-is-better engine scores into zero-sum seat rewards.

    Each seat receives its opponents' mean score minus its own score. The
    default scale expresses the result in units of one standard 101-point
    penalty.
    """

    if len(scores) < 2:
        raise ValueError("relative rewards require at least two seat scores")
    if not isfinite(scale) or scale <= 0:
        raise ValueError("reward scale must be finite and positive")
    if any(isinstance(score, bool) or not isinstance(score, int) for score in scores):
        raise TypeError("scores must contain only integers")

    total = sum(scores)
    opponent_count = len(scores) - 1
    return tuple(
        (((total - score) / opponent_count) - score) / scale
        for score in scores
    )
