"""Dependency-free action masks for variable-length legal-action catalogs."""

from __future__ import annotations

from .action_codec import ActionCatalog


class CandidateCapacityError(ValueError):
    """Raised when a padded action space cannot contain every legal action."""


def build_action_mask(
    catalog: ActionCatalog,
    *,
    capacity: int | None = None,
) -> tuple[bool, ...]:
    """Return a mask that disables padding, never legal actions.

    The codec catalog already contains only legal actions. Consequently every
    catalog entry, including legal-but-penalized actions, remains enabled.
    """

    legal_count = len(catalog)
    if capacity is None:
        capacity = legal_count
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer or None")
    if capacity < legal_count:
        raise CandidateCapacityError(
            f"capacity {capacity} cannot contain {legal_count} legal actions"
        )
    return (True,) * legal_count + (False,) * (capacity - legal_count)
