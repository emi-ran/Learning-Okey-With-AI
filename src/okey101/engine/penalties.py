"""Immediate, legal-but-penalized action scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .joker import is_real_okey

if TYPE_CHECKING:
    from .config import GameConfig, RulesConfig, ScoringConfig
    from .state import GameState
    from .tiles import PhysicalTile


def calculate_discard_penalty(
    state: GameState,
    tile: PhysicalTile,
    *,
    is_final: bool,
    config: ScoringConfig | RulesConfig | GameConfig,
    is_playable: bool | None = None,
) -> int:
    """Return the immediate penalty for a legal discard.

    Final-discard immunity is deliberately checked first. A non-final real Okey
    uses the Okey-discard rule and is not double-charged as a playable tile.
    ``is_playable`` is injectable so transition tests can isolate penalty logic;
    production callers may let the legal-action module derive it from the table.
    """

    if is_final:
        return 0
    scoring = getattr(config, "scoring", config)
    if is_real_okey(tile, state.okey_value):
        return int(getattr(scoring, "normal_okey_discard_penalty", 101))

    if is_playable is None:
        from .legal_actions import is_playable_discard

        rules_config = config if hasattr(config, "max_contiguous_attach") else None
        is_playable = is_playable_discard(state, tile, rules_config)
    if is_playable:
        return int(getattr(scoring, "playable_discard_penalty", 101))
    return 0
