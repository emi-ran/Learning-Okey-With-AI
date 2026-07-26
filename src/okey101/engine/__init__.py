"""Public primitives for the deterministic rules engine."""

from .config import GameConfig, RulesConfig, ScoringConfig
from .deck import Deck, build_deck
from .joker import effective_value, is_real_okey, okey_value_for_indicator
from .match import (
    MatchEngine,
    MatchReplayError,
    RoundRecord,
    derive_round_seed,
    replay_match_from_seed_and_actions,
)
from .melds import (
    Meld,
    MeldKind,
    MeldTile,
    build_meld,
    find_meld_assignments,
    validate_meld,
)
from .pairs import Pair, build_pair, validate_pair
from .state import DiscardRecord
from .tiles import (
    Color,
    PhysicalTile,
    TileKind,
    TileValue,
    build_tile_set,
)

__all__ = [
    "Color",
    "Deck",
    "DiscardRecord",
    "GameConfig",
    "Meld",
    "MeldKind",
    "MeldTile",
    "MatchEngine",
    "MatchReplayError",
    "Pair",
    "PhysicalTile",
    "RulesConfig",
    "RoundRecord",
    "ScoringConfig",
    "TileKind",
    "TileValue",
    "build_deck",
    "build_meld",
    "build_pair",
    "build_tile_set",
    "derive_round_seed",
    "effective_value",
    "find_meld_assignments",
    "is_real_okey",
    "okey_value_for_indicator",
    "replay_match_from_seed_and_actions",
    "validate_meld",
    "validate_pair",
]
