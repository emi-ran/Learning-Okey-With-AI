"""Public primitives for the deterministic rules engine."""

from .config import GameConfig, RulesConfig, ScoringConfig
from .deck import Deck, build_deck
from .joker import effective_value, is_real_okey, okey_value_for_indicator
from .match import MatchEngine, RoundRecord, derive_round_seed
from .melds import (
    Meld,
    MeldKind,
    MeldTile,
    build_meld,
    find_meld_assignments,
    validate_meld,
)
from .pairs import Pair, build_pair, validate_pair
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
    "GameConfig",
    "Meld",
    "MeldKind",
    "MeldTile",
    "MatchEngine",
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
    "validate_meld",
    "validate_pair",
]
