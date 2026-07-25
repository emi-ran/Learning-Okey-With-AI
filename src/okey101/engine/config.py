"""Explicit, immutable rule and scoring configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RulesConfig:
    """Rules which affect legal play rather than scoring."""

    player_count: int = 4
    initial_hand_size: int = 21
    starting_player_extra_tile: bool = True
    opening_min_score: int = 101
    opening_min_pairs: int = 5
    progressive_opening: bool = False
    max_contiguous_attach: int = 2
    require_final_discard: bool = True

    def __post_init__(self) -> None:
        if self.player_count < 2:
            raise ValueError("player_count must be at least 2")
        if self.initial_hand_size < 1:
            raise ValueError("initial_hand_size must be positive")
        if self.opening_min_score < 1:
            raise ValueError("opening_min_score must be positive")
        if self.opening_min_pairs < 1:
            raise ValueError("opening_min_pairs must be positive")
        if self.max_contiguous_attach < 1:
            raise ValueError("max_contiguous_attach must be positive")

    @property
    def starter_hand_size(self) -> int:
        """Number of tiles dealt to the player who starts the round."""

        return self.initial_hand_size + int(self.starting_player_extra_tile)


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Configurable score components from the locked project decisions."""

    normal_finish_reward: int = -101
    same_turn_open_finish_reward: int = -202
    same_turn_open_okey_finish_reward: int = -404
    okey_finish_reward: int = -202
    pair_finish_reward: int = -202
    elden_finish_reward: int = -202
    elden_okey_finish_reward: int = -404
    pair_okey_finish_reward: int = -404

    unopened_end_penalty: int = 202
    pair_remaining_multiplier: int = 2
    same_turn_open_finish_opponent_multiplier: int = 1
    okey_finish_opponent_multiplier: int = 2
    elden_finish_opponent_multiplier: int = 2
    pair_finish_opponent_multiplier: int = 2

    playable_discard_penalty: int = 101
    normal_okey_discard_penalty: int = 101
    opened_player_okey_in_hand_surcharge: int = 101

    multiply_okey_in_hand_surcharge_by_pair: bool = False
    multiply_okey_in_hand_surcharge_by_finish: bool = False


@dataclass(frozen=True, slots=True)
class GameConfig(RulesConfig):
    """Top-level configuration for a match."""

    rounds: int = 1
    void_round_counts_toward_match: bool = False
    scoring: ScoringConfig = ScoringConfig()

    def __post_init__(self) -> None:
        super(GameConfig, self).__post_init__()
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
