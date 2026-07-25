"""Composable end-of-round scoring for 101 Okey."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .joker import effective_value, is_real_okey
from .player import OpenedMode, PlayerState
from .state import GameState, TerminalReason
from .tiles import TileValue

if TYPE_CHECKING:
    from .config import GameConfig, ScoringConfig
    ScoringConfigLike = ScoringConfig | GameConfig
else:
    ScoringConfigLike = object


@dataclass(frozen=True, slots=True)
class HandScoreBreakdown:
    """Auditable components of one non-winning player's round score."""

    normal_value_sum: int
    unopened_penalty: int
    pair_multiplier: int
    finish_multiplier: int
    okey_count: int
    okey_surcharge: int
    immediate_penalty: int
    total: int


@dataclass(frozen=True, slots=True)
class PlayerRoundScore:
    player_id: int
    total: int
    winner_reward: int = 0
    hand: HandScoreBreakdown | None = None


@dataclass(frozen=True, slots=True)
class RoundScore:
    reason: TerminalReason
    players: tuple[PlayerRoundScore, ...]

    @property
    def totals(self) -> tuple[int, ...]:
        return tuple(player.total for player in self.players)


def _scoring_config(config: ScoringConfigLike) -> object:
    return getattr(config, "scoring", config)


def _config_int(config: ScoringConfigLike, name: str, default: int) -> int:
    return int(getattr(_scoring_config(config), name, default))


def _config_bool(
    config: ScoringConfigLike,
    name: str,
    default: bool = False,
) -> bool:
    return bool(getattr(_scoring_config(config), name, default))


def _finish_multiplier(reason: TerminalReason, config: ScoringConfigLike) -> int:
    okey = _config_int(config, "okey_finish_opponent_multiplier", 2)
    elden = _config_int(config, "elden_finish_opponent_multiplier", 2)
    pair = _config_int(config, "pair_finish_opponent_multiplier", 2)
    same_turn = _config_int(
        config,
        "same_turn_open_finish_opponent_multiplier",
        1,
    )

    if reason is TerminalReason.SAME_TURN_OPEN_OKEY_FINISH:
        return okey * same_turn
    if reason is TerminalReason.ELDEN_OKEY_FINISH:
        return okey * elden
    if reason is TerminalReason.PAIR_OKEY_FINISH:
        return okey * pair
    if reason is TerminalReason.OKEY_FINISH:
        return okey
    if reason is TerminalReason.SAME_TURN_OPEN_FINISH:
        return same_turn
    if reason is TerminalReason.ELDEN_FINISH:
        return elden
    if reason is TerminalReason.PAIR_FINISH:
        return pair
    return 1


def winner_reward(reason: TerminalReason, config: ScoringConfigLike) -> int:
    """Return the configured reward for the player who ended the round."""

    rewards = {
        TerminalReason.NORMAL_FINISH: ("normal_finish_reward", -101),
        TerminalReason.SAME_TURN_OPEN_FINISH: (
            "same_turn_open_finish_reward",
            -202,
        ),
        TerminalReason.SAME_TURN_OPEN_OKEY_FINISH: (
            "same_turn_open_okey_finish_reward",
            -404,
        ),
        TerminalReason.ELDEN_FINISH: ("elden_finish_reward", -202),
        TerminalReason.OKEY_FINISH: ("okey_finish_reward", -202),
        TerminalReason.ELDEN_OKEY_FINISH: ("elden_okey_finish_reward", -404),
        TerminalReason.PAIR_FINISH: ("pair_finish_reward", -202),
        TerminalReason.PAIR_OKEY_FINISH: ("pair_okey_finish_reward", -404),
    }
    try:
        field, default = rewards[reason]
    except KeyError as error:
        raise ValueError(f"{reason.value} does not have a winner reward") from error
    return _config_int(config, field, default)


def score_remaining_hand(
    player: PlayerState,
    okey_value: TileValue,
    *,
    finish_multiplier: int,
    config: ScoringConfigLike,
) -> HandScoreBreakdown:
    """Score a non-winning hand without mixing independent components."""

    if player.immediate_penalty < 0:
        raise ValueError("Immediate penalty cannot be negative")
    if finish_multiplier < 1:
        raise ValueError("Finish multiplier must be positive")

    if player.opened_mode is OpenedMode.NONE:
        unopened = _config_int(config, "unopened_end_penalty", 202)
        total = unopened * finish_multiplier + player.immediate_penalty
        return HandScoreBreakdown(
            normal_value_sum=0,
            unopened_penalty=unopened,
            pair_multiplier=1,
            finish_multiplier=finish_multiplier,
            okey_count=0,
            okey_surcharge=0,
            immediate_penalty=player.immediate_penalty,
            total=total,
        )

    normal_value_sum = 0
    okey_count = 0
    for tile in player.hand:
        if is_real_okey(tile, okey_value):
            okey_count += 1
            continue
        value = effective_value(tile, okey_value)
        if value is None:
            raise ValueError(f"Tile {tile.id} has no scoreable value")
        normal_value_sum += value.number

    pair_multiplier = (
        _config_int(config, "pair_remaining_multiplier", 2)
        if player.opened_mode is OpenedMode.PAIRS
        else 1
    )
    multiplied_normal = normal_value_sum * pair_multiplier * finish_multiplier

    surcharge = (
        _config_int(config, "opened_player_okey_in_hand_surcharge", 101)
        if okey_count
        else 0
    )
    if _config_bool(config, "multiply_okey_in_hand_surcharge_by_pair"):
        surcharge *= pair_multiplier
    if _config_bool(config, "multiply_okey_in_hand_surcharge_by_finish"):
        surcharge *= finish_multiplier

    total = multiplied_normal + surcharge + player.immediate_penalty
    return HandScoreBreakdown(
        normal_value_sum=normal_value_sum,
        unopened_penalty=0,
        pair_multiplier=pair_multiplier,
        finish_multiplier=finish_multiplier,
        okey_count=okey_count,
        okey_surcharge=surcharge,
        immediate_penalty=player.immediate_penalty,
        total=total,
    )


def score_round(state: GameState, config: ScoringConfigLike) -> RoundScore:
    """Calculate terminal round totals without mutating ``state``."""

    if not state.terminal or state.terminal_reason is None:
        raise ValueError("Round scoring requires a terminal state and reason")

    reason = state.terminal_reason
    if reason is TerminalReason.ALL_PLAYERS_OPENED_PAIRS:
        return RoundScore(
            reason=reason,
            players=tuple(
                PlayerRoundScore(player_id=player_id, total=0)
                for player_id in range(len(state.players))
            ),
        )

    has_winner = reason is not TerminalReason.STOCK_EXHAUSTED
    if has_winner and state.winner is None:
        raise ValueError(f"{reason.value} requires a winner")
    if not has_winner and state.winner is not None:
        raise ValueError("Stock exhaustion cannot have a winner")

    multiplier = _finish_multiplier(reason, config)
    results: list[PlayerRoundScore] = []
    for player_id, player in enumerate(state.players):
        if player_id == state.winner:
            reward = winner_reward(reason, config)
            total = reward + player.immediate_penalty
            results.append(
                PlayerRoundScore(
                    player_id=player_id,
                    total=total,
                    winner_reward=reward,
                )
            )
            continue

        hand = score_remaining_hand(
            player,
            state.okey_value,
            finish_multiplier=multiplier,
            config=config,
        )
        results.append(
            PlayerRoundScore(player_id=player_id, total=hand.total, hand=hand)
        )

    return RoundScore(reason=reason, players=tuple(results))


def calculate_round_scores(
    state: GameState,
    config: ScoringConfigLike,
) -> tuple[int, ...]:
    """Convenience API returning only per-seat totals."""

    return score_round(state, config).totals
