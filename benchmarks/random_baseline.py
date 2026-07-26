"""Deterministic RandomAgent evaluation through the policy-safe RL API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from okey101.agents.random_agent import RandomAgent
from okey101.engine.actions import Action, OpenMelds, OpenPairs
from okey101.engine.player import OpenedMode
from okey101.engine.state import TerminalReason
from okey101.rl.env import Decision, SingleRoundEnv
from okey101.rl.observation import PlayerObservation

_NO_WINNER_REASONS = {
    TerminalReason.STOCK_EXHAUSTED,
    TerminalReason.ALL_PLAYERS_OPENED_PAIRS,
}
_PENALTY_METRIC_NOTE = (
    "penalties_per_game unavailable: a terminal stock-exhaustion discard may "
    "add a penalty after the last policy-safe observation"
)


@dataclass(frozen=True, slots=True)
class RandomBaselineResult:
    episodes: int
    start_seed: int
    end_seed_exclusive: int
    actions: int
    elapsed_seconds: float
    episodes_per_second: float
    actions_per_second: float
    terminal_reason_counts: dict[str, int]
    seat_score_means: tuple[float, ...]
    overall_finish_rate: float
    seat_finish_rates: tuple[float, ...]
    overall_unopened_rate: float
    seat_unopened_rates: tuple[float, ...]
    penalties_per_game: float | None
    unavailable_metrics: tuple[str, ...]


def _absolute_opened_modes(
    decision: Decision,
    final_action: Action,
) -> tuple[OpenedMode, ...]:
    """Recover terminal public opening modes without accessing engine state."""

    player_count = len(decision.observation.player_statuses)
    modes: list[OpenedMode | None] = [None] * player_count
    for status in decision.observation.player_statuses:
        absolute_seat = (
            decision.seat + status.relative_seat
        ) % player_count
        modes[absolute_seat] = status.opened_mode

    # Four-pair voiding can terminate directly on the opening action, before
    # another public observation exists. The selected action itself is public.
    if isinstance(final_action, OpenPairs):
        modes[decision.seat] = OpenedMode.PAIRS
    elif isinstance(final_action, OpenMelds):
        modes[decision.seat] = OpenedMode.SERIES

    if any(mode is None for mode in modes):
        raise RuntimeError("Public player statuses did not cover every seat")
    return tuple(mode for mode in modes if mode is not None)


def run_random_baseline(
    *,
    episodes: int,
    start_seed: int = 0,
    max_actions_per_episode: int = 10_000,
) -> RandomBaselineResult:
    """Evaluate deterministic uniform-random policies over a seed range."""

    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if isinstance(start_seed, bool) or not isinstance(start_seed, int):
        raise TypeError("start_seed must be an integer")
    if (
        isinstance(max_actions_per_episode, bool)
        or not isinstance(max_actions_per_episode, int)
    ):
        raise TypeError("max_actions_per_episode must be an integer")
    if max_actions_per_episode < 1:
        raise ValueError("max_actions_per_episode must be positive")

    env = SingleRoundEnv()
    player_count = env.num_players
    score_totals = [0] * player_count
    finish_counts = [0] * player_count
    unopened_counts = [0] * player_count
    terminal_reason_counts: dict[str, int] = {}
    action_count = 0
    finished_episodes = 0
    started = perf_counter()

    for episode_seed in range(start_seed, start_seed + episodes):
        decision = env.reset(seed=episode_seed)
        agents: tuple[RandomAgent[PlayerObservation, Action], ...] = tuple(
            RandomAgent(seed=episode_seed * player_count + seat)
            for seat in range(player_count)
        )

        for _episode_step in range(max_actions_per_episode):
            action = agents[decision.seat].select_action(
                decision.observation,
                decision.legal_actions,
            )
            result = env.step(action)
            action_count += 1
            if not result.terminated:
                assert result.next_decision is not None
                decision = result.next_decision
                continue

            if result.terminal_reason is None or result.final_scores is None:
                raise RuntimeError("Terminal environment result is incomplete")
            reason = result.terminal_reason
            reason_name = reason.value
            terminal_reason_counts[reason_name] = (
                terminal_reason_counts.get(reason_name, 0) + 1
            )
            for seat, score in enumerate(result.final_scores):
                score_totals[seat] += score

            if reason not in _NO_WINNER_REASONS:
                finish_counts[result.acting_seat] += 1
                finished_episodes += 1

            opened_modes = _absolute_opened_modes(decision, action)
            for seat, mode in enumerate(opened_modes):
                if mode is OpenedMode.NONE:
                    unopened_counts[seat] += 1
            break
        else:
            raise RuntimeError(
                f"episode seed {episode_seed} exceeded "
                f"{max_actions_per_episode} actions"
            )

    elapsed = perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("benchmark timer did not advance")
    seat_denominator = episodes * player_count
    return RandomBaselineResult(
        episodes=episodes,
        start_seed=start_seed,
        end_seed_exclusive=start_seed + episodes,
        actions=action_count,
        elapsed_seconds=elapsed,
        episodes_per_second=episodes / elapsed,
        actions_per_second=action_count / elapsed,
        terminal_reason_counts=terminal_reason_counts,
        seat_score_means=tuple(total / episodes for total in score_totals),
        overall_finish_rate=finished_episodes / episodes,
        seat_finish_rates=tuple(count / episodes for count in finish_counts),
        overall_unopened_rate=sum(unopened_counts) / seat_denominator,
        seat_unopened_rates=tuple(
            count / episodes for count in unopened_counts
        ),
        penalties_per_game=None,
        unavailable_metrics=(_PENALTY_METRIC_NOTE,),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-actions-per-episode", type=int, default=10_000)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    result = run_random_baseline(
        episodes=args.episodes,
        start_seed=args.start_seed,
        max_actions_per_episode=args.max_actions_per_episode,
    )
    if args.as_json:
        print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
        return

    print(f"episodes: {result.episodes}")
    print(f"seed_range: [{result.start_seed}, {result.end_seed_exclusive})")
    print(f"actions: {result.actions}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"episodes_per_second: {result.episodes_per_second:.3f}")
    print(f"actions_per_second: {result.actions_per_second:.3f}")
    print(f"terminal_reason_counts: {result.terminal_reason_counts}")
    print(f"seat_score_means: {result.seat_score_means}")
    print(f"overall_finish_rate: {result.overall_finish_rate:.6f}")
    print(f"seat_finish_rates: {result.seat_finish_rates}")
    print(f"overall_unopened_rate: {result.overall_unopened_rate:.6f}")
    print(f"seat_unopened_rates: {result.seat_unopened_rates}")
    for note in result.unavailable_metrics:
        print(f"unavailable_metric: {note}")


if __name__ == "__main__":
    main()
