"""Debug-time state invariants with reproducible, location-rich failures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .melds import validate_meld
from .pairs import validate_pair
from .player import OpenedMode
from .state import GameState, TerminalReason, TurnPhase
from .tiles import PhysicalTile, build_tile_set


@dataclass(frozen=True, slots=True)
class InvariantViolation:
    code: str
    message: str


class InvariantError(AssertionError):
    def __init__(self, violations: Iterable[InvariantViolation]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(
            f"{violation.code}: {violation.message}"
            for violation in self.violations
        )
        super().__init__(message)


def _pair_tiles(pair: object) -> tuple[PhysicalTile, ...]:
    tiles = getattr(pair, "tiles", None)
    if tiles is not None:
        return tuple(
            getattr(tile, "physical_tile", tile)
            for tile in tiles
        )
    return tuple(
        tile
        for name in ("first", "second", "left", "right")
        if (tile := getattr(pair, name, None)) is not None
    )


def physical_tile_locations(
    state: GameState,
) -> tuple[tuple[str, PhysicalTile], ...]:
    """Flatten every owning state location into ``(location, tile)`` pairs."""

    located: list[tuple[str, PhysicalTile]] = [("indicator", state.indicator)]
    located.extend(
        (f"stock[{index}]", tile)
        for index, tile in enumerate(state.stock)
    )
    located.extend(
        (f"discard_pile[{index}]", tile)
        for index, tile in enumerate(state.discard_pile)
    )
    for player_id, player in enumerate(state.players):
        located.extend(
            (f"players[{player_id}].hand[{index}]", tile)
            for index, tile in enumerate(player.hand)
        )
    for table_meld in state.table.melds:
        located.extend(
            (
                f"table.melds[{table_meld.id}].tiles[{index}]",
                meld_tile.physical_tile,
            )
            for index, meld_tile in enumerate(table_meld.meld.tiles)
        )
    for pair_index, pair in enumerate(state.table.pairs):
        located.extend(
            (f"table.pairs[{pair_index}][{tile_index}]", tile)
            for tile_index, tile in enumerate(_pair_tiles(pair))
        )
    return tuple(located)


def find_invariant_violations(
    state: GameState,
    *,
    expected_tile_ids: Iterable[int] = range(106),
) -> tuple[InvariantViolation, ...]:
    """Return all core conservation and terminal-state violations."""

    violations: list[InvariantViolation] = []
    locations = physical_tile_locations(state)
    by_id: dict[int, list[str]] = defaultdict(list)
    for location, tile in locations:
        by_id[tile.id].append(location)

    duplicate_ids = {
        tile_id: owning_locations
        for tile_id, owning_locations in by_id.items()
        if len(owning_locations) > 1
    }
    if duplicate_ids:
        details = ", ".join(
            f"{tile_id} at {owning_locations}"
            for tile_id, owning_locations in sorted(duplicate_ids.items())
        )
        violations.append(
            InvariantViolation("DUPLICATE_TILE_ID", details)
        )

    expected = set(expected_tile_ids)
    actual = set(by_id)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        violations.append(
            InvariantViolation("MISSING_TILE_IDS", str(missing))
        )
    if unexpected:
        violations.append(
            InvariantViolation("UNEXPECTED_TILE_IDS", str(unexpected))
        )
    if len(locations) != len(expected):
        violations.append(
            InvariantViolation(
                "PHYSICAL_TILE_COUNT",
                f"expected {len(expected)}, found {len(locations)}",
            )
        )

    if expected == set(range(106)):
        canonical_by_id = {tile.id: tile for tile in build_tile_set()}
        for location, tile in locations:
            canonical = canonical_by_id.get(tile.id)
            if canonical is not None and tile != canonical:
                violations.append(
                    InvariantViolation(
                        "PHYSICAL_TILE_IDENTITY",
                        (
                            f"{location} has {tile!r}, expected "
                            f"{canonical!r} for id {tile.id}"
                        ),
                    )
                )

    physical_by_id = {
        tile_id: next(tile for _location, tile in locations if tile.id == tile_id)
        for tile_id in by_id
    }
    for index, record in enumerate(state.discard_history):
        location = f"discard_history[{index}]"
        current_tile = physical_by_id.get(record.tile.id)
        if current_tile is None:
            violations.append(
                InvariantViolation(
                    "UNKNOWN_DISCARD_HISTORY_TILE",
                    f"{location} references tile id {record.tile.id}",
                )
            )
        elif current_tile != record.tile:
            violations.append(
                InvariantViolation(
                    "DISCARD_HISTORY_TILE_MISMATCH",
                    f"{location} does not match physical tile {record.tile.id}",
                )
            )
        for field_name, player_id in (
            ("player_id", record.player_id),
            ("taken_by", record.taken_by),
        ):
            if player_id is not None and not 0 <= player_id < len(state.players):
                violations.append(
                    InvariantViolation(
                        "INVALID_DISCARD_HISTORY_PLAYER",
                        f"{location}.{field_name}={player_id}",
                    )
                )
        if record.turn_number > state.turn_number:
            violations.append(
                InvariantViolation(
                    "FUTURE_DISCARD_HISTORY_TURN",
                    f"{location}.turn_number={record.turn_number}",
                )
            )

    if not state.players:
        violations.append(
            InvariantViolation("NO_PLAYERS", "state must contain players")
        )
    elif not 0 <= state.current_player < len(state.players):
        violations.append(
            InvariantViolation(
                "INVALID_CURRENT_PLAYER",
                f"{state.current_player} outside 0..{len(state.players) - 1}",
            )
        )

    if state.terminal:
        if state.phase is not TurnPhase.TERMINAL:
            violations.append(
                InvariantViolation(
                    "TERMINAL_PHASE",
                    f"terminal state has phase {state.phase.value}",
                )
            )
        if state.terminal_reason is None:
            violations.append(
                InvariantViolation(
                    "MISSING_TERMINAL_REASON",
                    "terminal state requires a reason",
                )
            )
    else:
        if state.phase is TurnPhase.TERMINAL:
            violations.append(
                InvariantViolation(
                    "NONTERMINAL_PHASE",
                    "non-terminal state cannot use TERMINAL phase",
                )
            )
        if state.terminal_reason is not None:
            violations.append(
                InvariantViolation(
                    "EARLY_TERMINAL_REASON",
                    "non-terminal state cannot have a terminal reason",
                )
            )
        if state.winner is not None:
            violations.append(
                InvariantViolation(
                    "EARLY_WINNER",
                    "non-terminal state cannot have a winner",
                )
            )

    no_winner_reasons = {
        TerminalReason.STOCK_EXHAUSTED,
        TerminalReason.ALL_PLAYERS_OPENED_PAIRS,
    }
    if state.terminal_reason in no_winner_reasons and state.winner is not None:
        violations.append(
            InvariantViolation(
                "UNEXPECTED_WINNER",
                f"{state.terminal_reason.value} cannot have a winner",
            )
        )
    if (
        state.terminal
        and state.terminal_reason not in no_winner_reasons
        and state.winner is None
    ):
        violations.append(
            InvariantViolation(
                "MISSING_WINNER",
                "finish terminal state requires a winner",
            )
        )
    if state.winner is not None and not 0 <= state.winner < len(state.players):
        violations.append(
            InvariantViolation(
                "INVALID_WINNER",
                f"{state.winner} outside player range",
            )
        )

    if state.terminal_reason is TerminalReason.ALL_PLAYERS_OPENED_PAIRS:
        if any(
            player.opened_mode is not OpenedMode.PAIRS
            for player in state.players
        ):
            violations.append(
                InvariantViolation(
                    "ALL_PAIRS_REASON_MISMATCH",
                    "not every player opened pairs",
                )
            )

    for table_meld in state.table.melds:
        if not validate_meld(table_meld.meld, state.okey_value):
            violations.append(
                InvariantViolation(
                    "INVALID_TABLE_MELD",
                    f"table meld {table_meld.id} is not legal",
                )
            )
    for pair_index, pair in enumerate(state.table.pairs):
        if not validate_pair(pair, state.okey_value):
            violations.append(
                InvariantViolation(
                    "INVALID_TABLE_PAIR",
                    f"table pair {pair_index} is not legal",
                )
            )

    return tuple(violations)


def validate_invariants(
    state: GameState,
    *,
    expected_tile_ids: Iterable[int] = range(106),
) -> None:
    """Raise ``InvariantError`` when any core state invariant fails."""

    violations = find_invariant_violations(
        state,
        expected_tile_ids=expected_tile_ids,
    )
    if violations:
        raise InvariantError(violations)
