from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
from random import Random
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

from .actions import Action
from .config import GameConfig
from .joker import okey_value_for_indicator
from .melds import Meld, MeldKind, MeldTile
from .pairs import Pair
from .player import OpenedMode, PlayerState
from .state import (
    AttachmentUsage,
    DrawSource,
    EngineEvent,
    EventType,
    GameState,
    TerminalReason,
    TurnContext,
    TurnPhase,
)
from .table import AttachmentSide, TableMeld, TableState
from .tiles import Color, PhysicalTile, TileKind, TileValue, build_tile_set
from .transition import IllegalAction, apply_action


def _draw_indicator(tiles: list[PhysicalTile]) -> PhysicalTile:
    for index in range(len(tiles) - 1, -1, -1):
        if tiles[index].kind is TileKind.NORMAL:
            return tiles.pop(index)
    raise RuntimeError("A standard tile set has no normal indicator candidate")


def create_round_state(
    *,
    seed: int | None = None,
    config: GameConfig | None = None,
    round_id: int = 1,
    starting_player: int | None = None,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    """Shuffle, choose a normal indicator and deal one deterministic round."""

    rules = config or GameConfig()
    rng = Random(seed)
    tiles = list(build_tile_set())
    rng.shuffle(tiles)
    indicator = _draw_indicator(tiles)

    if starting_player is None:
        starting_player = rng.randrange(rules.player_count)
    if not 0 <= starting_player < rules.player_count:
        raise ValueError("starting_player is outside the configured seat range")

    hands: list[list[PhysicalTile]] = [[] for _ in range(rules.player_count)]
    for _ in range(rules.initial_hand_size):
        for player_id in range(rules.player_count):
            hands[player_id].append(tiles.pop())
    if rules.starting_player_extra_tile:
        hands[starting_player].append(tiles.pop())

    players = tuple(PlayerState(hand=tuple(hand)) for hand in hands)
    opening_context = TurnContext(
        draw_source=DrawSource.DEAL if rules.starting_player_extra_tile else None,
        drawn_tile_id=(
            hands[starting_player][-1].id if rules.starting_player_extra_tile else None
        ),
        opened_mode_at_start=players[starting_player].opened_mode,
    )
    phase = (
        TurnPhase.TABLE_ACTIONS
        if rules.starting_player_extra_tile
        else TurnPhase.DRAW_DECISION
    )
    state = GameState(
        round_id=round_id,
        turn_number=0,
        current_player=starting_player,
        starting_player=starting_player,
        indicator=indicator,
        okey_value=okey_value_for_indicator(indicator),
        stock=tuple(tiles),
        discard_pile=(),
        players=players,
        table=TableState(),
        progressive_series_threshold=rules.opening_min_score,
        progressive_pair_threshold=rules.opening_min_pairs,
        phase=phase,
        turn_context=opening_context,
    )
    event = EngineEvent(
        EventType.DEAL,
        details={
            "round_id": round_id,
            "starting_player": starting_player,
            "indicator_tile_id": indicator.id,
            "hand_sizes": tuple(len(hand) for hand in hands),
            "stock_count": len(tiles),
            "seed": seed,
        },
    )
    return state, (event,)


def _to_primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    return value


def _invalid(path: str, message: str) -> NoReturn:
    raise ValueError(f"Invalid serialized state at {path}: {message}")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid(path, "expected an object")
    return value


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        _invalid(path, "expected an array")
    return list(value)


def _required(data: Mapping[str, object], key: str, path: str) -> object:
    if key not in data:
        _invalid(path, f"missing field {key!r}")
    return data[key]


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(path, "expected an integer")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        _invalid(path, "expected a boolean")
    return value


def _enum(enum_type: type[Enum], value: object, path: str) -> Enum:
    if not isinstance(value, str):
        _invalid(path, "expected an enum string")
    try:
        return enum_type(value)
    except ValueError as exc:
        _invalid(path, str(exc))


def _tile_value(value: object, path: str) -> TileValue:
    data = _mapping(value, path)
    return TileValue(
        color=_enum(
            Color,
            _required(data, "color", path),
            f"{path}.color",
        ),
        number=_integer(
            _required(data, "number", path),
            f"{path}.number",
        ),
    )


def _physical_tile(value: object, path: str) -> PhysicalTile:
    data = _mapping(value, path)
    raw_color = data.get("color")
    raw_number = data.get("number")
    return PhysicalTile(
        id=_integer(_required(data, "id", path), f"{path}.id"),
        kind=_enum(
            TileKind,
            _required(data, "kind", path),
            f"{path}.kind",
        ),
        color=(
            None
            if raw_color is None
            else _enum(Color, raw_color, f"{path}.color")
        ),
        number=None if raw_number is None else _integer(raw_number, f"{path}.number"),
    )


def _meld_tile(value: object, path: str) -> MeldTile:
    data = _mapping(value, path)
    return MeldTile(
        physical_tile=_physical_tile(
            _required(data, "physical_tile", path),
            f"{path}.physical_tile",
        ),
        represented_value=_tile_value(
            _required(data, "represented_value", path),
            f"{path}.represented_value",
        ),
    )


def _meld(value: object, path: str) -> Meld:
    data = _mapping(value, path)
    tiles = tuple(
        _meld_tile(item, f"{path}.tiles[{index}]")
        for index, item in enumerate(
            _sequence(_required(data, "tiles", path), f"{path}.tiles")
        )
    )
    return Meld(
        kind=_enum(
            MeldKind,
            _required(data, "kind", path),
            f"{path}.kind",
        ),
        tiles=tiles,
    )


def _pair(value: object, path: str) -> Pair:
    data = _mapping(value, path)
    items = _sequence(_required(data, "tiles", path), f"{path}.tiles")
    if len(items) != 2:
        _invalid(f"{path}.tiles", "a pair must contain exactly two tiles")
    return Pair(
        (
            _meld_tile(items[0], f"{path}.tiles[0]"),
            _meld_tile(items[1], f"{path}.tiles[1]"),
        )
    )


def _player(value: object, path: str) -> PlayerState:
    data = _mapping(value, path)
    raw_opening_turn = data.get("opening_turn")
    return PlayerState(
        hand=tuple(
            _physical_tile(item, f"{path}.hand[{index}]")
            for index, item in enumerate(
                _sequence(_required(data, "hand", path), f"{path}.hand")
            )
        ),
        opened_mode=_enum(
            OpenedMode,
            _required(data, "opened_mode", path),
            f"{path}.opened_mode",
        ),
        opening_turn=(
            None
            if raw_opening_turn is None
            else _integer(raw_opening_turn, f"{path}.opening_turn")
        ),
        immediate_penalty=_integer(
            _required(data, "immediate_penalty", path),
            f"{path}.immediate_penalty",
        ),
        score=_integer(_required(data, "score", path), f"{path}.score"),
    )


def _table(value: object, path: str) -> TableState:
    data = _mapping(value, path)
    melds = tuple(
        TableMeld(
            id=_integer(
                _required(_mapping(item, item_path), "id", item_path),
                f"{item_path}.id",
            ),
            meld=_meld(
                _required(_mapping(item, item_path), "meld", item_path),
                f"{item_path}.meld",
            ),
        )
        for index, item in enumerate(
            _sequence(_required(data, "melds", path), f"{path}.melds")
        )
        for item_path in (f"{path}.melds[{index}]",)
    )
    pairs = tuple(
        _pair(item, f"{path}.pairs[{index}]")
        for index, item in enumerate(
            _sequence(_required(data, "pairs", path), f"{path}.pairs")
        )
    )
    return TableState(
        melds=melds,
        pairs=pairs,
        next_meld_id=_integer(
            _required(data, "next_meld_id", path),
            f"{path}.next_meld_id",
        ),
    )


def _turn_context(value: object, path: str) -> TurnContext:
    data = _mapping(value, path)
    raw_draw_source = data.get("draw_source")
    raw_drawn_tile_id = data.get("drawn_tile_id")
    raw_taken_tile_id = data.get("taken_discard_tile_id")
    usage = tuple(
        AttachmentUsage(
            meld_id=_integer(
                _required(_mapping(item, item_path), "meld_id", item_path),
                f"{item_path}.meld_id",
            ),
            side=_enum(
                AttachmentSide,
                _required(_mapping(item, item_path), "side", item_path),
                f"{item_path}.side",
            ),
            count=_integer(
                _required(_mapping(item, item_path), "count", item_path),
                f"{item_path}.count",
            ),
        )
        for index, item in enumerate(
            _sequence(
                _required(data, "attachment_usage", path),
                f"{path}.attachment_usage",
            )
        )
        for item_path in (f"{path}.attachment_usage[{index}]",)
    )
    return TurnContext(
        draw_source=(
            None
            if raw_draw_source is None
            else _enum(DrawSource, raw_draw_source, f"{path}.draw_source")
        ),
        drawn_tile_id=(
            None
            if raw_drawn_tile_id is None
            else _integer(raw_drawn_tile_id, f"{path}.drawn_tile_id")
        ),
        taken_discard_tile_id=(
            None
            if raw_taken_tile_id is None
            else _integer(raw_taken_tile_id, f"{path}.taken_discard_tile_id")
        ),
        taken_discard_used=_boolean(
            _required(data, "taken_discard_used", path),
            f"{path}.taken_discard_used",
        ),
        opened_mode_at_start=_enum(
            OpenedMode,
            _required(data, "opened_mode_at_start", path),
            f"{path}.opened_mode_at_start",
        ),
        opened_this_turn=_boolean(
            _required(data, "opened_this_turn", path),
            f"{path}.opened_this_turn",
        ),
        stock_exhausted_after_draw=_boolean(
            _required(data, "stock_exhausted_after_draw", path),
            f"{path}.stock_exhausted_after_draw",
        ),
        attachment_usage=usage,
    )


def deserialize_state(payload: Mapping[str, object]) -> GameState:
    """Rebuild a fully typed ``GameState`` from JSON-compatible state data."""

    data = _mapping(payload, "$")
    raw_terminal_reason = data.get("terminal_reason")
    raw_winner = data.get("winner")
    return GameState(
        round_id=_integer(_required(data, "round_id", "$"), "$.round_id"),
        turn_number=_integer(_required(data, "turn_number", "$"), "$.turn_number"),
        current_player=_integer(
            _required(data, "current_player", "$"),
            "$.current_player",
        ),
        starting_player=_integer(
            _required(data, "starting_player", "$"),
            "$.starting_player",
        ),
        indicator=_physical_tile(
            _required(data, "indicator", "$"),
            "$.indicator",
        ),
        okey_value=_tile_value(
            _required(data, "okey_value", "$"),
            "$.okey_value",
        ),
        stock=tuple(
            _physical_tile(item, f"$.stock[{index}]")
            for index, item in enumerate(
                _sequence(_required(data, "stock", "$"), "$.stock")
            )
        ),
        discard_pile=tuple(
            _physical_tile(item, f"$.discard_pile[{index}]")
            for index, item in enumerate(
                _sequence(
                    _required(data, "discard_pile", "$"),
                    "$.discard_pile",
                )
            )
        ),
        players=tuple(
            _player(item, f"$.players[{index}]")
            for index, item in enumerate(
                _sequence(_required(data, "players", "$"), "$.players")
            )
        ),
        table=_table(_required(data, "table", "$"), "$.table"),
        progressive_series_threshold=_integer(
            _required(data, "progressive_series_threshold", "$"),
            "$.progressive_series_threshold",
        ),
        progressive_pair_threshold=_integer(
            _required(data, "progressive_pair_threshold", "$"),
            "$.progressive_pair_threshold",
        ),
        phase=_enum(
            TurnPhase,
            _required(data, "phase", "$"),
            "$.phase",
        ),
        terminal=_boolean(_required(data, "terminal", "$"), "$.terminal"),
        terminal_reason=(
            None
            if raw_terminal_reason is None
            else _enum(TerminalReason, raw_terminal_reason, "$.terminal_reason")
        ),
        winner=None if raw_winner is None else _integer(raw_winner, "$.winner"),
        turn_context=_turn_context(
            _required(data, "turn_context", "$"),
            "$.turn_context",
        ),
    )


class ReplayError(RuntimeError):
    def __init__(self, action_index: int, action: Action, reason: str) -> None:
        self.action_index = action_index
        self.action = action
        super().__init__(
            f"Replay failed at action {action_index} "
            f"({type(action).__name__}): {reason}"
        )


class RoundEngine:
    """Small stateful facade over the pure round transition function."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.state: GameState | None = None
        self.seed: int | None = None
        self.action_history: list[Action] = []
        self.event_log: list[EngineEvent] = []

    def reset(
        self,
        seed: int | None = None,
        *,
        round_id: int = 1,
        starting_player: int | None = None,
    ) -> GameState:
        self.seed = seed
        self.action_history.clear()
        self.state, events = create_round_state(
            seed=seed,
            config=self.config,
            round_id=round_id,
            starting_player=starting_player,
        )
        self.event_log = list(events)
        return self.state

    def step(self, action: Action) -> tuple[GameState, tuple[EngineEvent, ...]]:
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        state, events = apply_action(self.state, action, self.config)
        self.state = state
        self.action_history.append(action)
        self.event_log.extend(events)
        return state, events

    def is_terminal(self) -> bool:
        return bool(self.state and self.state.terminal)

    def clone_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("reset() must be called before clone_state()")
        return deepcopy(self.state)

    def get_legal_actions(self, player_id: int | None = None) -> tuple[Action, ...]:
        if self.state is None:
            raise RuntimeError("reset() must be called before get_legal_actions()")
        requested_player = self.state.current_player if player_id is None else player_id
        if requested_player != self.state.current_player:
            raise ValueError("Legal actions are available only for the current player")
        from .legal_actions import get_legal_actions

        return get_legal_actions(self.state, self.config)

    def get_observation(self, player_id: int):
        if self.state is None:
            raise RuntimeError("reset() must be called before get_observation()")
        from okey101.rl.observation import get_observation

        return get_observation(self.state, player_id)

    def serialize_state(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("reset() must be called before serialize_state()")
        return _to_primitive(self.state)

    def load_state(self, payload: Mapping[str, object]) -> GameState:
        """Replace the active state from a JSON-compatible serialized payload."""

        state = deserialize_state(payload)
        from .invariants import validate_invariants

        validate_invariants(state)
        self.state = state
        self.seed = None
        self.action_history.clear()
        self.event_log.clear()
        return state

    def get_scores(self) -> tuple[int, ...]:
        if self.state is None:
            raise RuntimeError("reset() must be called before get_scores()")
        from .scoring import calculate_round_scores

        return calculate_round_scores(self.state, self.config.scoring)


def replay_from_seed_and_actions(
    seed: int | None,
    actions: Iterable[Action],
    *,
    config: GameConfig | None = None,
    round_id: int = 1,
    starting_player: int | None = None,
) -> RoundEngine:
    """Replay typed actions from a deterministic deal and return the engine."""

    engine = RoundEngine(config)
    engine.reset(
        seed=seed,
        round_id=round_id,
        starting_player=starting_player,
    )
    for index, action in enumerate(actions):
        try:
            engine.step(action)
        except (IllegalAction, TypeError, ValueError) as exc:
            raise ReplayError(index, action, str(exc)) from exc
    return engine
