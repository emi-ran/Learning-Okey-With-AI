"""Deterministic spectator replay recording without raw state consumers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

from okey101.agents.random_agent import RandomAgent
from okey101.engine.actions import Action
from okey101.engine.config import GameConfig, ScoringConfig
from okey101.engine.joker import effective_value, is_real_okey
from okey101.engine.round import RoundEngine, deserialize_action, serialize_action
from okey101.engine.scoring import score_round
from okey101.engine.state import EngineEvent, EventType, GameState
from okey101.engine.tiles import PhysicalTile, TileKind, TileValue
from okey101.engine.transition import IllegalAction
from okey101.rl.observation import PlayerObservation

from .schema import REPLAY_SCHEMA_VERSION, content_digest, validate_replay_document


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    id: str
    label: str
    training_step: int

    def __post_init__(self) -> None:
        if not self.id or not self.label:
            raise ValueError("checkpoint id and label must not be empty")
        if self.training_step < 0:
            raise ValueError("training_step cannot be negative")


@dataclass(frozen=True, slots=True)
class SelectionContext:
    seed: int
    action_index: int
    player_id: int
    checkpoint: CheckpointMetadata


@dataclass(frozen=True, slots=True)
class CandidateProbability:
    action: Action
    probability: float


@dataclass(frozen=True, slots=True)
class ActionSelection:
    action: Action
    selected_probability: float | None = None
    value: float | None = None
    candidates: tuple[CandidateProbability, ...] = ()


ActionSelector = Callable[
    [PlayerObservation, Sequence[Action], SelectionContext],
    Action | ActionSelection,
]


def _primitive(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def _tile_value(value: TileValue) -> dict[str, object]:
    return {"color": value.color.value, "number": value.number}


def _tile(tile: PhysicalTile, state: GameState) -> dict[str, object]:
    display = effective_value(tile, state.okey_value)
    if display is None:
        display = state.okey_value
    return {
        "id": tile.id,
        "kind": tile.kind.value,
        "physical": (
            None
            if tile.kind is TileKind.FAKE_OKEY
            else {"color": tile.color.value, "number": tile.number}
        ),
        "display": _tile_value(display),
        "is_real_okey": is_real_okey(tile, state.okey_value),
        "is_fake_okey": tile.kind is TileKind.FAKE_OKEY,
        "is_indicator": tile.id == state.indicator.id,
    }


def _meld_tile(value: object, state: GameState) -> dict[str, object]:
    physical = value.physical_tile
    return {
        "tile": _tile(physical, state),
        "represented_value": _tile_value(value.represented_value),
        "uses_joker_assignment": effective_value(physical, state.okey_value) is None,
    }


def _view(state: GameState) -> dict[str, object]:
    return {
        "round_id": state.round_id,
        "turn_number": state.turn_number,
        "current_player": state.current_player,
        "starting_player": state.starting_player,
        "phase": state.phase.value,
        "indicator": _tile(state.indicator, state),
        "okey_value": _tile_value(state.okey_value),
        "stock_count": state.stock_count,
        "discard_top": (
            None if state.discard_top is None else _tile(state.discard_top, state)
        ),
        "discard_pile": [_tile(tile, state) for tile in state.discard_pile],
        "discard_history": [
            {
                "tile": _tile(record.tile, state),
                "player_id": record.player_id,
                "turn_number": record.turn_number,
                "taken_by": record.taken_by,
            }
            for record in state.discard_history
        ],
        "players": [
            {
                "seat": seat,
                "label": f"Oyuncu {seat + 1}",
                "hand": [_tile(tile, state) for tile in player.hand],
                "hand_count": len(player.hand),
                "opened_mode": player.opened_mode.value,
                "opening_turn": player.opening_turn,
                "immediate_penalty": player.immediate_penalty,
                "cumulative_score": player.score,
            }
            for seat, player in enumerate(state.players)
        ],
        "table": {
            "melds": [
                {
                    "id": table_meld.id,
                    "kind": table_meld.meld.kind.value,
                    "score": table_meld.meld.score,
                    "tiles": [
                        _meld_tile(tile, state) for tile in table_meld.meld.tiles
                    ],
                }
                for table_meld in state.table.melds
            ],
            "pairs": [
                {
                    "tiles": [_meld_tile(tile, state) for tile in pair.tiles],
                }
                for pair in state.table.pairs
            ],
        },
        "thresholds": {
            "series": state.progressive_series_threshold,
            "pairs": state.progressive_pair_threshold,
        },
        "turn_context": {
            "draw_source": (
                None
                if state.turn_context.draw_source is None
                else state.turn_context.draw_source.value
            ),
            "drawn_tile_id": state.turn_context.drawn_tile_id,
            "taken_discard_tile_id": state.turn_context.taken_discard_tile_id,
            "taken_discard_used": state.turn_context.taken_discard_used,
            "opened_this_turn": state.turn_context.opened_this_turn,
            "stock_exhausted_after_draw": (
                state.turn_context.stock_exhausted_after_draw
            ),
        },
    }


_EVENT_TEXT = {
    EventType.DEAL: "Taşlar dağıtıldı.",
    EventType.DRAW_STOCK: "Oyuncu {player} ortadan taş çekti.",
    EventType.TAKE_DISCARD: "Oyuncu {player} önceki atılan taşı aldı.",
    EventType.OPEN_SERIES: "Oyuncu {player} seri açtı.",
    EventType.LAY_MELDS: "Oyuncu {player} yeni perler açtı.",
    EventType.OPEN_PAIRS: "Oyuncu {player} çift açtı.",
    EventType.ADD_TO_MELD: "Oyuncu {player} masadaki pere taş işledi.",
    EventType.ADD_PAIR: "Oyuncu {player} çift alanına çift ekledi.",
    EventType.REPLACE_JOKER: "Oyuncu {player} Okey'i geri aldı.",
    EventType.END_TABLE_ACTIONS: "Oyuncu {player} masa hamlelerini bitirdi.",
    EventType.DISCARD: "Oyuncu {player} taş attı.",
    EventType.PENALTY: "Oyuncu {player} ceza aldı.",
    EventType.FINISH: "Oyuncu {player} eli bitirdi.",
    EventType.ROUND_END: "El sona erdi.",
}


def _event(event: EngineEvent) -> dict[str, object]:
    player = "?" if event.player_id is None else str(event.player_id + 1)
    return {
        "type": event.type.value,
        "player_id": event.player_id,
        "details": _primitive(event.details),
        "narration": _EVENT_TEXT[event.type].format(player=player),
    }


def _action_narration(action: Action | None, actor_seat: int | None) -> str:
    if action is None:
        return "Başlangıç eli dağıtıldı."
    player = "?" if actor_seat is None else str(actor_seat + 1)
    labels = {
        "draw_from_stock": "ortadan taş çekiyor",
        "take_previous_discard": "önceki atılan taşı alıyor",
        "open_melds": "seri açıyor",
        "open_pairs": "çift açıyor",
        "add_to_meld": "masadaki pere taş işliyor",
        "add_pair": "çift alanına çift ekliyor",
        "replace_joker": "Okey'i geri alıyor",
        "end_table_actions": "masa hamlelerini tamamlıyor",
        "discard": "taş atıyor",
    }
    return f"Oyuncu {player} {labels[action.type.value]}."


def _scores(state: GameState, config: GameConfig) -> dict[str, object]:
    final = score_round(state, config) if state.terminal else None
    return {
        "immediate_penalties": [
            player.immediate_penalty for player in state.players
        ],
        "cumulative_scores": [player.score for player in state.players],
        "final_totals": None if final is None else list(final.totals),
        "breakdown": None if final is None else _primitive(final),
    }


def _terminal(state: GameState) -> dict[str, object]:
    return {
        "is_terminal": state.terminal,
        "reason": (
            None if state.terminal_reason is None else state.terminal_reason.value
        ),
        "winner": state.winner,
    }


def _policy_step(selection: ActionSelection | None) -> dict[str, object] | None:
    if selection is None:
        return None
    for probability in (
        [selection.selected_probability]
        if selection.selected_probability is not None
        else []
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("selected_probability must be between 0 and 1")
    candidates = []
    for candidate in selection.candidates:
        if not 0.0 <= candidate.probability <= 1.0:
            raise ValueError("candidate probability must be between 0 and 1")
        candidates.append(
            {
                "action": serialize_action(candidate.action),
                "probability": candidate.probability,
            }
        )
    return {
        "selected_probability": selection.selected_probability,
        "value": selection.value,
        "candidates": candidates,
    }


def _frame(
    *,
    state: GameState,
    config: GameConfig,
    frame_index: int,
    action: Action | None,
    actor_seat: int | None,
    events: Sequence[EngineEvent],
    selection: ActionSelection | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "frame_index": frame_index,
        "action_index": None if action is None else frame_index - 1,
        "actor_seat": actor_seat,
        "action": None if action is None else serialize_action(action),
        "narration": _action_narration(action, actor_seat),
        "events": [_event(event) for event in events],
        "view": _view(state),
        "scores": _scores(state, config),
        "terminal": _terminal(state),
        "policy_step": _policy_step(selection),
    }
    payload["state_digest"] = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key not in {"narration", "policy_step"}
        }
    )
    return payload


def _config_from_payload(payload: Mapping[str, object]) -> GameConfig:
    scoring_raw = payload.get("scoring")
    if not isinstance(scoring_raw, Mapping):
        raise ValueError("replay config.scoring must be an object")
    scoring_names = {field.name for field in fields(ScoringConfig)}
    game_names = {field.name for field in fields(GameConfig)} - {"scoring"}
    scoring = ScoringConfig(
        **{name: scoring_raw[name] for name in scoring_names if name in scoring_raw}
    )
    return GameConfig(
        **{name: payload[name] for name in game_names if name in payload},
        scoring=scoring,
    )


def _selection_from_result(result: Action | ActionSelection) -> ActionSelection:
    return result if isinstance(result, ActionSelection) else ActionSelection(result)


def record_episode(
    *,
    seed: int,
    selector: ActionSelector,
    checkpoint: CheckpointMetadata,
    policy_name: str,
    policy_version: str | None = None,
    config: GameConfig | None = None,
    round_id: int = 1,
    starting_player: int | None = None,
    spectator_mode: bool = False,
    max_actions: int = 10_000,
) -> dict[str, Any]:
    """Record one complete deterministic episode as renderer-ready JSON."""

    if not spectator_mode:
        raise ValueError("spectator_mode=True must be explicitly enabled")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if max_actions < 1:
        raise ValueError("max_actions must be positive")

    rules = config or GameConfig()
    engine = RoundEngine(rules)
    initial = engine.reset(
        seed=seed,
        round_id=round_id,
        starting_player=starting_player,
    )
    frames_payload = [
        _frame(
            state=initial,
            config=rules,
            frame_index=0,
            action=None,
            actor_seat=None,
            events=tuple(engine.event_log),
        )
    ]

    while not engine.is_terminal():
        action_index = len(engine.action_history)
        assert engine.state is not None
        actor = engine.state.current_player
        legal = engine.get_legal_actions()
        observation = engine.get_observation(actor)
        selection = _selection_from_result(
            selector(
                observation,
                legal,
                SelectionContext(seed, action_index, actor, checkpoint),
            )
        )
        if selection.action not in legal:
            raise ValueError(
                f"selector returned an illegal action at action {action_index}"
            )
        state, events = engine.step(selection.action)
        frames_payload.append(
            _frame(
                state=state,
                config=rules,
                frame_index=len(frames_payload),
                action=selection.action,
                actor_seat=actor,
                events=events,
                selection=selection,
            )
        )
        if len(engine.action_history) >= max_actions and not engine.is_terminal():
            raise RuntimeError(f"episode seed {seed} exceeded {max_actions} actions")

    assert engine.state is not None
    actual_starting_player = initial.starting_player
    final_scores = engine.get_scores()
    document: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "kind": "episode_replay",
        "visibility": {"mode": "spectator", "reveal_all_hands": True},
        "episode": {
            "seed": seed,
            "round_id": round_id,
            "starting_player": actual_starting_player,
        },
        "checkpoint": asdict(checkpoint),
        "policy": {"name": policy_name, "version": policy_version},
        "config": _primitive(rules),
        "frames": frames_payload,
        "summary": {
            "action_count": len(engine.action_history),
            "frame_count": len(frames_payload),
            "terminal_reason": engine.state.terminal_reason.value,
            "winner": engine.state.winner,
            "final_scores": list(final_scores),
            "total_immediate_penalty": sum(
                player.immediate_penalty for player in engine.state.players
            ),
        },
    }
    document["integrity"] = {
        "algorithm": "sha256",
        "document_digest": content_digest(document),
    }
    return validate_replay_document(document)


def record_random_episode(
    *,
    seed: int,
    checkpoint: CheckpointMetadata | None = None,
    config: GameConfig | None = None,
    spectator_mode: bool = False,
    max_actions: int = 10_000,
) -> dict[str, Any]:
    metadata = checkpoint or CheckpointMetadata("random", "Random başlangıç", 0)
    player_count = (config or GameConfig()).player_count
    agents: tuple[RandomAgent[PlayerObservation, Action], ...] = tuple(
        RandomAgent(seed=seed * player_count + seat)
        for seat in range(player_count)
    )

    def select(
        observation: PlayerObservation,
        legal_actions: Sequence[Action],
        context: SelectionContext,
    ) -> Action:
        return agents[context.player_id].select_action(observation, legal_actions)

    return record_episode(
        seed=seed,
        selector=select,
        checkpoint=metadata,
        policy_name="RandomAgent",
        policy_version="uniform-v1",
        config=config,
        spectator_mode=spectator_mode,
        max_actions=max_actions,
    )


class ReplayVerificationError(ValueError):
    """Raised when deterministic engine replay differs from recorded frames."""


def verify_replay(document: Mapping[str, object]) -> None:
    """Replay all actions and compare every engine-derived frame exactly."""

    replay = validate_replay_document(document)
    episode = replay["episode"]
    assert isinstance(episode, Mapping)
    config_payload = replay["config"]
    assert isinstance(config_payload, Mapping)
    config = _config_from_payload(config_payload)
    engine = RoundEngine(config)
    initial = engine.reset(
        seed=int(episode["seed"]),
        round_id=int(episode["round_id"]),
        starting_player=int(episode["starting_player"]),
    )
    frames_payload = replay["frames"]
    assert isinstance(frames_payload, list)

    expected_initial = _frame(
        state=initial,
        config=config,
        frame_index=0,
        action=None,
        actor_seat=None,
        events=tuple(engine.event_log),
    )
    _compare_engine_frame(frames_payload[0], expected_initial, 0)

    for frame_index, recorded in enumerate(frames_payload[1:], start=1):
        if not isinstance(recorded, Mapping):
            raise ReplayVerificationError(f"frame {frame_index} is not an object")
        action_payload = recorded["action"]
        if not isinstance(action_payload, Mapping):
            raise ReplayVerificationError(f"frame {frame_index} action is missing")
        try:
            action = deserialize_action(action_payload)
            assert engine.state is not None
            actor = engine.state.current_player
            state, events = engine.step(action)
        except (IllegalAction, TypeError, ValueError) as exc:
            raise ReplayVerificationError(
                f"replay failed at frame {frame_index}: {exc}"
            ) from exc
        expected = _frame(
            state=state,
            config=config,
            frame_index=frame_index,
            action=action,
            actor_seat=actor,
            events=events,
        )
        _compare_engine_frame(recorded, expected, frame_index)

    if not engine.is_terminal():
        raise ReplayVerificationError("replay action sequence is not terminal")
    assert engine.state is not None
    expected_summary = {
        "action_count": len(engine.action_history),
        "frame_count": len(frames_payload),
        "terminal_reason": engine.state.terminal_reason.value,
        "winner": engine.state.winner,
        "final_scores": list(engine.get_scores()),
        "total_immediate_penalty": sum(
            player.immediate_penalty for player in engine.state.players
        ),
    }
    if replay["summary"] != expected_summary:
        raise ReplayVerificationError("deterministic mismatch at summary")


def _compare_engine_frame(
    recorded: Mapping[str, object],
    expected: Mapping[str, object],
    frame_index: int,
) -> None:
    keys = (
        "frame_index",
        "action_index",
        "actor_seat",
        "action",
        "narration",
        "events",
        "view",
        "scores",
        "terminal",
        "state_digest",
    )
    for key in keys:
        if recorded.get(key) != expected.get(key):
            raise ReplayVerificationError(
                f"deterministic mismatch at frame {frame_index}.{key}"
            )
