from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from .player import OpenedMode, PlayerState
from .table import AttachmentSide, TableState
from .tiles import PhysicalTile, TileValue


class TurnPhase(str, Enum):
    DRAW_DECISION = "draw_decision"
    TABLE_ACTIONS = "table_actions"
    DISCARD = "discard"
    TERMINAL = "terminal"


class DrawSource(str, Enum):
    DEAL = "deal"
    STOCK = "stock"
    PREVIOUS_DISCARD = "previous_discard"


class TerminalReason(str, Enum):
    NORMAL_FINISH = "normal_finish"
    SAME_TURN_OPEN_FINISH = "same_turn_open_finish"
    SAME_TURN_OPEN_OKEY_FINISH = "same_turn_open_okey_finish"
    ELDEN_FINISH = "elden_finish"
    OKEY_FINISH = "okey_finish"
    ELDEN_OKEY_FINISH = "elden_okey_finish"
    PAIR_FINISH = "pair_finish"
    PAIR_OKEY_FINISH = "pair_okey_finish"
    STOCK_EXHAUSTED = "stock_exhausted"
    ALL_PLAYERS_OPENED_PAIRS = "all_players_opened_pairs"


class EventType(str, Enum):
    DEAL = "deal"
    DRAW_STOCK = "draw_stock"
    TAKE_DISCARD = "take_discard"
    OPEN_SERIES = "open_series"
    LAY_MELDS = "lay_melds"
    OPEN_PAIRS = "open_pairs"
    ADD_TO_MELD = "add_to_meld"
    ADD_PAIR = "add_pair"
    REPLACE_JOKER = "replace_joker"
    END_TABLE_ACTIONS = "end_table_actions"
    DISCARD = "discard"
    PENALTY = "penalty"
    FINISH = "finish"
    ROUND_END = "round_end"


@dataclass(frozen=True, slots=True)
class EngineEvent:
    type: EventType
    player_id: int | None = None
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttachmentUsage:
    meld_id: int
    side: AttachmentSide
    count: int


@dataclass(frozen=True, slots=True)
class TurnContext:
    draw_source: DrawSource | None = None
    drawn_tile_id: int | None = None
    taken_discard_tile_id: int | None = None
    taken_discard_used: bool = False
    opened_mode_at_start: OpenedMode = OpenedMode.NONE
    opened_this_turn: bool = False
    stock_exhausted_after_draw: bool = False
    attachment_usage: tuple[AttachmentUsage, ...] = ()

    @property
    def must_use_taken_discard(self) -> bool:
        return self.taken_discard_tile_id is not None and not self.taken_discard_used

    def mark_tiles_used(self, tile_ids: set[int]) -> TurnContext:
        if self.taken_discard_tile_id is None or self.taken_discard_tile_id not in tile_ids:
            return self
        return replace(self, taken_discard_used=True)

    def attachment_count(self, meld_id: int, side: AttachmentSide) -> int:
        for usage in self.attachment_usage:
            if usage.meld_id == meld_id and usage.side is side:
                return usage.count
        return 0

    def add_attachment_usage(
        self,
        meld_id: int,
        side: AttachmentSide,
        amount: int,
    ) -> TurnContext:
        existing = self.attachment_count(meld_id, side)
        retained = tuple(
            usage
            for usage in self.attachment_usage
            if not (usage.meld_id == meld_id and usage.side is side)
        )
        return replace(
            self,
            attachment_usage=(
                *retained,
                AttachmentUsage(meld_id=meld_id, side=side, count=existing + amount),
            ),
        )


@dataclass(frozen=True, slots=True)
class GameState:
    round_id: int
    turn_number: int
    current_player: int
    starting_player: int
    indicator: PhysicalTile
    okey_value: TileValue
    stock: tuple[PhysicalTile, ...]
    discard_pile: tuple[PhysicalTile, ...]
    players: tuple[PlayerState, ...]
    table: TableState = field(default_factory=TableState)
    progressive_series_threshold: int = 101
    progressive_pair_threshold: int = 5
    phase: TurnPhase = TurnPhase.TABLE_ACTIONS
    terminal: bool = False
    terminal_reason: TerminalReason | None = None
    winner: int | None = None
    turn_context: TurnContext = field(default_factory=TurnContext)

    @property
    def discard_top(self) -> PhysicalTile | None:
        return self.discard_pile[-1] if self.discard_pile else None

    @property
    def stock_count(self) -> int:
        return len(self.stock)

    @property
    def current_player_state(self) -> PlayerState:
        return self.players[self.current_player]

    def replace_player(self, player_id: int, player: PlayerState) -> GameState:
        if not 0 <= player_id < len(self.players):
            raise ValueError(f"Invalid player id: {player_id}")
        players = list(self.players)
        players[player_id] = player
        return replace(self, players=tuple(players))
