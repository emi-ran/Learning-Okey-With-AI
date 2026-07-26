"""Player-perspective observations with an explicit hidden-information boundary."""

from __future__ import annotations

from dataclasses import dataclass

from okey101.engine.joker import is_real_okey
from okey101.engine.player import OpenedMode
from okey101.engine.state import GameState, TurnPhase
from okey101.engine.tiles import Color, PhysicalTile, TileKind, TileValue

_COLORS = tuple(Color)
_VALUE_INDEX = {
    TileValue(color, number): color_index * 13 + number - 1
    for color_index, color in enumerate(_COLORS)
    for number in range(1, 14)
}


@dataclass(frozen=True, slots=True)
class VisibleTile:
    """A public physical tile without exposing an engine-only object reference."""

    tile_id: int
    kind: TileKind
    color: Color | None
    number: int | None
    is_real_okey: bool


@dataclass(frozen=True, slots=True)
class VisibleMeldTile:
    tile: VisibleTile
    represented_value: TileValue


@dataclass(frozen=True, slots=True)
class VisibleMeld:
    meld_id: int
    kind: str
    tiles: tuple[VisibleMeldTile, ...]


@dataclass(frozen=True, slots=True)
class VisiblePair:
    tiles: tuple[VisibleMeldTile, VisibleMeldTile]


@dataclass(frozen=True, slots=True)
class VisibleDiscard:
    tile: VisibleTile
    player_relative: int
    turn_number: int
    taken_by_relative: int | None


@dataclass(frozen=True, slots=True)
class PublicPlayerStatus:
    """Public per-seat information in perspective-relative seat order."""

    relative_seat: int
    opened_mode: OpenedMode
    score: int
    immediate_penalty: int


@dataclass(frozen=True, slots=True)
class PlayerObservation:
    """Everything a real player may observe, and no hidden tile locations."""

    player_id: int
    current_player_relative: int
    turn_number: int
    phase: TurnPhase
    own_normal_counts: tuple[int, ...]
    own_fake_okey_count: int
    own_tile_ids: tuple[int, ...]
    indicator: VisibleTile
    okey_value: TileValue
    table_melds: tuple[VisibleMeld, ...]
    pair_area: tuple[VisiblePair, ...]
    discard_history: tuple[VisibleDiscard, ...]
    player_statuses: tuple[PublicPlayerStatus, ...]
    progressive_series_threshold: int
    progressive_pair_threshold: int
    stock_count: int


def _visible_tile(tile: PhysicalTile, okey_value: TileValue) -> VisibleTile:
    value = tile.value
    return VisibleTile(
        tile_id=tile.id,
        kind=tile.kind,
        color=value.color if value is not None else None,
        number=value.number if value is not None else None,
        is_real_okey=is_real_okey(tile, okey_value),
    )


def _hand_counts(
    hand: tuple[PhysicalTile, ...],
) -> tuple[tuple[int, ...], int]:
    counts = [0] * 52
    fake_okeys = 0
    for tile in hand:
        if tile.kind is TileKind.FAKE_OKEY:
            fake_okeys += 1
            continue
        value = tile.value
        assert value is not None
        counts[_VALUE_INDEX[value]] += 1
    return tuple(counts), fake_okeys


def get_observation(state: GameState, player_id: int) -> PlayerObservation:
    """Project ``state`` into the information visible to ``player_id``.

    Opponent hands and stock identities are intentionally absent. Own physical
    IDs remain available because discard and table actions address physical
    tiles, while the neural features can use the canonical count vector.
    """

    if not 0 <= player_id < len(state.players):
        raise ValueError(f"Invalid player id: {player_id}")

    own_hand = state.players[player_id].hand
    normal_counts, fake_okey_count = _hand_counts(own_hand)
    player_count = len(state.players)

    statuses = tuple(
        PublicPlayerStatus(
            relative_seat=relative_seat,
            opened_mode=state.players[absolute_seat].opened_mode,
            score=state.players[absolute_seat].score,
            immediate_penalty=state.players[absolute_seat].immediate_penalty,
        )
        for relative_seat in range(player_count)
        for absolute_seat in ((player_id + relative_seat) % player_count,)
    )

    melds = tuple(
        VisibleMeld(
            meld_id=table_meld.id,
            kind=table_meld.meld.kind.value,
            tiles=tuple(
                VisibleMeldTile(
                    tile=_visible_tile(meld_tile.physical_tile, state.okey_value),
                    represented_value=meld_tile.represented_value,
                )
                for meld_tile in table_meld.meld.tiles
            ),
        )
        for table_meld in state.table.melds
    )
    pairs = tuple(
        VisiblePair(
            tiles=(
                VisibleMeldTile(
                    tile=_visible_tile(pair.tiles[0].physical_tile, state.okey_value),
                    represented_value=pair.tiles[0].represented_value,
                ),
                VisibleMeldTile(
                    tile=_visible_tile(pair.tiles[1].physical_tile, state.okey_value),
                    represented_value=pair.tiles[1].represented_value,
                ),
            )
        )
        for pair in state.table.pairs
    )

    return PlayerObservation(
        player_id=player_id,
        current_player_relative=(state.current_player - player_id) % player_count,
        turn_number=state.turn_number,
        phase=state.phase,
        own_normal_counts=normal_counts,
        own_fake_okey_count=fake_okey_count,
        own_tile_ids=tuple(sorted(tile.id for tile in own_hand)),
        indicator=_visible_tile(state.indicator, state.okey_value),
        okey_value=state.okey_value,
        table_melds=melds,
        pair_area=pairs,
        discard_history=tuple(
            VisibleDiscard(
                tile=_visible_tile(record.tile, state.okey_value),
                player_relative=(record.player_id - player_id) % player_count,
                turn_number=record.turn_number,
                taken_by_relative=(
                    None
                    if record.taken_by is None
                    else (record.taken_by - player_id) % player_count
                ),
            )
            for record in state.discard_history
        ),
        player_statuses=statuses,
        progressive_series_threshold=state.progressive_series_threshold,
        progressive_pair_threshold=state.progressive_pair_threshold,
        stock_count=state.stock_count,
    )
