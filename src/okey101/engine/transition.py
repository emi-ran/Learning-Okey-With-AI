from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .actions import (
    Action,
    AddPair,
    AddToMeld,
    Discard,
    DrawFromStock,
    EndTableActions,
    OpenMelds,
    OpenPairs,
    ReplaceJoker,
    TakePreviousDiscard,
)
from .config import RulesConfig
from .joker import is_real_okey
from .melds import Meld, MeldKind, MeldTile, validate_meld
from .pairs import Pair, validate_pair
from .player import OpenedMode, PlayerState
from .state import (
    DiscardRecord,
    DrawSource,
    EngineEvent,
    EventType,
    GameState,
    TerminalReason,
    TurnContext,
    TurnPhase,
)
from .table import AttachmentSide
from .tiles import PhysicalTile, TileKind


class IllegalAction(ValueError):
    """Raised when an action violates the current phase or a game rule."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IllegalAction(message)


def _meld_tile_physical(meld_tile: MeldTile) -> PhysicalTile:
    return meld_tile.physical_tile


def _meld_tile_ids(meld: Meld) -> tuple[int, ...]:
    return tuple(_meld_tile_physical(tile).id for tile in meld.tiles)


def _pair_tile_ids(pair: Pair) -> tuple[int, ...]:
    return tuple(_meld_tile_physical(tile).id for tile in pair.tiles)


def _all_unique(ids: Iterable[int]) -> bool:
    values = tuple(ids)
    return len(values) == len(set(values))


def _validate_meld(meld: Meld, state: GameState) -> None:
    try:
        valid = validate_meld(meld, state.okey_value)
    except (TypeError, ValueError) as exc:
        raise IllegalAction(f"Invalid meld: {exc}") from exc
    _require(valid, "Invalid meld")


def _validate_pair(pair: Pair, state: GameState) -> None:
    try:
        valid = validate_pair(pair, state.okey_value)
    except (TypeError, ValueError) as exc:
        raise IllegalAction(f"Invalid pair: {exc}") from exc
    _require(valid, "Invalid pair")


def _require_hand_tiles(
    player: PlayerState,
    tiles: Iterable[PhysicalTile],
) -> tuple[int, ...]:
    supplied = tuple(tiles)
    ids = tuple(tile.id for tile in supplied)
    _require(_all_unique(ids), "A physical tile cannot be used more than once")
    hand_by_id = {tile.id: tile for tile in player.hand}
    missing = set(ids) - set(hand_by_id)
    _require(not missing, f"Tiles are not in the player's hand: {sorted(missing)}")
    mismatched = sorted(
        tile.id for tile in supplied if hand_by_id[tile.id] != tile
    )
    _require(
        not mismatched,
        f"Action tile payload does not match hand identity: {mismatched}",
    )
    return ids


def _consume_tiles(player: PlayerState, tile_ids: tuple[int, ...]) -> PlayerState:
    try:
        updated = player.remove_tiles(tile_ids)
    except ValueError as exc:
        raise IllegalAction(str(exc)) from exc
    _require(updated.hand, "At least one tile must remain for the final discard")
    return updated


def _meld_score(melds: tuple[Meld, ...]) -> int:
    return sum(tile.represented_value.number for meld in melds for tile in meld.tiles)


def _terminal_state(
    state: GameState,
    reason: TerminalReason,
    *,
    winner: int | None,
) -> GameState:
    return replace(
        state,
        phase=TurnPhase.TERMINAL,
        terminal=True,
        terminal_reason=reason,
        winner=winner,
    )


def _mark_used(context: TurnContext, tile_ids: Iterable[int]) -> TurnContext:
    return context.mark_tiles_used(set(tile_ids))


def _finish_reason(state: GameState, discarded: PhysicalTile) -> TerminalReason:
    player = state.current_player_state
    okey_finish = is_real_okey(discarded, state.okey_value)
    if player.opened_mode is OpenedMode.PAIRS:
        return TerminalReason.PAIR_OKEY_FINISH if okey_finish else TerminalReason.PAIR_FINISH
    if state.turn_context.opened_this_turn:
        return (
            TerminalReason.SAME_TURN_OPEN_OKEY_FINISH
            if okey_finish
            else TerminalReason.SAME_TURN_OPEN_FINISH
        )
    return TerminalReason.OKEY_FINISH if okey_finish else TerminalReason.NORMAL_FINISH


def _advance_turn(state: GameState) -> GameState:
    next_player = (state.current_player + 1) % len(state.players)
    return replace(
        state,
        turn_number=state.turn_number + 1,
        current_player=next_player,
        phase=TurnPhase.DRAW_DECISION,
        turn_context=TurnContext(
            opened_mode_at_start=state.players[next_player].opened_mode,
        ),
    )


def _draw_from_stock(
    state: GameState,
    action: DrawFromStock,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    del action
    _require(state.phase is TurnPhase.DRAW_DECISION, "Stock draw is not allowed in this phase")
    _require(bool(state.stock), "The stock is empty")

    tile = state.stock[-1]
    stock = state.stock[:-1]
    player = state.current_player_state.add_tiles((tile,))
    context = replace(
        state.turn_context,
        draw_source=DrawSource.STOCK,
        drawn_tile_id=tile.id,
        stock_exhausted_after_draw=not stock,
    )
    updated = state.replace_player(state.current_player, player)
    updated = replace(updated, stock=stock, phase=TurnPhase.TABLE_ACTIONS, turn_context=context)
    return updated, (
        EngineEvent(
            EventType.DRAW_STOCK,
            state.current_player,
            {"tile_id": tile.id, "stock_count": len(stock)},
        ),
    )


def _take_previous_discard(
    state: GameState,
    action: TakePreviousDiscard,
    config: RulesConfig,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    del action
    _require(
        state.phase is TurnPhase.DRAW_DECISION,
        "Previous discard cannot be taken in this phase",
    )
    _require(bool(state.discard_pile), "There is no previous discard to take")
    from .legal_actions import can_use_previous_discard

    _require(
        can_use_previous_discard(state, config),
        "Previous discard cannot be used legally this turn",
    )

    tile = state.discard_pile[-1]
    player = state.current_player_state.add_tiles((tile,))
    context = replace(
        state.turn_context,
        draw_source=DrawSource.PREVIOUS_DISCARD,
        drawn_tile_id=tile.id,
        taken_discard_tile_id=tile.id,
        taken_discard_used=False,
    )
    updated = state.replace_player(state.current_player, player)
    history = list(state.discard_history)
    for index in range(len(history) - 1, -1, -1):
        record = history[index]
        if record.tile.id == tile.id and record.taken_by is None:
            history[index] = replace(record, taken_by=state.current_player)
            break
    updated = replace(
        updated,
        discard_pile=state.discard_pile[:-1],
        discard_history=tuple(history),
        phase=TurnPhase.TABLE_ACTIONS,
        turn_context=context,
    )
    return updated, (
        EngineEvent(EventType.TAKE_DISCARD, state.current_player, {"tile_id": tile.id}),
    )


def _open_melds(
    state: GameState,
    action: OpenMelds,
    config: RulesConfig,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    _require(state.phase is TurnPhase.TABLE_ACTIONS, "Melds cannot be laid in this phase")
    _require(bool(action.melds), "At least one meld is required")

    player = state.current_player_state
    _require(
        player.opened_mode is not OpenedMode.PAIRS,
        "A pair-opened player cannot lay independent melds",
    )
    for meld in action.melds:
        _validate_meld(meld, state)

    tile_ids = tuple(tile_id for meld in action.melds for tile_id in _meld_tile_ids(meld))
    _require_hand_tiles(
        player,
        (
            tile.physical_tile
            for meld in action.melds
            for tile in meld.tiles
        ),
    )
    score = _meld_score(action.melds)

    opening = player.opened_mode is OpenedMode.NONE
    if opening:
        _require(
            score >= state.progressive_series_threshold,
            (
                f"Opening score {score} is below the required "
                f"{state.progressive_series_threshold}"
            ),
        )

    player = _consume_tiles(player, tile_ids)
    if opening:
        player = player.open(OpenedMode.SERIES, state.turn_number)

    table, meld_ids = state.table.add_melds(action.melds)
    context = _mark_used(state.turn_context, tile_ids)
    if opening:
        context = replace(context, opened_this_turn=True)

    threshold = state.progressive_series_threshold
    if opening and config.progressive_opening:
        threshold = max(threshold, score + 1)

    updated = state.replace_player(state.current_player, player)
    updated = replace(
        updated,
        table=table,
        turn_context=context,
        progressive_series_threshold=threshold,
    )
    event_type = EventType.OPEN_SERIES if opening else EventType.LAY_MELDS
    return updated, (
        EngineEvent(
            event_type,
            state.current_player,
            {"meld_ids": meld_ids, "score": score, "tile_ids": tile_ids},
        ),
    )


def _open_pairs(
    state: GameState,
    action: OpenPairs,
    config: RulesConfig,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    _require(state.phase is TurnPhase.TABLE_ACTIONS, "Pairs cannot be opened in this phase")
    _require(bool(action.pairs), "At least one pair is required")

    player = state.current_player_state
    _require(player.opened_mode is OpenedMode.NONE, "Player has already opened")
    _require(
        len(action.pairs) >= state.progressive_pair_threshold,
        (
            f"Pair opening count {len(action.pairs)} is below the required "
            f"{state.progressive_pair_threshold}"
        ),
    )
    for pair in action.pairs:
        _validate_pair(pair, state)

    tile_ids = tuple(tile_id for pair in action.pairs for tile_id in _pair_tile_ids(pair))
    _require_hand_tiles(
        player,
        (
            tile.physical_tile
            for pair in action.pairs
            for tile in pair.tiles
        ),
    )
    player = _consume_tiles(player, tile_ids).open(OpenedMode.PAIRS, state.turn_number)
    context = replace(
        _mark_used(state.turn_context, tile_ids),
        opened_this_turn=True,
    )
    table = state.table.add_pairs(action.pairs)
    threshold = state.progressive_pair_threshold
    if config.progressive_opening:
        threshold = max(threshold, len(action.pairs) + 1)

    updated = state.replace_player(state.current_player, player)
    updated = replace(
        updated,
        table=table,
        turn_context=context,
        progressive_pair_threshold=threshold,
    )
    event = EngineEvent(
        EventType.OPEN_PAIRS,
        state.current_player,
        {"pair_count": len(action.pairs), "tile_ids": tile_ids},
    )

    if all(candidate.opened_mode is OpenedMode.PAIRS for candidate in updated.players):
        updated = _terminal_state(
            updated,
            TerminalReason.ALL_PLAYERS_OPENED_PAIRS,
            winner=None,
        )
        return updated, (
            event,
            EngineEvent(
                EventType.ROUND_END,
                details={"reason": TerminalReason.ALL_PLAYERS_OPENED_PAIRS.value},
            ),
        )
    return updated, (event,)


def _add_to_meld(
    state: GameState,
    action: AddToMeld,
    config: RulesConfig,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    _require(state.phase is TurnPhase.TABLE_ACTIONS, "Attachment is not allowed in this phase")
    _require(
        state.current_player_state.opened_mode is not OpenedMode.NONE,
        "Player must open before attaching to table melds",
    )
    _require(bool(action.tiles), "At least one attachment tile is required")

    table_meld = state.table.meld(action.meld_id)
    kind = table_meld.meld.kind
    if kind is MeldKind.RUN:
        _require(
            action.side in (AttachmentSide.LEFT, AttachmentSide.RIGHT),
            "Run attachments require LEFT or RIGHT side",
        )
    else:
        _require(action.side is AttachmentSide.SET, "Set attachments require SET side")

    prior_count = state.turn_context.attachment_count(action.meld_id, action.side)
    _require(
        prior_count + len(action.tiles) <= config.max_contiguous_attach,
        (
            "At most "
            f"{config.max_contiguous_attach} tiles may be added to the same meld extension "
            "in one turn"
        ),
    )

    tile_ids = tuple(_meld_tile_physical(tile).id for tile in action.tiles)
    _require_hand_tiles(
        state.current_player_state,
        (tile.physical_tile for tile in action.tiles),
    )
    if action.side is AttachmentSide.LEFT:
        combined = (*action.tiles, *table_meld.meld.tiles)
    else:
        combined = (*table_meld.meld.tiles, *action.tiles)
    candidate = Meld(kind=kind, tiles=combined)
    _validate_meld(candidate, state)

    player = _consume_tiles(state.current_player_state, tile_ids)
    table = state.table.replace_meld(action.meld_id, candidate)
    context = _mark_used(state.turn_context, tile_ids)
    context = context.add_attachment_usage(action.meld_id, action.side, len(action.tiles))
    updated = state.replace_player(state.current_player, player)
    updated = replace(updated, table=table, turn_context=context)
    return updated, (
        EngineEvent(
            EventType.ADD_TO_MELD,
            state.current_player,
            {
                "meld_id": action.meld_id,
                "side": action.side.value,
                "tile_ids": tile_ids,
            },
        ),
    )


def _add_pair(
    state: GameState,
    action: AddPair,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    _require(state.phase is TurnPhase.TABLE_ACTIONS, "Pair cannot be added in this phase")
    player = state.current_player_state
    _require(player.opened_mode is not OpenedMode.NONE, "Player must open first")
    _require(
        player.opened_mode is OpenedMode.PAIRS or bool(state.table.pairs),
        "There is no open pair area",
    )
    _validate_pair(action.pair, state)
    tile_ids = _pair_tile_ids(action.pair)
    _require_hand_tiles(
        player,
        (tile.physical_tile for tile in action.pair.tiles),
    )
    player = _consume_tiles(player, tile_ids)
    context = _mark_used(state.turn_context, tile_ids)
    updated = state.replace_player(state.current_player, player)
    updated = replace(
        updated,
        table=state.table.add_pairs((action.pair,)),
        turn_context=context,
    )
    return updated, (
        EngineEvent(
            EventType.ADD_PAIR,
            state.current_player,
            {"tile_ids": tile_ids},
        ),
    )


def _replace_joker(
    state: GameState,
    action: ReplaceJoker,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    _require(
        state.phase is TurnPhase.TABLE_ACTIONS,
        "Joker cannot be replaced in this phase",
    )
    player = state.current_player_state
    _require(
        player.opened_mode is not OpenedMode.NONE,
        "Player must open before replacing a table joker",
    )

    table_meld = state.table.meld(action.meld_id)
    joker_index = next(
        (
            index
            for index, meld_tile in enumerate(table_meld.meld.tiles)
            if _meld_tile_physical(meld_tile).id == action.joker_tile_id
        ),
        None,
    )
    _require(joker_index is not None, "Joker tile is not in the selected meld")
    joker_meld_tile = table_meld.meld.tiles[joker_index]
    joker_tile = _meld_tile_physical(joker_meld_tile)
    _require(is_real_okey(joker_tile, state.okey_value), "Selected table tile is not a real Okey")

    try:
        replacement_tile = player.tile(action.replacement_tile_id)
    except ValueError as exc:
        raise IllegalAction(str(exc)) from exc
    _require(
        replacement_tile.kind is TileKind.NORMAL
        and not is_real_okey(replacement_tile, state.okey_value),
        "Replacement must be the exact normal physical tile, not a joker",
    )
    _require(
        replacement_tile.value == joker_meld_tile.represented_value,
        "Replacement tile does not match the joker's represented value",
    )

    replacement_meld_tile = MeldTile(
        physical_tile=replacement_tile,
        represented_value=joker_meld_tile.represented_value,
    )
    meld_tiles = list(table_meld.meld.tiles)
    meld_tiles[joker_index] = replacement_meld_tile
    candidate = Meld(kind=table_meld.meld.kind, tiles=tuple(meld_tiles))
    _validate_meld(candidate, state)

    player = player.remove_tiles((replacement_tile.id,)).add_tiles((joker_tile,))
    context = _mark_used(state.turn_context, (replacement_tile.id,))
    updated = state.replace_player(state.current_player, player)
    updated = replace(
        updated,
        table=state.table.replace_meld(action.meld_id, candidate),
        turn_context=context,
    )
    return updated, (
        EngineEvent(
            EventType.REPLACE_JOKER,
            state.current_player,
            {
                "meld_id": action.meld_id,
                "joker_tile_id": joker_tile.id,
                "replacement_tile_id": replacement_tile.id,
            },
        ),
    )


def _end_table_actions(
    state: GameState,
    action: EndTableActions,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    del action
    _require(
        state.phase is TurnPhase.TABLE_ACTIONS,
        "Table actions cannot end in this phase",
    )
    _require(
        not state.turn_context.must_use_taken_discard,
        "The taken discard must be used on the table this turn",
    )
    _require(bool(state.current_player_state.hand), "A final discard is required")
    return replace(state, phase=TurnPhase.DISCARD), (
        EngineEvent(EventType.END_TABLE_ACTIONS, state.current_player),
    )


def _discard(
    state: GameState,
    action: Discard,
    config: RulesConfig,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    _require(state.phase is TurnPhase.DISCARD, "Discard is not allowed in this phase")
    _require(
        not state.turn_context.must_use_taken_discard,
        "The taken discard must be used before discarding",
    )
    try:
        discarded = state.current_player_state.tile(action.tile_id)
    except ValueError as exc:
        raise IllegalAction(str(exc)) from exc

    player = state.current_player_state.remove_tiles((discarded.id,))
    is_final = not player.hand
    penalty = 0
    if not is_final:
        from .penalties import calculate_discard_penalty

        penalty = calculate_discard_penalty(
            state,
            discarded,
            is_final=False,
            config=config,
        )
        if penalty:
            player = player.add_immediate_penalty(penalty)

    updated = state.replace_player(state.current_player, player)
    updated = replace(
        updated,
        discard_pile=(*state.discard_pile, discarded),
        discard_history=(
            *state.discard_history,
            DiscardRecord(
                tile=discarded,
                player_id=state.current_player,
                turn_number=state.turn_number,
            ),
        ),
    )
    events: list[EngineEvent] = [
        EngineEvent(
            EventType.DISCARD,
            state.current_player,
            {"tile_id": discarded.id, "is_final": is_final},
        )
    ]
    if penalty:
        events.append(
            EngineEvent(
                EventType.PENALTY,
                state.current_player,
                {"amount": penalty, "reason": "discard"},
            )
        )

    if is_final:
        reason = _finish_reason(state, discarded)
        updated = _terminal_state(updated, reason, winner=state.current_player)
        events.extend(
            (
                EngineEvent(
                    EventType.FINISH,
                    state.current_player,
                    {
                        "reason": reason.value,
                        "tile_id": discarded.id,
                        "same_turn_open": state.turn_context.opened_this_turn,
                    },
                ),
                EngineEvent(EventType.ROUND_END, details={"reason": reason.value}),
            )
        )
        return updated, tuple(events)

    if state.turn_context.stock_exhausted_after_draw:
        updated = _terminal_state(updated, TerminalReason.STOCK_EXHAUSTED, winner=None)
        events.append(
            EngineEvent(
                EventType.ROUND_END,
                details={"reason": TerminalReason.STOCK_EXHAUSTED.value},
            )
        )
        return updated, tuple(events)

    return _advance_turn(updated), tuple(events)


def apply_action(
    state: GameState,
    action: Action,
    config: RulesConfig,
) -> tuple[GameState, tuple[EngineEvent, ...]]:
    """Apply one legal state-machine action without mutating the input state."""

    _require(not state.terminal, "Terminal state cannot accept actions")
    if isinstance(action, DrawFromStock):
        return _draw_from_stock(state, action)
    if isinstance(action, TakePreviousDiscard):
        return _take_previous_discard(state, action, config)
    if isinstance(action, OpenMelds):
        return _open_melds(state, action, config)
    if isinstance(action, OpenPairs):
        return _open_pairs(state, action, config)
    if isinstance(action, AddToMeld):
        return _add_to_meld(state, action, config)
    if isinstance(action, AddPair):
        return _add_pair(state, action)
    if isinstance(action, ReplaceJoker):
        return _replace_joker(state, action)
    if isinstance(action, EndTableActions):
        return _end_table_actions(state, action)
    if isinstance(action, Discard):
        return _discard(state, action, config)
    raise TypeError(f"Unsupported action type: {type(action).__name__}")


transition = apply_action
