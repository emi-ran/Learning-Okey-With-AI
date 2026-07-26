"""Deterministic legal-action enumeration for the phased round engine."""

from __future__ import annotations

from dataclasses import replace

from okey101.solver.attachment_solver import generate_attachments
from okey101.solver.meld_generator import (
    generate_melds as generate_meld_candidates,
)
from okey101.solver.opening_solver import _has_legal_opening, find_legal_openings
from okey101.solver.pair_solver import (
    _has_pair_opening,
    find_pair_openings,
    generate_pairs as generate_pair_candidates,
)

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
from .joker import effective_value, is_real_okey
from .melds import Meld
from .pairs import Pair
from .player import OpenedMode
from .state import DrawSource, GameState, TurnContext, TurnPhase
from .tiles import PhysicalTile, TileKind


def _meld_ids(meld: Meld) -> frozenset[int]:
    return frozenset(tile.physical_tile.id for tile in meld.tiles)


def _pair_ids(pair: Pair) -> frozenset[int]:
    return frozenset(tile.physical_tile.id for tile in pair.tiles)


def _attachment_actions(
    state: GameState,
    hand: tuple[PhysicalTile, ...],
    config: RulesConfig,
    *,
    preserve_final_discard: bool = True,
) -> tuple[AddToMeld, ...]:
    actions: list[AddToMeld] = []
    usable_hand_size = len(hand) - int(preserve_final_discard)
    if usable_hand_size < 1:
        return ()
    for table_meld in state.table.melds:
        candidates = generate_attachments(
            table_meld.meld,
            hand,
            state.okey_value,
            max_tiles=min(
                config.max_contiguous_attach,
                usable_hand_size,
            ),
        )
        for candidate in candidates:
            already_used = state.turn_context.attachment_count(
                table_meld.id,
                candidate.side,
            )
            remaining = config.max_contiguous_attach - already_used
            if len(candidate.tiles) <= remaining:
                actions.append(
                    AddToMeld(
                        meld_id=table_meld.id,
                        tiles=candidate.tiles,
                        side=candidate.side,
                    )
                )
    return tuple(actions)


def _replacement_actions(state: GameState) -> tuple[ReplaceJoker, ...]:
    actions: list[ReplaceJoker] = []
    for table_meld in state.table.melds:
        for meld_tile in table_meld.meld.tiles:
            if not is_real_okey(meld_tile.physical_tile, state.okey_value):
                continue
            for hand_tile in state.current_player_state.hand:
                if (
                    hand_tile.kind is TileKind.NORMAL
                    and not is_real_okey(hand_tile, state.okey_value)
                    and effective_value(hand_tile, state.okey_value)
                    == meld_tile.represented_value
                ):
                    actions.append(
                        ReplaceJoker(
                            meld_id=table_meld.id,
                            joker_tile_id=meld_tile.physical_tile.id,
                            replacement_tile_id=hand_tile.id,
                        )
                    )
    return tuple(actions)


def _action_tile_ids(action: Action) -> frozenset[int]:
    if isinstance(action, OpenMelds):
        return frozenset(
            tile.physical_tile.id
            for meld in action.melds
            for tile in meld.tiles
        )
    if isinstance(action, OpenPairs):
        return frozenset(
            tile.physical_tile.id
            for pair in action.pairs
            for tile in pair.tiles
        )
    if isinstance(action, AddToMeld):
        return frozenset(tile.physical_tile.id for tile in action.tiles)
    if isinstance(action, AddPair):
        return _pair_ids(action.pair)
    if isinstance(action, ReplaceJoker):
        return frozenset((action.replacement_tile_id,))
    if isinstance(action, Discard):
        return frozenset((action.tile_id,))
    return frozenset()


def _table_actions(state: GameState, config: RulesConfig) -> tuple[Action, ...]:
    player = state.current_player_state
    hand = player.hand
    actions: list[Action] = []
    required_tile_id = (
        state.turn_context.taken_discard_tile_id
        if state.turn_context.must_use_taken_discard
        else None
    )

    if player.opened_mode is OpenedMode.NONE:
        actions.extend(
            OpenMelds(candidate.melds)
            for candidate in find_legal_openings(
                hand,
                state.okey_value,
                threshold=state.progressive_series_threshold,
                required_tile_id=required_tile_id,
                preserve_final_discard=True,
            )
        )
        actions.extend(
            OpenPairs(candidate.pairs)
            for candidate in find_pair_openings(
                hand,
                state.okey_value,
                threshold=state.progressive_pair_threshold,
                required_tile_id=required_tile_id,
                preserve_final_discard=True,
            )
        )
    elif (
        player.opened_mode is OpenedMode.SERIES
        and required_tile_id is None
    ):
        actions.extend(
            OpenMelds((meld,))
            for meld in generate_meld_candidates(
                hand,
                state.okey_value,
                required_tile_id=required_tile_id,
            )
            if len(_meld_ids(meld)) < len(hand)
        )

    if player.opened_mode is not OpenedMode.NONE:
        actions.extend(_attachment_actions(state, hand, config))
        actions.extend(_replacement_actions(state))
        if player.opened_mode is OpenedMode.PAIRS or state.table.pairs:
            actions.extend(
                AddPair(pair)
                for pair in generate_pair_candidates(
                    hand,
                    state.okey_value,
                    required_tile_id=required_tile_id,
                )
                if len(_pair_ids(pair)) < len(hand)
            )

    if not state.turn_context.must_use_taken_discard and hand:
        actions.append(EndTableActions())
    if state.turn_context.must_use_taken_discard:
        required_tile_id = state.turn_context.taken_discard_tile_id
        assert required_tile_id is not None
        # Canonicalize the turn ordering: the mandatory tile is used first.
        # Unrelated table actions remain available immediately afterwards.
        actions = [
            action
            for action in actions
            if required_tile_id in _action_tile_ids(action)
        ]
    return tuple(actions)


def can_use_previous_discard(
    state: GameState,
    config: RulesConfig,
) -> bool:
    tile = state.discard_top
    if tile is None:
        return False
    hypothetical_player = state.current_player_state.add_tiles((tile,))
    hypothetical = state.replace_player(state.current_player, hypothetical_player)
    hypothetical = replace(
        hypothetical,
        discard_pile=state.discard_pile[:-1],
        phase=TurnPhase.TABLE_ACTIONS,
        turn_context=TurnContext(
            draw_source=DrawSource.PREVIOUS_DISCARD,
            drawn_tile_id=tile.id,
            taken_discard_tile_id=tile.id,
            opened_mode_at_start=hypothetical_player.opened_mode,
        ),
    )
    if hypothetical_player.opened_mode is OpenedMode.NONE:
        return _has_legal_opening(
            hypothetical_player.hand,
            hypothetical.okey_value,
            threshold=hypothetical.progressive_series_threshold,
            required_tile_id=tile.id,
            preserve_final_discard=True,
        ) or _has_pair_opening(
            hypothetical_player.hand,
            hypothetical.okey_value,
            threshold=hypothetical.progressive_pair_threshold,
            required_tile_id=tile.id,
            preserve_final_discard=True,
        )
    return bool(_table_actions(hypothetical, config))


def get_legal_actions(
    state: GameState,
    config: RulesConfig,
) -> tuple[Action, ...]:
    """Enumerate only actions accepted by the deterministic transition layer."""

    if state.terminal or state.phase is TurnPhase.TERMINAL:
        return ()
    if state.phase is TurnPhase.DRAW_DECISION:
        actions: list[Action] = []
        if state.stock:
            actions.append(DrawFromStock())
        if can_use_previous_discard(state, config):
            actions.append(TakePreviousDiscard())
        return tuple(actions)
    if state.phase is TurnPhase.TABLE_ACTIONS:
        return _table_actions(state, config)
    if state.phase is TurnPhase.DISCARD:
        return tuple(Discard(tile.id) for tile in state.current_player_state.hand)
    raise ValueError(f"Unsupported turn phase: {state.phase!r}")


def is_playable_discard(
    state: GameState,
    tile: PhysicalTile,
    config: RulesConfig | None = None,
) -> bool:
    """Whether one non-Okey tile can extend a current table meld right now."""

    rules = config or RulesConfig()
    return any(
        tile.id in _action_tile_ids(action)
        for action in _attachment_actions(
            state,
            (tile,),
            rules,
            preserve_final_discard=False,
        )
    )


def explain_action_legality(
    state: GameState,
    action: Action,
    config: RulesConfig,
) -> str:
    """Return a stable debug explanation for a candidate action."""

    if action in get_legal_actions(state, config):
        return "legal"
    try:
        from .transition import IllegalAction, apply_action

        apply_action(state, action, config)
    except IllegalAction as error:
        return str(error)
    except (TypeError, ValueError) as error:
        return str(error)
    return "action is valid in transition but was not canonically generated"
