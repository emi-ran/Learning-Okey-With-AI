"""Deterministic legal-action enumeration for the phased round engine."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import combinations

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
from .melds import Meld, MeldKind, MeldTile
from .pairs import Pair, build_pair
from .player import OpenedMode
from .state import DrawSource, GameState, TurnContext, TurnPhase
from .table import AttachmentSide, TableMeld
from .tiles import Color, PhysicalTile, TileKind, TileValue


_RUN_TARGETS = tuple(
    tuple(TileValue(color, number) for number in range(start, end + 1))
    for color in Color
    for start in range(1, 12)
    for end in range(start + 2, 14)
)
_SET_TARGETS = tuple(
    tuple(TileValue(color, number) for color in colors)
    for number in range(1, 14)
    for length in (3, 4)
    for colors in combinations(Color, length)
)


def _index_hand(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
) -> tuple[dict[TileValue, tuple[PhysicalTile, ...]], tuple[PhysicalTile, ...]]:
    fixed_by_value: dict[TileValue, list[PhysicalTile]] = {}
    jokers: list[PhysicalTile] = []
    for tile in hand:
        value = effective_value(tile, okey_value)
        if value is None:
            jokers.append(tile)
        else:
            fixed_by_value.setdefault(value, []).append(tile)
    return (
        {
            value: tuple(sorted(tiles, key=lambda candidate: candidate.id))
            for value, tiles in fixed_by_value.items()
        },
        tuple(sorted(jokers, key=lambda candidate: candidate.id)),
    )


def _assign_indexed_targets(
    fixed_by_value: dict[TileValue, tuple[PhysicalTile, ...]],
    jokers: tuple[PhysicalTile, ...],
    targets: Sequence[TileValue],
) -> tuple[tuple[MeldTile, ...], ...]:
    if sum(bool(fixed_by_value.get(target)) for target in targets) + len(jokers) < len(
        targets
    ):
        return ()
    results: list[tuple[MeldTile, ...]] = []

    def visit(
        target_index: int,
        used_ids: frozenset[int],
        assigned: tuple[MeldTile, ...],
    ) -> None:
        if target_index == len(targets):
            results.append(assigned)
            return
        target = targets[target_index]
        choices = (*fixed_by_value.get(target, ()), *jokers)
        for tile in choices:
            if tile.id in used_ids:
                continue
            visit(
                target_index + 1,
                used_ids | {tile.id},
                (*assigned, MeldTile(tile, target)),
            )

    visit(0, frozenset(), ())
    return tuple(results)


def _target_tile_assignments(
    hand: Sequence[PhysicalTile],
    targets: Sequence[TileValue],
    okey_value: TileValue,
) -> tuple[tuple[MeldTile, ...], ...]:
    """Assign distinct physical tiles to an ordered logical target."""

    fixed_by_value, jokers = _index_hand(hand, okey_value)
    return _assign_indexed_targets(fixed_by_value, jokers, targets)


def generate_meld_candidates(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
) -> tuple[Meld, ...]:
    """Enumerate canonical run/set candidates without subset brute force."""

    candidates: list[Meld] = []
    fixed_by_value, jokers = _index_hand(hand, okey_value)
    for targets in _RUN_TARGETS:
        for assignment in _assign_indexed_targets(
            fixed_by_value,
            jokers,
            targets,
        ):
            candidates.append(Meld(MeldKind.RUN, assignment))

    for targets in _SET_TARGETS:
        for assignment in _assign_indexed_targets(
            fixed_by_value,
            jokers,
            targets,
        ):
            candidates.append(Meld(MeldKind.SET, assignment))

    deduplicated: dict[tuple[object, ...], Meld] = {}
    for meld in candidates:
        key = (
            meld.kind,
            tuple(
                (
                    tile.physical_tile.id,
                    tile.represented_value.color,
                    tile.represented_value.number,
                )
                for tile in meld.tiles
            ),
        )
        deduplicated[key] = meld
    return tuple(deduplicated[key] for key in sorted(deduplicated, key=repr))


def generate_pair_candidates(
    hand: Sequence[PhysicalTile],
    okey_value: TileValue,
) -> tuple[Pair, ...]:
    """Enumerate all physical two-tile pairs, including Okey combinations."""

    pairs: list[Pair] = []
    for first, second in combinations(hand, 2):
        try:
            pairs.append(build_pair((first, second), okey_value))
        except ValueError:
            continue
    return tuple(pairs)


def _meld_ids(meld: Meld) -> frozenset[int]:
    return frozenset(tile.physical_tile.id for tile in meld.tiles)


def _pair_ids(pair: Pair) -> frozenset[int]:
    return frozenset(tile.physical_tile.id for tile in pair.tiles)


def _opening_meld_groups(
    candidates: Sequence[Meld],
    *,
    threshold: int,
    hand_size: int,
) -> tuple[tuple[Meld, ...], ...]:
    groups: list[tuple[Meld, ...]] = []
    candidate_ids = tuple(_meld_ids(meld) for meld in candidates)

    def visit(
        start: int,
        selected: tuple[Meld, ...],
        used_ids: frozenset[int],
        score: int,
    ) -> None:
        if selected and score >= threshold:
            groups.append(selected)
        for index in range(start, len(candidates)):
            ids = candidate_ids[index]
            if ids & used_ids or len(used_ids | ids) >= hand_size:
                continue
            visit(
                index + 1,
                (*selected, candidates[index]),
                used_ids | ids,
                score + candidates[index].score,
            )

    visit(0, (), frozenset(), 0)
    return tuple(groups)


def _opening_pair_groups(
    candidates: Sequence[Pair],
    *,
    threshold: int,
    hand_size: int,
) -> tuple[tuple[Pair, ...], ...]:
    groups: list[tuple[Pair, ...]] = []
    candidate_ids = tuple(_pair_ids(pair) for pair in candidates)

    def visit(
        start: int,
        selected: tuple[Pair, ...],
        used_ids: frozenset[int],
    ) -> None:
        if len(selected) >= threshold:
            groups.append(selected)
        for index in range(start, len(candidates)):
            ids = candidate_ids[index]
            if ids & used_ids or len(used_ids | ids) >= hand_size:
                continue
            visit(index + 1, (*selected, candidates[index]), used_ids | ids)

    visit(0, (), frozenset())
    return tuple(groups)


def _attachment_targets(
    table_meld: TableMeld,
    side: AttachmentSide,
    amount: int,
) -> tuple[TileValue, ...]:
    represented = tuple(tile.represented_value for tile in table_meld.meld.tiles)
    if table_meld.meld.kind is MeldKind.RUN:
        color = represented[0].color
        if side is AttachmentSide.LEFT:
            start = represented[0].number - amount
            if start < 1:
                return ()
            return tuple(
                TileValue(color, number)
                for number in range(start, represented[0].number)
            )
        end = represented[-1].number + amount
        if end > 13:
            return ()
        return tuple(
            TileValue(color, number)
            for number in range(represented[-1].number + 1, end + 1)
        )

    if side is not AttachmentSide.SET:
        return ()
    number = represented[0].number
    used_colors = {value.color for value in represented}
    missing = tuple(color for color in Color if color not in used_colors)
    if amount > len(missing):
        return ()
    # SET callers expand each color combination separately.
    return tuple(TileValue(color, number) for color in missing)


def _attachment_actions(
    state: GameState,
    hand: Sequence[PhysicalTile],
    config: RulesConfig,
    *,
    preserve_final_discard: bool = True,
) -> tuple[AddToMeld, ...]:
    actions: list[AddToMeld] = []
    for table_meld in state.table.melds:
        if table_meld.meld.kind is MeldKind.RUN:
            sides = (AttachmentSide.LEFT, AttachmentSide.RIGHT)
        else:
            sides = (AttachmentSide.SET,)
        for side in sides:
            already_used = state.turn_context.attachment_count(table_meld.id, side)
            remaining = config.max_contiguous_attach - already_used
            usable_hand_size = len(hand) - int(preserve_final_discard)
            for amount in range(1, min(2, remaining, usable_hand_size) + 1):
                targets = _attachment_targets(table_meld, side, amount)
                target_groups: Iterable[tuple[TileValue, ...]]
                if table_meld.meld.kind is MeldKind.SET:
                    target_groups = combinations(targets, amount)
                else:
                    target_groups = (targets,) if len(targets) == amount else ()
                for target_group in target_groups:
                    for assignment in _target_tile_assignments(
                        hand,
                        target_group,
                        state.okey_value,
                    ):
                        actions.append(
                            AddToMeld(
                                meld_id=table_meld.id,
                                tiles=assignment,
                                side=side,
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
    meld_candidates = generate_meld_candidates(hand, state.okey_value)
    pair_candidates = generate_pair_candidates(hand, state.okey_value)

    if player.opened_mode is OpenedMode.NONE:
        actions.extend(
            OpenMelds(group)
            for group in _opening_meld_groups(
                meld_candidates,
                threshold=state.progressive_series_threshold,
                hand_size=len(hand),
            )
        )
        actions.extend(
            OpenPairs(group)
            for group in _opening_pair_groups(
                pair_candidates,
                threshold=state.progressive_pair_threshold,
                hand_size=len(hand),
            )
        )
    elif player.opened_mode is OpenedMode.SERIES:
        actions.extend(
            OpenMelds((meld,))
            for meld in meld_candidates
            if len(_meld_ids(meld)) < len(hand)
        )

    if player.opened_mode is not OpenedMode.NONE:
        actions.extend(_attachment_actions(state, hand, config))
        actions.extend(_replacement_actions(state))
        if player.opened_mode is OpenedMode.PAIRS or state.table.pairs:
            actions.extend(
                AddPair(pair)
                for pair in pair_candidates
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


def _can_use_previous_discard(
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
    return any(
        tile.id in _action_tile_ids(action)
        for action in _table_actions(hypothetical, config)
    )


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
        if _can_use_previous_discard(state, config):
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
