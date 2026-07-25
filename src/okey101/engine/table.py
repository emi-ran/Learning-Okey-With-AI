from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .melds import Meld
from .pairs import Pair


class AttachmentSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    SET = "set"


@dataclass(frozen=True, slots=True)
class TableMeld:
    id: int
    meld: Meld


@dataclass(frozen=True, slots=True)
class TableState:
    melds: tuple[TableMeld, ...] = ()
    pairs: tuple[Pair, ...] = ()
    next_meld_id: int = 0

    def meld(self, meld_id: int) -> TableMeld:
        for table_meld in self.melds:
            if table_meld.id == meld_id:
                return table_meld
        raise ValueError(f"Unknown table meld: {meld_id}")

    def add_melds(self, melds: tuple[Meld, ...]) -> tuple[TableState, tuple[int, ...]]:
        ids = tuple(range(self.next_meld_id, self.next_meld_id + len(melds)))
        additions = tuple(TableMeld(id=meld_id, meld=meld) for meld_id, meld in zip(ids, melds))
        return (
            TableState(
                melds=(*self.melds, *additions),
                pairs=self.pairs,
                next_meld_id=self.next_meld_id + len(melds),
            ),
            ids,
        )

    def replace_meld(self, meld_id: int, meld: Meld) -> TableState:
        if not any(table_meld.id == meld_id for table_meld in self.melds):
            raise ValueError(f"Unknown table meld: {meld_id}")
        return TableState(
            melds=tuple(
                TableMeld(id=table_meld.id, meld=meld)
                if table_meld.id == meld_id
                else table_meld
                for table_meld in self.melds
            ),
            pairs=self.pairs,
            next_meld_id=self.next_meld_id,
        )

    def add_pairs(self, pairs: tuple[Pair, ...]) -> TableState:
        return TableState(
            melds=self.melds,
            pairs=(*self.pairs, *pairs),
            next_meld_id=self.next_meld_id,
        )
