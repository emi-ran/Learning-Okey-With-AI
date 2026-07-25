"""Indicator, real Okey and fake Okey semantics."""

from __future__ import annotations

from .tiles import (
    PhysicalTile,
    TileKind,
    TileValue,
    effective_value,
    is_real_okey,
)


def _as_value(indicator: TileValue | PhysicalTile) -> TileValue:
    if isinstance(indicator, TileValue):
        return indicator
    if indicator.kind is TileKind.FAKE_OKEY:
        raise ValueError("the indicator must be a normal physical tile")
    value = indicator.value
    assert value is not None
    return value


def okey_value_for_indicator(indicator: TileValue | PhysicalTile) -> TileValue:
    """Return the same-color successor, wrapping indicator 13 to Okey 1."""

    value = _as_value(indicator)
    return TileValue(value.color, value.number % 13 + 1)


__all__ = [
    "effective_value",
    "is_real_okey",
    "okey_value_for_indicator",
]
