"""Immutable data contracts for the H7 dose-response research pipeline.

This module deliberately contains no market-signal detection or return calculation.
It validates event records so discovery, layer classification and outcome measurement
can be implemented and tested as separate stages.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

H7_VERSION = "H7-v1"
LAYER_NAMES = ("L1_liquidity", "L2_support_resistance", "L3_structure", "L4_fvg")


def _require_utc(value: datetime, field: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")


@dataclass(frozen=True)
class LayerResult:
    """One independently evaluated binary layer, with its information timestamp."""

    name: str
    qualified: bool
    observed_at: datetime

    def __post_init__(self) -> None:
        if self.name not in LAYER_NAMES:
            raise ValueError(f"unknown H7 layer: {self.name}")
        if type(self.qualified) is not bool:
            raise TypeError("qualified must be a bool; unknown is not equivalent to False")
        _require_utc(self.observed_at, "observed_at")


@dataclass(frozen=True)
class H7Event:
    """A qualifying sweep event and its four as-of-decision layer observations."""

    symbol: str
    parent_time: datetime
    sweep_time: datetime
    decision_time: datetime
    entry_time: datetime
    direction: int
    layers: Tuple[LayerResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        for field in ("parent_time", "sweep_time", "decision_time", "entry_time"):
            _require_utc(getattr(self, field), field)
        if type(self.direction) is not int or self.direction not in (-1, 1):
            raise ValueError("direction must be integer +1 (long) or -1 (short)")
        if not isinstance(self.layers, tuple):
            raise TypeError("layers must be an immutable tuple of LayerResult values")
        if not self.parent_time < self.sweep_time:
            raise ValueError("sweep_time must follow parent_time")
        if not self.sweep_time < self.decision_time:
            raise ValueError("decision_time must follow sweep_time")
        if not self.decision_time < self.entry_time:
            raise ValueError("entry_time must follow decision_time")
        if len(self.layers) != len(LAYER_NAMES):
            raise ValueError(f"exactly {len(LAYER_NAMES)} layer results are required")
        names = tuple(layer.name for layer in self.layers)
        if set(names) != set(LAYER_NAMES) or len(set(names)) != len(LAYER_NAMES):
            raise ValueError(f"layers must contain each of {LAYER_NAMES} exactly once")
        for layer in self.layers:
            if layer.observed_at > self.decision_time:
                raise ValueError(
                    f"{layer.name} was observed after the common decision cutoff"
                )

    @property
    def dose(self) -> int:
        """Number of qualifying layers (0..4); unknown layers are rejected."""
        return sum(layer.qualified for layer in self.layers)

    @property
    def event_key(self) -> tuple[str, datetime, int]:
        """Deduplication key: one event per symbol, parent candle and direction."""
        return (self.symbol, self.parent_time, self.direction)

    def layer_map(self) -> dict[str, bool]:
        """Return layer flags keyed by their stable names."""
        return {layer.name: layer.qualified for layer in self.layers}
