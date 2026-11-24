"""H7-R1 volatility bucketing and deterministic matched-control selection.

This module is deliberately independent of H6 and contains no market-data loading,
event discovery, return calculation, validation access, or trading actions.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSet
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

VOLATILITY_LOOKBACK = 1_000
VOLATILITY_BINS = (0.0, 0.8, 1.0, 1.25, 1.6, float("inf"))
MAX_CONTROLS = 5
MIN_CONTROLS = 3
MATCH_WINDOW_BARS = 1_440
EVENT_EXCLUSION_RADIUS_BARS = 96

_MATCH_FIELDS = ("symbol", "period", "hour", "volatility_bucket", "direction")


def volatility_buckets(
    atr: Iterable[float],
    lookback: int = VOLATILITY_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ATR/previous-median ratio, bucket) for each completed M15 bar.

    The median uses exactly the previous `lookback` ATR observations and excludes
    the current bar. The current ATR is the completed ATR(14) for that bar.
    Bucket -1 denotes an unavailable/invalid warm-up observation.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")

    values = np.asarray(list(atr), dtype=float)
    if values.ndim != 1:
        raise ValueError("atr must be a one-dimensional sequence")

    series = pd.Series(values, dtype="float64")
    previous_median = (
        series.shift(1)
        .rolling(window=lookback, min_periods=lookback)
        .median()
        .to_numpy()
    )

    ratio = np.full(values.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(values)
        & (values > 0)
        & np.isfinite(previous_median)
        & (previous_median > 0)
    )
    ratio[valid] = values[valid] / previous_median[valid]

    # np.digitize with the frozen edges yields:
    # [0, .8), [.8, 1), [1, 1.25), [1.25, 1.6), [1.6, +inf).
    buckets = np.full(values.shape, -1, dtype=np.int8)
    buckets[valid] = (
        np.digitize(ratio[valid], np.asarray(VOLATILITY_BINS)) - 1
    ).astype(np.int8)
    return ratio, buckets


def _required_int(record: Mapping[str, Any], field: str, allowed=None) -> int:
    if field not in record:
        raise ValueError(f"missing matching field: {field}")
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    value = int(value)
    if allowed is not None and value not in allowed:
        raise ValueError(f"{field} must be one of {allowed}")
    return value


def _validate_match_record(record: Mapping[str, Any], label: str) -> None:
    missing = [field for field in (*_MATCH_FIELDS, "idx") if field not in record]
    if missing:
        raise ValueError(f"{label} missing fields: {', '.join(missing)}")
    if not isinstance(record["symbol"], str) or not record["symbol"].strip():
        raise ValueError(f"{label} symbol must be a non-empty string")
    if not isinstance(record["period"], str) or not record["period"].strip():
        raise ValueError(f"{label} period must be a non-empty string")
    _required_int(record, "idx")
    _required_int(record, "hour", range(24))
    _required_int(record, "volatility_bucket", range(5))
    _required_int(record, "direction", (-1, 1))


def select_h7_controls(
    event: Mapping[str, Any],
    candidate_controls: Iterable[Mapping[str, Any]],
    qualifying_event_indices: Iterable[int],
    used_controls: MutableSet[tuple[str, int]],
    *,
    max_controls: int = MAX_CONTROLS,
    min_controls: int = MIN_CONTROLS,
    window_bars: int = MATCH_WINDOW_BARS,
    exclusion_radius_bars: int = EVENT_EXCLUSION_RADIUS_BARS,
) -> tuple[int, ...]:
    """Select up to five nearest controls for one event without widening the window.

    Candidates must exactly match symbol, period, hour, volatility bucket and
    direction; lie within +/-1,440 M15 indices; and be more than 96 indices from
    every qualifying H7 sweep event for that symbol.

    The caller supplies one shared `used_controls` set across all events, periods
    and dose groups for a symbol. A control is reserved only when at least
    `min_controls` are available. If fewer are available, their indices are
    returned for attrition reporting but none are marked as used; the caller must
    exclude that event from matched-excess analysis.
    """
    _validate_match_record(event, "event")
    if isinstance(max_controls, bool) or not isinstance(max_controls, int):
        raise ValueError("max_controls must be an integer")
    if isinstance(min_controls, bool) or not isinstance(min_controls, int):
        raise ValueError("min_controls must be an integer")
    if not 1 <= min_controls <= max_controls:
        raise ValueError("require 1 <= min_controls <= max_controls")
    if isinstance(window_bars, bool) or not isinstance(window_bars, int) or window_bars < 0:
        raise ValueError("window_bars must be a non-negative integer")
    if (
        isinstance(exclusion_radius_bars, bool)
        or not isinstance(exclusion_radius_bars, int)
        or exclusion_radius_bars < 0
    ):
        raise ValueError("exclusion_radius_bars must be a non-negative integer")

    event_idx = _required_int(event, "idx")
    if event_idx < 0:
        raise ValueError("event idx must be non-negative")
    symbol = event["symbol"]
    event_indices = []
    for value in qualifying_event_indices:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("qualifying event indices must be non-negative integers")
        event_indices.append(int(value))
    if event_idx not in event_indices:
        raise ValueError("qualifying_event_indices must include the event being matched")

    unique_candidates: dict[int, Mapping[str, Any]] = {}
    for candidate in candidate_controls:
        _validate_match_record(candidate, "candidate control")
        if any(candidate[field] != event[field] for field in _MATCH_FIELDS):
            continue
        idx = _required_int(candidate, "idx")
        if idx < 0 or abs(idx - event_idx) > window_bars:
            continue
        if (symbol, idx) in used_controls:
            continue
        if any(abs(idx - other_event_idx) <= exclusion_radius_bars
               for other_event_idx in event_indices):
            continue
        if idx in unique_candidates:
            raise ValueError(f"duplicate candidate control index for {symbol}: {idx}")
        unique_candidates[idx] = candidate

    chosen = tuple(sorted(
        unique_candidates,
        key=lambda idx: (abs(idx - event_idx), idx),
    )[:max_controls])

    # An event with fewer than the required number is retained for attrition
    # reporting, but it must not consume controls that a later event can use.
    if len(chosen) >= min_controls:
        used_controls.update((symbol, idx) for idx in chosen)
    return chosen
