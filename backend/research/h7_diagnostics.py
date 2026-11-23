"""Outcome-blind H7 event diagnostics.

This module accepts already-classified, eligible H7 sweep events. It never reads
price returns, computes performance, or accesses market data. It is intended for
the pre-outcome sample/count gate only.

Input record keys:
    symbol, parent_time, event_time, period, direction,
    L1_liquidity, L2_support_resistance, L3_structure, L4_fvg

Timestamps must be timezone-aware UTC datetimes or ISO-8601 strings with +00:00/Z.
Only the registered development (2022-2023) and validation (2024) periods are
accepted. 2025 onward is rejected to protect the sealed holdout.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable, Mapping, Any

LAYER_KEYS = (
    "L1_liquidity",
    "L2_support_resistance",
    "L3_structure",
    "L4_fvg",
)
PERIOD_START = {
    "development": datetime(2022, 1, 1, tzinfo=timezone.utc),
    "validation": datetime(2024, 1, 1, tzinfo=timezone.utc),
}
PERIOD_END = {
    "development": datetime(2024, 1, 1, tzinfo=timezone.utc),
    "validation": datetime(2025, 1, 1, tzinfo=timezone.utc),
}


def _utc_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise TypeError(f"{field} must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field} must be UTC")
    return value.astimezone(timezone.utc)


def _normalise_event(row: Mapping[str, Any]) -> dict[str, Any]:
    required = ("symbol", "parent_time", "event_time", "period", "direction", *LAYER_KEYS)
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"missing H7 event fields: {', '.join(missing)}")

    symbol = row["symbol"]
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    period = row["period"]
    if period not in PERIOD_START:
        raise ValueError("period must be development or validation; holdout is prohibited")

    parent_time = _utc_datetime(row["parent_time"], "parent_time")
    event_time = _utc_datetime(row["event_time"], "event_time")
    if not parent_time < event_time:
        raise ValueError("event_time must follow parent_time")
    if not PERIOD_START[period] <= event_time < PERIOD_END[period]:
        raise ValueError(f"event_time does not belong to registered {period} period")

    direction = row["direction"]
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("direction must be integer +1 or -1")

    layers = {}
    for key in LAYER_KEYS:
        value = row[key]
        if type(value) is not bool:
            raise ValueError(f"{key} must be bool; unknown is not False")
        layers[key] = value

    return {
        "symbol": symbol.strip(),
        "parent_time": parent_time,
        "event_time": event_time,
        "period": period,
        "direction": direction,
        **layers,
        "dose": sum(layers.values()),
    }


def _counts(rows: list[dict[str, Any]], key_fn) -> dict[str, int]:
    counts = Counter(key_fn(row) for row in rows)
    return {str(key): counts[key] for key in sorted(counts, key=str)}


def summarize_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return counts and L3/L4 overlap diagnostics; no outcome fields are accepted."""
    rows = [_normalise_event(row) for row in events]
    seen = set()
    for row in rows:
        key = (row["symbol"], row["parent_time"], row["direction"])
        if key in seen:
            raise ValueError(f"duplicate H7 event key: {key}")
        seen.add(key)

    by_period: dict[str, Any] = {}
    for period in ("development", "validation"):
        subset = [row for row in rows if row["period"] == period]
        dose_counts = {str(dose): sum(row["dose"] == dose for row in subset)
                       for dose in range(5)}
        l3_l4 = {
            "neither": sum(not r["L3_structure"] and not r["L4_fvg"] for r in subset),
            "L3_only": sum(r["L3_structure"] and not r["L4_fvg"] for r in subset),
            "L4_only": sum(not r["L3_structure"] and r["L4_fvg"] for r in subset),
            "both": sum(r["L3_structure"] and r["L4_fvg"] for r in subset),
        }
        by_period[period] = {
            "events": len(subset),
            "dose_counts": dose_counts,
            "dose_3_4": dose_counts["3"] + dose_counts["4"],
            "by_symbol": _counts(subset, lambda r: r["symbol"]),
            "by_direction": _counts(subset, lambda r: "long" if r["direction"] == 1 else "short"),
            "by_half_year": _counts(
                subset, lambda r: f"{r['event_time'].year}H{1 if r['event_time'].month <= 6 else 2}"
            ),
            "distinct_symbol_month_blocks": len({
                (r["symbol"], r["event_time"].year, r["event_time"].month) for r in subset
            }),
            "l3_l4_overlap": l3_l4,
        }

    return {
        "status": "OUTCOME_BLIND_COUNTS_ONLY",
        "events_total": len(rows),
        "by_period": by_period,
        "notice": (
            "Counts only. No returns, costs, controls, confidence intervals or validation "
            "performance have been loaded or calculated. Sample thresholds remain provisional."
        ),
    }


def evaluate_provisional_sample_gate(
    summary: Mapping[str, Any],
    minimums: Mapping[str, int],
) -> dict[str, Any]:
    """Evaluate caller-supplied thresholds without changing or inferring them.

    Expected minimum keys:
      development_total, validation_total,
      development_dose_3_4, validation_dose_3_4,
      development_dose_0, development_dose_1, development_dose_2.

    This is explicitly provisional until H7's sample gate is formally frozen.
    """
    required = (
        "development_total", "validation_total",
        "development_dose_3_4", "validation_dose_3_4",
        "development_dose_0", "development_dose_1", "development_dose_2",
    )
    missing = [key for key in required if key not in minimums]
    if missing:
        raise ValueError(f"missing explicit minimums: {', '.join(missing)}")

    dev = summary["by_period"]["development"]
    val = summary["by_period"]["validation"]
    actual = {
        "development_total": dev["events"],
        "validation_total": val["events"],
        "development_dose_3_4": dev["dose_3_4"],
        "validation_dose_3_4": val["dose_3_4"],
        "development_dose_0": int(dev["dose_counts"]["0"]),
        "development_dose_1": int(dev["dose_counts"]["1"]),
        "development_dose_2": int(dev["dose_counts"]["2"]),
    }
    checks = {
        key: {
            "actual": actual[key],
            "minimum": int(minimums[key]),
            "passed": actual[key] >= int(minimums[key]),
        }
        for key in required
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "status": "MEETS_PROVISIONAL_MINIMUMS" if passed else "INCONCLUSIVE",
        "checks": checks,
        "notice": "This is a sample-count screen, not evidence of predictive value or profitability.",
    }
