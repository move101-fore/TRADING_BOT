from datetime import datetime, timezone

import pytest

from backend.research.h7_diagnostics import (
    evaluate_provisional_sample_gate,
    summarize_events,
)

UTC = timezone.utc


def event(symbol="EURUSD", parent_day=2, event_time=None, period="development",
          direction=1, layers=(False, False, False, False)):
    event_time = event_time or datetime(2022, 1, parent_day, 10, 0, tzinfo=UTC)
    return {
        "symbol": symbol,
        "parent_time": datetime(2022, 1, parent_day, 8, 0, tzinfo=UTC),
        "event_time": event_time,
        "period": period,
        "direction": direction,
        "L1_liquidity": layers[0],
        "L2_support_resistance": layers[1],
        "L3_structure": layers[2],
        "L4_fvg": layers[3],
    }


def test_counts_dose_and_l3_l4_overlap_without_returns():
    rows = [
        event(parent_day=2, layers=(False, False, False, False)),
        event(parent_day=3, layers=(True, False, True, False)),
        event(parent_day=4, layers=(True, True, False, True)),
        event(parent_day=5, layers=(True, True, True, True)),
    ]
    result = summarize_events(rows)
    dev = result["by_period"]["development"]
    assert dev["dose_counts"] == {"0": 1, "1": 0, "2": 1, "3": 1, "4": 1}
    assert dev["l3_l4_overlap"] == {
        "neither": 1, "L3_only": 1, "L4_only": 1, "both": 1
    }
    assert result["status"] == "OUTCOME_BLIND_COUNTS_ONLY"
    assert "returns" not in dev


def test_reports_symbol_month_blocks_half_year_and_direction():
    rows = [
        event(parent_day=2),
        event(parent_day=3, event_time=datetime(2022, 2, 3, 10, tzinfo=UTC),
              parent_day=3, symbol="GBPUSD", direction=-1),
    ]
    dev = summarize_events(rows)["by_period"]["development"]
    assert dev["distinct_symbol_month_blocks"] == 2
    assert dev["by_half_year"] == {"2022H1": 2}
    assert dev["by_direction"] == {"long": 1, "short": 1}


def test_duplicate_symbol_parent_direction_is_rejected():
    row = event()
    with pytest.raises(ValueError, match="duplicate H7 event key"):
        summarize_events([row, row])


def test_holdout_period_is_rejected():
    row = event(period="validation", event_time=datetime(2025, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="registered validation period"):
        summarize_events([row])


def test_holdout_label_is_rejected():
    with pytest.raises(ValueError, match="period must be development or validation"):
        summarize_events([event(period="holdout")])


def test_non_utc_timestamp_is_rejected():
    row = event()
    row["event_time"] = datetime(2022, 1, 2, 10, tzinfo=timezone.utc).astimezone(
        __import__("datetime").timezone(__import__("datetime").timedelta(hours=3))
    )
    with pytest.raises(ValueError, match="must be UTC"):
        summarize_events([row])


def test_unknown_layer_is_not_silently_counted_as_false():
    row = event()
    row["L4_fvg"] = None
    with pytest.raises(ValueError, match="unknown is not False"):
        summarize_events([row])


def test_provisional_sample_gate_requires_explicit_thresholds():
    summary = summarize_events([event()])
    with pytest.raises(ValueError, match="missing explicit minimums"):
        evaluate_provisional_sample_gate(summary, {})


def test_provisional_sample_gate_does_not_claim_profitability():
    summary = summarize_events([event()])
    mins = {
        "development_total": 1,
        "validation_total": 0,
        "development_dose_3_4": 0,
        "validation_dose_3_4": 0,
        "development_dose_0": 1,
        "development_dose_1": 0,
        "development_dose_2": 0,
    }
    result = evaluate_provisional_sample_gate(summary, mins)
    assert result["status"] == "MEETS_PROVISIONAL_MINIMUMS"
    assert "profitability" in result["notice"]
