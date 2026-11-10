import json

import numpy as np
import pandas as pd
import pytest

from backend.backtesting import backtest_engine
from backend.backtesting.backtest_engine import (BacktestConfig, backtest, count_swap_nights,
                                                 load_swap)

NO_COST = BacktestConfig(slippage_atr=0.0)
POINT = 0.0001
FLAT = (1.0, 1.001, 0.999, 1.0)


def ts(text):
    return pd.Timestamp(text, tz="UTC")


def test_single_night_counts_once():
    assert count_swap_nights(ts("2026-01-05 10:00"), ts("2026-01-06 10:00")) == 1      # Mon -> Tue


def test_wednesday_rollover_counts_triple():
    assert count_swap_nights(ts("2026-01-07 10:00"), ts("2026-01-08 10:00")) == 3      # Wed -> Thu


def test_friday_rollover_counts_once_and_weekend_nights_are_free():
    assert count_swap_nights(ts("2026-01-09 10:00"), ts("2026-01-12 10:00")) == 1      # Fri -> Mon


def test_no_rollover_crossed_means_no_swap():
    assert count_swap_nights(ts("2026-01-05 08:00"), ts("2026-01-05 20:00")) == 0


def test_rollover_boundary_is_included_at_the_end_not_the_start():
    assert count_swap_nights(ts("2026-01-05 10:00"), ts("2026-01-05 22:00")) == 1
    assert count_swap_nights(ts("2026-01-05 22:00"), ts("2026-01-06 10:00")) == 0


def long_trade_rows():
    # signal on bar 0, entry bar 1 (Mon 01:00); take-profit reached 25 bars later (Tue 01:00), crossing Mon 22:00
    return [FLAT] * 25 + [(1.0, 1.021, 0.999, 1.02)]


def run(direction, swap):
    n = len(long_trade_rows())
    times = pd.date_range("2026-01-05 00:00", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(long_trade_rows(), columns=["open", "high", "low", "close"])
    df.insert(0, "time", times)
    df["spread"] = 0
    sig = pd.DataFrame({"signal": 0, "sl_dist": np.nan, "rr": np.nan, "max_hold": 0, "reason": ""}, index=range(n))
    sig.loc[0, ["signal", "sl_dist", "rr", "max_hold"]] = [direction, 0.01, 2.0, 40]
    sig["signal"] = sig["signal"].astype(int)
    return backtest(df, sig, POINT, NO_COST, atr=pd.Series(0.001, index=df.index), swap=swap)


def test_swap_cost_reduces_r_for_the_charged_side():
    swap = {"long_pts": -10.0, "short_pts": 0.0, "triple_weekday": 2}
    t = run(1, swap)
    assert t.iloc[0]["nights"] == 1
    assert abs(t.iloc[0]["swap_r"] + 0.1) < 1e-9          # 10 points = 0.001 price; risk 0.01 -> -0.1R
    assert abs(t.iloc[0]["r"] - 1.9) < 1e-9


def test_swap_does_not_touch_the_free_side_or_missing_model():
    swap = {"long_pts": -10.0, "short_pts": 0.0, "triple_weekday": 2}
    short_rows = [FLAT] * 25 + [(1.0, 1.001, 0.979, 0.98)]
    n = len(short_rows)
    df = pd.DataFrame(short_rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", pd.date_range("2026-01-05 00:00", periods=n, freq="h", tz="UTC"))
    df["spread"] = 0
    sig = pd.DataFrame({"signal": 0, "sl_dist": np.nan, "rr": np.nan, "max_hold": 0, "reason": ""}, index=range(n))
    sig.loc[0, ["signal", "sl_dist", "rr", "max_hold"]] = [-1, 0.01, 2.0, 40]
    sig["signal"] = sig["signal"].astype(int)
    atr = pd.Series(0.001, index=df.index)
    with_swap = backtest(df, sig, POINT, NO_COST, atr=atr, swap=swap)
    without = backtest(df, sig, POINT, NO_COST, atr=atr)
    assert abs(with_swap.iloc[0]["r"] - 2.0) < 1e-9 and abs(without.iloc[0]["r"] - 2.0) < 1e-9
    assert without.iloc[0]["nights"] == 0 and without.iloc[0]["swap_r"] == 0


def test_positive_swap_is_a_credit():
    swap = {"long_pts": 10.0, "short_pts": 0.0, "triple_weekday": 2}
    assert abs(run(1, swap).iloc[0]["r"] - 2.1) < 1e-9


def test_load_swap_converts_triple_day_and_refuses_unknown_modes(tmp_path, monkeypatch):
    path = tmp_path / "swaps.json"
    path.write_text(json.dumps({
        "EURUSD": {"swap_long": -5.6, "swap_short": 0.0, "swap_mode": 1, "triple_day": 3},
        "XAUUSD": {"swap_long": -1.0, "swap_short": 0.0, "swap_mode": 2, "triple_day": 3},
    }))
    monkeypatch.setattr(backtest_engine, "SWAP_INFO_PATH", path)
    eur = load_swap("EURUSD")
    assert eur["long_pts"] == -5.6 and eur["triple_weekday"] == 2          # MT5 Wednesday=3 -> pandas Wednesday=2
    with pytest.raises(NotImplementedError):
        load_swap("XAUUSD")
