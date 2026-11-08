import numpy as np
import pandas as pd

from backend.backtesting.backtest_engine import BacktestConfig, backtest
from backend.backtesting.performance import summarize, wilson_interval

POINT = 0.0001
NO_COST = BacktestConfig(slippage_atr=0.0)


def candles(rows, spread=0, step_hours=1):
    """rows = list of (open, high, low, close)."""
    n = len(rows)
    times = pd.date_range("2026-01-05 00:00", periods=n, freq=f"{step_hours}h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "time", times)
    df["spread"] = spread
    return df


def one_signal(n, idx, direction, sl_dist=0.01, rr=2.0, max_hold=10):
    s = pd.DataFrame({
        "signal": 0, "sl_dist": np.nan, "rr": np.nan, "max_hold": 0, "reason": "",
    }, index=range(n))
    s.loc[idx, ["signal", "sl_dist", "rr", "max_hold", "reason"]] = [direction, sl_dist, rr, max_hold, "test"]
    s["signal"] = s["signal"].astype(int)
    return s


def run(rows, direction, cfg=NO_COST, spread=0, **kw):
    df = candles(rows, spread=spread)
    sig = one_signal(len(df), 0, direction, **kw)
    atr = pd.Series(0.001, index=df.index)
    return backtest(df, sig, POINT, cfg, atr=atr)


FLAT = (1.0, 1.001, 0.999, 1.0)


def test_buy_hits_take_profit():
    t = run([FLAT, (1.0, 1.001, 0.999, 1.0), (1.0, 1.021, 0.999, 1.02)], 1)
    assert len(t) == 1 and t.iloc[0]["outcome"] == "tp"
    assert abs(t.iloc[0]["r"] - 2.0) < 1e-9


def test_buy_hits_stop():
    t = run([FLAT, (1.0, 1.001, 0.989, 0.99)], 1)
    assert t.iloc[0]["outcome"] == "sl"
    assert abs(t.iloc[0]["r"] + 1.0) < 1e-9


def test_both_touched_same_candle_is_a_loss():
    t = run([FLAT, (1.0, 1.03, 0.98, 1.0)], 1)
    assert t.iloc[0]["outcome"] == "sl"


def test_sell_hits_take_profit():
    t = run([FLAT, (1.0, 1.001, 0.979, 0.98)], -1)
    assert t.iloc[0]["outcome"] == "tp"
    assert abs(t.iloc[0]["r"] - 2.0) < 1e-9


def test_entry_is_next_candle_open():
    df = candles([FLAT, (1.005, 1.006, 1.004, 1.005), (1.005, 1.03, 1.004, 1.03)])
    sig = one_signal(len(df), 0, 1)
    t = backtest(df, sig, POINT, NO_COST, atr=pd.Series(0.001, index=df.index))
    assert t.iloc[0]["entry_time"] == df["time"].iloc[1]
    assert abs(t.iloc[0]["entry"] - 1.005) < 1e-9


def test_spread_reduces_result():
    t = run([FLAT, (1.0, 1.001, 0.999, 1.0), (1.0, 1.021, 0.999, 1.02)], 1, spread=10)
    expected = (0.02 - 0.001) / 0.011
    assert abs(t.iloc[0]["r"] - expected) < 1e-9


def test_time_exit():
    rows = [FLAT] + [(1.0, 1.002, 0.998, 1.001)] * 4
    t = run(rows, 1, max_hold=2)
    assert t.iloc[0]["outcome"] == "time" and t.iloc[0]["bars"] == 2


def test_no_overlapping_trades():
    df = candles([FLAT] * 3 + [(1.0, 1.001, 0.999, 1.0)] * 3)
    sig = one_signal(len(df), 0, 1, max_hold=5)
    sig.loc[1, ["signal", "sl_dist", "rr", "max_hold"]] = [1, 0.01, 2.0, 5]
    t = backtest(df, sig, POINT, NO_COST, atr=pd.Series(0.001, index=df.index))
    assert len(t) == 1


def test_weekend_gap_is_skipped():
    df = candles([FLAT, (1.0, 1.03, 0.999, 1.03)], step_hours=48)
    sig = one_signal(len(df), 0, 1)
    t = backtest(df, sig, POINT, NO_COST, atr=pd.Series(0.001, index=df.index))
    assert len(t) == 0


def test_wide_spread_is_skipped():
    t = run([FLAT, (1.0, 1.001, 0.999, 1.0), (1.0, 1.03, 0.999, 1.03)], 1, spread=500)
    assert len(t) == 0


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert abs(lo - 0.4038) < 0.001 and abs(hi - 0.5962) < 0.001


def test_summary_numbers():
    trades = pd.DataFrame({
        "r": [1.0, -1.0, 2.0, -1.0, -1.0],
        "exit_time": pd.date_range("2026-01-01", periods=5, freq="D"),
        "outcome": ["tp", "sl", "tp", "sl", "sl"],
    })
    s = summarize(trades)
    assert s["trades"] == 5 and s["wins"] == 2 and s["losses"] == 3
    assert abs(s["profit_factor"] - 1.0) < 1e-9
    assert abs(s["max_dd_r"] - 2.0) < 1e-9
    assert s["max_win_streak"] == 1 and s["max_loss_streak"] == 2
    assert abs(s["net_r"]) < 1e-9
