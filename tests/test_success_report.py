import numpy as np
import pandas as pd

from backend.backtesting.success_report import (breakeven_win_rate, mean_ci, random_signals,
                                                success_table)
from backend.backtesting.run_backtest import TRAIN_END


def test_breakeven_win_rate_for_two_to_one():
    r = np.array([2.0, 2.0, -1.0, -1.0, -1.0, -1.0])
    assert abs(breakeven_win_rate(r) - 1 / 3) < 1e-9


def test_mean_ci_contains_the_mean_and_shrinks_with_data():
    rng = np.random.default_rng(0)
    small = rng.normal(0.0, 1.0, 50)
    big = rng.normal(0.0, 1.0, 5000)
    lo1, hi1 = mean_ci(small)
    lo2, hi2 = mean_ci(big)
    assert lo1 < small.mean() < hi1
    assert (hi2 - lo2) < (hi1 - lo1)


def test_random_signals_respect_rules():
    n = 2000
    df = pd.DataFrame({"close": np.linspace(1, 2, n)})
    atr = np.full(n, 0.001)
    s = random_signals(df, atr, 300, np.random.default_rng(1))
    assert set(s["signal"].unique()) <= {-1, 0, 1}
    has = s["signal"] != 0
    assert has.sum() == 300
    assert (s.loc[has, "sl_dist"] == 0.0015).all() and (s.loc[has, "rr"] == 2.0).all()
    assert (s["signal"].iloc[:250] == 0).all()


def test_success_table_outcome_mix_sums_to_100():
    times = pd.date_range("2022-01-01", periods=6, freq="D", tz="UTC")
    trades = pd.DataFrame({
        "strategy": "x", "symbol": "EURUSD", "entry_time": times, "exit_time": times,
        "r": [2.0, -1.0, -1.0, 0.3, 2.0, -1.0],
        "outcome": ["tp", "sl", "sl", "time", "tp", "sl"],
    })
    t = success_table(trades)
    row = t[(t["group"] == "x") & (t["period"] == "train")].iloc[0]
    assert row["trades"] == 6
    assert abs(row["tp_pct"] + row["sl_pct"] + row["time_pct"] - 100) < 1e-9
    assert abs(row["win_pct"] - 50.0) < 1e-9
