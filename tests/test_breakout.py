import numpy as np
import pandas as pd

from backend.strategies.breakout import Breakout, BreakoutConfig


def make_df(n=3000, seed=2):
    rng = np.random.default_rng(seed)
    # Regime-switching volatility so tight ranges and breakouts occur.
    vol = np.where((np.arange(n) // 120) % 2 == 0, 0.0003, 0.0012)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 1, n) * vol))
    high = close + np.abs(rng.normal(0, 0.0004, n))
    low = close - np.abs(rng.normal(0, 0.0004, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_signal_values_valid():
    s = Breakout().generate(make_df())
    assert set(s["signal"].unique()) <= {-1, 0, 1}
    has = s["signal"] != 0
    assert (s.loc[has, "sl_dist"] > 0).all()


def test_no_signals_during_warmup():
    s = Breakout().generate(make_df())
    assert (s["signal"].iloc[:30] == 0).all()


def test_no_lookahead():
    df = make_df()
    for cfg in (BreakoutConfig(), BreakoutConfig(require_retest=False)):
        full = Breakout(cfg).generate(df).iloc[:2000]
        part = Breakout(cfg).generate(df.iloc[:2000])
        assert (full["signal"].values == part["signal"].values).all()


def test_retest_mode_is_stricter():
    df = make_df()
    strict = int((Breakout(BreakoutConfig(require_retest=True)).generate(df).signal != 0).sum())
    loose = int((Breakout(BreakoutConfig(require_retest=False)).generate(df).signal != 0).sum())
    assert strict <= loose
