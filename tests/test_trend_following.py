import numpy as np
import pandas as pd

from backend.strategies.trend_following import TrendFollowing


def make_df(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0.00005, 0.001, n)))
    high = close + np.abs(rng.normal(0, 0.0005, n))
    low = close - np.abs(rng.normal(0, 0.0005, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_signal_values_valid():
    s = TrendFollowing().generate(make_df())
    assert set(s["signal"].unique()) <= {-1, 0, 1}
    has = s["signal"] != 0
    assert (s.loc[has, "sl_dist"] > 0).all()


def test_no_signals_during_warmup():
    s = TrendFollowing().generate(make_df())
    assert (s["signal"].iloc[:200] == 0).all()


def test_no_lookahead():
    """Signals on the first 1000 bars must not change when later bars exist."""
    df = make_df()
    full = TrendFollowing().generate(df).iloc[:1000]
    part = TrendFollowing().generate(df.iloc[:1000])
    assert (full["signal"].values == part["signal"].values).all()
