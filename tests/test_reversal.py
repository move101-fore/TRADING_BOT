import numpy as np
import pandas as pd

from backend.strategies.reversal import Reversal, ReversalConfig


def make_df(n=4000, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wave = 0.006 * np.sin(t / 25.0) + 0.004 * np.sin(t / 61.0)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.0006, n)) * 0.5 + wave)
    high = close + np.abs(rng.normal(0, 0.0004, n))
    low = close - np.abs(rng.normal(0, 0.0004, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def mirror(df):
    m = pd.DataFrame({
        "open": 2.2 - df["open"],
        "high": 2.2 - df["low"],
        "low": 2.2 - df["high"],
        "close": 2.2 - df["close"],
    })
    return m


def test_signal_values_valid():
    s = Reversal().generate(make_df())
    assert set(s["signal"].unique()) <= {-1, 0, 1}
    has = s["signal"] != 0
    assert (s.loc[has, "sl_dist"] > 0).all()


def test_produces_some_signals():
    s = Reversal().generate(make_df())
    assert (s["signal"] != 0).sum() > 0


def test_no_signals_during_warmup():
    s = Reversal().generate(make_df())
    assert (s["signal"].iloc[:30] == 0).all()


def test_no_lookahead():
    df = make_df()
    full = Reversal().generate(df).iloc[:3000]
    part = Reversal().generate(df.iloc[:3000])
    assert (full["signal"].values == part["signal"].values).all()


def test_mirror_symmetry():
    """A market flipped upside down must give the opposite signals."""
    df = make_df()
    a = Reversal().generate(df)["signal"].to_numpy()
    b = Reversal().generate(mirror(df))["signal"].to_numpy()
    assert (a == -b).mean() > 0.999
