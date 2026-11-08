import numpy as np
import pandas as pd

from backend.strategies.pullback import Pullback, PullbackConfig


def make_df(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Slow drift that flips direction, plus waves, so pullbacks occur inside trends.
    drift = 0.0004 * np.sign(np.sin(t / 700.0))
    wave = 0.004 * np.sin(t / 12.0)
    close = pd.Series(1.10 + np.cumsum(drift + rng.normal(0, 0.0006, n)) + wave)
    high = close + np.abs(rng.normal(0, 0.0004, n))
    low = close - np.abs(rng.normal(0, 0.0004, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_signal_values_valid():
    s = Pullback().generate(make_df())
    assert set(s["signal"].unique()) <= {-1, 0, 1}
    has = s["signal"] != 0
    assert (s.loc[has, "sl_dist"] > 0).all()


def test_no_signals_during_warmup():
    s = Pullback().generate(make_df())
    assert (s["signal"].iloc[:200] == 0).all()


def test_no_lookahead():
    df = make_df()
    for cfg in (PullbackConfig(), PullbackConfig(require_sr_confluence=True)):
        full = Pullback(cfg).generate(df).iloc[:3000]
        part = Pullback(cfg).generate(df.iloc[:3000])
        assert (full["signal"].values == part["signal"].values).all()


def test_sr_confluence_is_stricter():
    df = make_df()
    loose = int((Pullback(PullbackConfig()).generate(df).signal != 0).sum())
    strict = int((Pullback(PullbackConfig(require_sr_confluence=True)).generate(df).signal != 0).sum())
    assert strict <= loose
