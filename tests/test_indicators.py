import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from backend.market_data.indicators import add_indicators, rsi


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.001, n)))
    high = close + np.abs(rng.normal(0, 0.0005, n))
    low = close - np.abs(rng.normal(0, 0.0005, n))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_rsi_in_range():
    r = rsi(make_df()["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1.0, 60.0))
    assert rsi(s).iloc[-1] == 100.0


def test_atr_positive():
    out = add_indicators(make_df())
    assert (out["atr_14"].dropna() > 0).all()


def test_adx_in_range():
    out = add_indicators(make_df())
    a = out["adx"].dropna()
    assert ((a >= 0) & (a <= 100)).all()


def test_no_lookahead():
    """Indicators on the first 250 bars must be identical whether or not
    later bars exist. If they differ, the code is peeking at the future."""
    df = make_df()
    full = add_indicators(df)
    part = add_indicators(df.iloc[:250])
    assert_frame_equal(full.iloc[:250], part, check_exact=False, rtol=1e-9)
