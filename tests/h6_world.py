"""Synthetic M15 world with one hand-built H6 setup, used by the tests (not a test file itself)."""
import numpy as np
import pandas as pd

PARENT_IDX = 3 * 96 + 32           # Thursday 08:00: first candle of the parent H4 window
BASE = 1.0900


def baseline(n, base=BASE):
    close = base + np.where(np.arange(n) % 2 == 0, 0.00004, -0.00004)
    open_ = np.concatenate([[base], close[:-1]])
    high = np.maximum(open_, close) + 0.00003
    low = np.minimum(open_, close) - 0.00003
    return np.column_stack([open_, high, low, close])


def planted_short(n_days=8, start="2022-01-03 00:00", extra_events=()):
    """
    Candles as an (n, 4) array of open/high/low/close. Contains exactly one fully-qualified SHORT setup:
    previous-day high 1.10002 next to a parent high of exactly 1.1000 (a round number), a sweep + reclaim,
    a structure break, a bearish FVG and a retest. Returns (dataframe, indexes of the key candles).
    """
    n = n_days * 96
    c = baseline(n)
    # previous-day high (Wednesday 10:00)
    spike = 2 * 96 + 40
    c[spike] = [1.0900, 1.10002, 1.0898, 1.0901]
    # parent H4 window: a climb to a high of exactly 1.1000
    p = PARENT_IDX
    step = (1.0986 - 1.0900) / 14
    for k in range(14):
        o = 1.0900 + step * k
        cl = o + step
        c[p + k] = [o, cl + 0.0001, o - 0.0001, cl]
    c[p + 14] = [1.0986, 1.0994, 1.0986, 1.0992]
    c[p + 15] = [1.0992, 1.1000, 1.0990, 1.0998]
    s0 = p + 16
    plan = [
        [1.0998, 1.0999, 1.0990, 1.0992],   # s0
        [1.0992, 1.0996, 1.0985, 1.0994],   # s1  swing low (the structure reference)
        [1.0994, 1.0999, 1.0988, 1.0997],   # s2
        [1.0997, 1.1006, 1.0996, 1.1002],   # s3  breach of the parent high
        [1.1002, 1.1008, 1.0993, 1.0994],   # s4  reclaim = the sweep candle s (also c1 of the FVG)
        [1.0993, 1.0993, 1.0978, 1.0988],   # s5  displacement (c2), still above the structure reference
        [1.0984, 1.0988, 1.0960, 1.0963],   # s6  structure break k (c3 of the FVG, gap down)
        [1.0963, 1.0975, 1.0958, 1.0970],   # s7
        [1.0970, 1.0992, 1.0968, 1.0989],   # s8  retest of the FVG midpoint, closes below it = e
    ]
    for j, row in enumerate(plan):
        c[s0 + j] = row
    # price continues from the last candle
    tail = baseline(n - (s0 + 9), base=1.0989)
    c[s0 + 9:] = tail
    times = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame({"time": times, "open": c[:, 0], "high": c[:, 1], "low": c[:, 2], "close": c[:, 3], "spread": 8})
    keys = {"parent": p, "s": s0 + 4, "k": s0 + 6, "e": s0 + 8, "entry": s0 + 9}
    return df, keys


def to_h1(m15):
    t = m15.set_index("time")
    h1 = t.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "spread": "max"}).dropna()
    return h1.reset_index()


def flip(df):
    m = df.copy()
    m["open"], m["close"] = 2.2 - df["open"], 2.2 - df["close"]
    m["high"], m["low"] = 2.2 - df["low"], 2.2 - df["high"]
    return m
