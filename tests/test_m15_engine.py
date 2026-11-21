import numpy as np
import pandas as pd

from backend.research import event_study as es
from backend.research.event_study import (SymbolData, completed_atr_reference, gap_flags, study)

POINT = 0.0001
M15_HORIZONS = (24, 48, 96)


def make_m15(n=20000, seed=3, start="2022-01-03 00:00", vol=0.0003):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, vol, n)) * 0.6 + 0.003 * np.sin(t / 120.0))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, vol / 2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol / 2, n))
    times = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close, "spread": 8})


def make_h1_from(m15):
    t = m15.set_index("time")
    h1 = t.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "spread": "max"}).dropna()
    return h1.reset_index()


def test_m15_settings_scale_with_the_candle_size():
    sd = SymbolData(make_m15(6000), POINT, bar_minutes=15, horizons=M15_HORIZONS)
    assert sd.horizons == M15_HORIZONS and sd.moves.shape == (sd.n, 2, 3)
    assert sd.exclude_radius == 96 and sd.control_window == 1440


def test_h1_defaults_are_unchanged():
    from tests.test_research import make_h1
    sd = SymbolData(make_h1(4000), POINT)
    assert sd.horizons == es.HORIZONS and sd.exclude_radius == 24 and sd.control_window == 360


def test_m15_vectorised_moves_match_the_reference_with_swap():
    sd = SymbolData(make_m15(12000, seed=5), POINT, bar_minutes=15, horizons=M15_HORIZONS,
                    swap={"long_pts": -30.0, "short_pts": -12.0, "triple_weekday": 2})
    idx = np.flatnonzero(sd.valid)[::53]
    dirs = np.where(np.arange(len(idx)) % 2 == 0, 1, -1)
    assert np.allclose(sd.net_moves(idx, dirs), sd.net_moves_reference(idx, dirs), equal_nan=True)


def test_outcome_atr_reference_is_used_for_normalisation_and_slippage():
    df = make_m15(4000, seed=7)
    df["atr_14"] = 0.0005
    ref = np.full(len(df), 0.002)
    own = SymbolData(df, POINT, bar_minutes=15, horizons=M15_HORIZONS)
    out = SymbolData(df, POINT, bar_minutes=15, horizons=M15_HORIZONS, atr_ref=ref)
    i = int(np.flatnonzero(own.valid)[100])
    e, x = i + 1, i + 24
    gross = df["close"].iloc[x] - df["open"].iloc[e]
    want = (gross - 8 * POINT - 2 * es.SLIPPAGE_ATR * 0.002) / 0.002
    assert abs(out.net_moves(np.array([i]), np.array([1]))[0, 0] - want) < 1e-9
    assert abs(own.net_moves(np.array([i]), np.array([1]))[0, 0] - want) > 1e-3


def test_completed_atr_reference_never_uses_an_unfinished_h1_candle():
    m15 = make_m15(2000, seed=2)
    h1 = make_h1_from(m15)
    base = completed_atr_reference(m15["time"], 15, h1)
    from backend.market_data.indicators import atr
    h1_atr = atr(h1, 14).to_numpy()
    h1_time = pd.to_datetime(h1["time"], utc=True)
    # M15 candle opening 05:45 closes at 06:00, exactly when the 05:00 H1 candle completes -> uses it
    t_open = pd.Timestamp("2022-01-05 05:45", tz="UTC")
    i = int(np.flatnonzero(m15["time"] == t_open)[0])
    assert abs(base[i] - h1_atr[int(np.flatnonzero(h1_time == pd.Timestamp("2022-01-05 05:00", tz="UTC"))[0])]) < 1e-12
    # M15 candle opening 05:30 closes at 05:45, before the 05:00 H1 candle completes -> still the 04:00 candle
    j = int(np.flatnonzero(m15["time"] == pd.Timestamp("2022-01-05 05:30", tz="UTC"))[0])
    assert abs(base[j] - h1_atr[int(np.flatnonzero(h1_time == pd.Timestamp("2022-01-05 04:00", tz="UTC"))[0])]) < 1e-12


def test_changing_future_h1_data_cannot_change_past_alignment():
    m15 = make_m15(2000, seed=2)
    h1 = make_h1_from(m15)
    a = completed_atr_reference(m15["time"], 15, h1)
    h1b = h1.copy()
    cut = int(np.flatnonzero(pd.to_datetime(h1b["time"], utc=True) >= pd.Timestamp("2022-01-04 12:00", tz="UTC"))[0])
    h1b.loc[cut:, ["open", "high", "low", "close"]] *= 1.5
    b = completed_atr_reference(m15["time"], 15, h1b)
    safe = (m15["time"] < pd.Timestamp("2022-01-04 11:30", tz="UTC")).to_numpy()   # decided before the changed candle completed
    assert np.allclose(a[safe], b[safe], equal_nan=True)


def test_gap_threshold_is_configurable():
    times = list(pd.date_range("2024-01-02 00:00", "2024-01-02 06:00", freq="15min", tz="UTC"))
    times += list(pd.date_range("2024-01-02 08:00", "2024-01-02 10:00", freq="15min", tz="UTC"))    # a 2h hole
    assert gap_flags(times, gap_hours=1.0).sum() == 1 and gap_flags(times, gap_hours=3.0).sum() == 0


def test_study_runs_on_m15_and_finds_a_planted_effect():
    n = 40000
    rng = np.random.default_rng(8)
    vol = 0.0003
    r = rng.normal(0, vol, n)
    cells = np.arange(500, n - 400, 240)
    idx = np.sort(cells + rng.integers(0, 240, len(cells)))
    for i in idx:
        r[i + 1: i + 49] += 0.04 * vol * 4                     # drift after each event (48 bars)
    close = pd.Series(1.10 + np.cumsum(r))
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({"time": pd.date_range("2022-01-03", periods=n, freq="15min", tz="UTC"),
                       "open": open_, "close": close,
                       "high": np.maximum(open_, close) + np.abs(rng.normal(0, vol / 2, n)),
                       "low": np.minimum(open_, close) - np.abs(rng.normal(0, vol / 2, n)), "spread": 8})
    sd = {"EURUSD": SymbolData(df, POINT, bar_minutes=15, horizons=M15_HORIZONS)}
    ev = pd.DataFrame({"symbol": "EURUSD", "idx": idx, "time": df["time"].iloc[idx].to_numpy(),
                       "kind": "PLANT", "direction": 1})
    res = study(ev, sd, n_boot=300, periods={"development": es.PERIODS["development"]})
    assert list(res["horizon"]) == list(M15_HORIZONS)
    assert res[res["horizon"] == 48].iloc[0]["excess_lo"] > 0
