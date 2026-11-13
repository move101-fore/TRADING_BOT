import numpy as np
import pandas as pd

from backend.market_data.indicators import add_indicators
from backend.research import event_study as es
from backend.backtesting.backtest_engine import count_swap_nights
from backend.research.event_study import (SymbolData, add_placebo_pvalues, block_bootstrap_ci, control_pools,
                                          gap_flags, nearest_controls, placebo_calibration, placebo_events,
                                          study, swap_nights_vector)
from backend.research.verdict import STATUS_ELIGIBLE, STATUS_FAIL, verdict
from backend.research.structure import structure_events

POINT = 0.0001


def make_h1(n=6000, seed=1, start="2022-01-03 00:00", vol=0.0006):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    wave = 0.004 * np.sin(t / 30.0) + 0.003 * np.sin(t / 90.0)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, vol, n)) * 0.6 + wave)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, vol / 2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol / 2, n))
    times = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close, "spread": 8})


def flip(df):
    m = df.copy()
    m["open"], m["close"] = 2.2 - df["open"], 2.2 - df["close"]
    m["high"], m["low"] = 2.2 - df["low"], 2.2 - df["high"]
    return m


# ---------------------------------------------------------------- structure
def test_structure_events_are_valid_and_fire_once_per_level():
    ev = structure_events(make_h1())
    assert len(ev) > 20
    assert set(ev["kind"]) <= {"BOS", "CHoCH"} and set(ev["direction"]) <= {-1, 1}
    assert not ev.duplicated(subset=["kind", "direction", "level"]).any()
    bos = ev[ev["kind"] == "BOS"]
    assert (bos["direction"] == bos["trend"]).all()
    choch = ev[ev["kind"] == "CHoCH"]
    assert (choch["direction"] == -choch["trend"]).all()


def test_structure_events_have_no_lookahead():
    df = make_h1(4000)
    full = structure_events(df)
    part = structure_events(df.iloc[:3000].copy())
    cut = full[full["idx"] < 3000].reset_index(drop=True)
    assert len(cut) == len(part)
    assert (cut["idx"].values == part["idx"].values).all() and (cut["kind"].values == part["kind"].values).all()


def test_structure_events_mirror_symmetry():
    df = make_h1(4000)
    a = structure_events(df)
    b = structure_events(flip(df))
    assert len(a) == len(b)
    assert (a["idx"].values == b["idx"].values).all()
    assert (a["kind"].values == b["kind"].values).all()
    assert (a["direction"].values == -b["direction"].values).all()


# ---------------------------------------------------------------- gaps, holdout, costs
def test_gap_flags_ignore_weekends_but_catch_real_gaps():
    times = list(pd.date_range("2024-01-01 00:00", "2024-01-05 20:00", freq="h", tz="UTC"))
    times += list(pd.date_range("2024-01-07 22:00", "2024-01-08 20:00", freq="h", tz="UTC"))
    times += list(pd.date_range("2024-01-09 06:00", "2024-01-09 10:00", freq="h", tz="UTC"))
    flags = gap_flags(times)
    assert flags.sum() == 1 and flags[-5]                      # only the 10-hour hole on Tuesday


def test_holdout_data_never_enters_the_research_module():
    df = make_h1(30000, start="2024-06-01 00:00")
    sd = SymbolData(df, POINT)
    assert sd.t.max() < es.RESEARCH_END


def flat_data(n=400):
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"time": times, "open": 1.0, "high": 1.001, "low": 0.999, "close": 1.0,
                       "spread": 10, "atr_14": 0.001})
    return df


def test_net_move_matches_hand_calculation():
    df = flat_data()
    df.loc[301, "open"] = 1.0000                       # entry candle (signal on 300)
    df.loc[306, "close"] = 1.0020                      # exit candle for a 6-bar hold: 301 + 6 - 1
    sd = SymbolData(df, POINT)
    got = sd.net_moves(np.array([300]), np.array([1]))[0, 0]
    gross = 1.0020 - 1.0000
    cost = 10 * POINT + 2 * es.SLIPPAGE_ATR * 0.001
    assert abs(got - (gross - cost) / 0.001) < 1e-9
    short = sd.net_moves(np.array([300]), np.array([-1]))[0, 0]
    assert abs(short - (-gross - cost) / 0.001) < 1e-9


def test_swap_is_charged_on_the_costly_side_only():
    df = flat_data()
    swap = {"long_pts": -50.0, "short_pts": 0.0, "triple_weekday": 2}
    sd0, sd1 = SymbolData(df, POINT), SymbolData(df, POINT, swap=swap)
    hours = pd.to_datetime(df["time"]).dt.hour.to_numpy()
    i = int(np.flatnonzero((hours == 20) & (np.arange(len(df)) >= 260))[0])     # after warm-up; trade crosses 22:00
    long0, long1 = sd0.net_moves(np.array([i]), np.array([1]))[0, 0], sd1.net_moves(np.array([i]), np.array([1]))[0, 0]
    short0, short1 = sd0.net_moves(np.array([i]), np.array([-1]))[0, 0], sd1.net_moves(np.array([i]), np.array([-1]))[0, 0]
    assert long1 < long0 and abs(short1 - short0) < 1e-12


def test_spread_uses_entry_candle_for_longs_and_exit_candle_for_shorts():
    df = flat_data()
    df.loc[301, "spread"] = 40                          # entry candle for a signal on 300
    df.loc[306, "spread"] = 100                         # exit candle of a 6-bar hold
    sd = SymbolData(df, POINT)
    slip = 2 * es.SLIPPAGE_ATR * 0.001
    long = sd.net_moves(np.array([300]), np.array([1]))[0, 0]
    short = sd.net_moves(np.array([300]), np.array([-1]))[0, 0]
    assert abs(long - (-(40 * POINT + slip) / 0.001)) < 1e-9
    assert abs(short - (-(100 * POINT + slip) / 0.001)) < 1e-9


def test_vectorised_moves_match_the_slow_reference_including_swap():
    df = make_h1(9000, seed=4)
    swap = {"long_pts": -30.0, "short_pts": -12.0, "triple_weekday": 2}
    sd = SymbolData(df, POINT, swap=swap)
    idx = np.flatnonzero(sd.valid)[::37]
    dirs = np.where(np.arange(len(idx)) % 2 == 0, 1, -1)
    assert np.allclose(sd.net_moves(idx, dirs), sd.net_moves_reference(idx, dirs), equal_nan=True)


def test_swap_night_vector_matches_the_scalar_function():
    rng = np.random.default_rng(1)
    base = np.datetime64("2024-01-01T00:00")
    start = base + (rng.integers(0, 24 * 60, 300)).astype("timedelta64[h]")
    end = start + (rng.integers(1, 30, 300)).astype("timedelta64[h]")
    got = swap_nights_vector(start, end, 2)
    for s_, e_, g in zip(start, end, got):
        want = count_swap_nights(pd.Timestamp(s_), pd.Timestamp(e_), 22, 2)
        assert g == want


def test_cost_multiplier_makes_every_trade_worse():
    df = make_h1(4000, seed=2)
    a, b = SymbolData(df, POINT), SymbolData(df, POINT, cost_mult=2.0)
    idx = np.flatnonzero(a.valid)[::50]
    assert (b.net_moves(idx, np.ones(len(idx))) < a.net_moves(idx, np.ones(len(idx)))).all()


# ---------------------------------------------------------------- controls and bootstrap
def test_controls_share_hour_and_volatility_and_avoid_events():
    sd = SymbolData(make_h1(6000), POINT)
    ev_idx = np.arange(400, 5000, 97)
    pools = control_pools(sd, ev_idx, {"development": es.PERIODS["development"]})
    assert pools
    near = set()
    for i in ev_idx:
        near.update(range(i - es.EXCLUDE_RADIUS, i + es.EXCLUDE_RADIUS + 1))
    for (pname, key), idx in pools.items():
        assert (sd.hour[idx] * 10 + sd.bin[idx] == key).all()
        assert not (set(idx.tolist()) & near)
        assert sd.valid[idx].all()


def test_block_bootstrap_ci_behaviour():
    rng = np.random.default_rng(0)
    weeks = np.repeat(np.arange(100), 5)
    zero = rng.normal(0, 1, 500)
    m, lo, hi = block_bootstrap_ci(zero, weeks)
    assert lo < 0 < hi
    pos = rng.normal(0.8, 1, 500)
    m2, lo2, hi2 = block_bootstrap_ci(pos, weeks)
    assert lo2 > 0 and lo2 < m2 < hi2


def test_clustering_widens_the_interval():
    rng = np.random.default_rng(3)
    weeks = np.repeat(np.arange(50), 10)
    clustered = np.repeat(rng.normal(0, 1, 50), 10) + rng.normal(0, 0.1, 500)     # events in a week move together
    _, lo_c, hi_c = block_bootstrap_ci(clustered, weeks)
    independent = rng.normal(0, 1, 500)
    _, lo_i, hi_i = block_bootstrap_ci(independent, weeks)
    assert (hi_c - lo_c) > (hi_i - lo_i)


# ---------------------------------------------------------------- the study itself
def planted_market(plant, n=17000, seed=5):
    rng = np.random.default_rng(seed)
    vol = 0.0006
    r = rng.normal(0, vol, n)
    cells = np.arange(300, n - 100, 60)
    idx = np.sort(cells + rng.integers(0, 60, len(cells)))        # one event at a random spot in each 60-bar cell
    if plant:
        for i in idx:
            r[i + 1: i + 13] += 0.15 * vol * 4                    # a drift AFTER each event candle
    close = 1.10 + np.cumsum(r)
    s = pd.Series(close)
    open_ = s.shift(1).fillna(s.iloc[0])
    df = pd.DataFrame({
        "time": pd.date_range("2022-01-03", periods=n, freq="h", tz="UTC"),
        "open": open_, "close": s,
        "high": np.maximum(open_, s) + np.abs(rng.normal(0, vol / 2, n)),
        "low": np.minimum(open_, s) - np.abs(rng.normal(0, vol / 2, n)),
        "spread": 8,
    })
    return df, idx


def events_for(df, idx, kind="PLANT"):
    return pd.DataFrame({"symbol": "EURUSD", "idx": idx, "time": df["time"].iloc[idx].values,
                         "kind": kind, "direction": 1})


def test_study_detects_a_real_planted_effect():
    df, idx = planted_market(plant=True)
    sd = {"EURUSD": SymbolData(df, POINT)}
    res = study(events_for(df, idx), sd, n_boot=500, periods={"development": es.PERIODS["development"]})
    row = res[res["horizon"] == 12].iloc[0]
    assert row["excess_lo"] > 0 and row["excess_mean"] > 0.5
    assert verdict(res, min_events=100).iloc[0]["status"] == STATUS_ELIGIBLE


def test_study_finds_nothing_in_pure_noise():
    df, idx = planted_market(plant=False)
    sd = {"EURUSD": SymbolData(df, POINT)}
    res = study(events_for(df, idx), sd, n_boot=500, periods={"development": es.PERIODS["development"]})
    assert verdict(res, min_events=100).iloc[0]["status"] == STATUS_FAIL
    assert (res["excess_lo"] < 0.15).all() and (res["excess_hi"] > -0.15).all()


def test_study_reports_unmatched_and_invalid_events_instead_of_hiding_them():
    df, idx = planted_market(plant=False)
    sd = {"EURUSD": SymbolData(df, POINT)}
    res = study(events_for(df, idx), sd, n_boot=200, periods={"development": es.PERIODS["development"]})
    assert {"unmatched", "invalid"} <= set(res.columns) and (res["unmatched"] >= 0).all()


def test_controls_are_never_reused():
    pool = np.arange(100, 400, 3)
    taken = np.zeros(1000, dtype=bool)
    first = nearest_controls(pool, 250, 8, taken)
    second = nearest_controls(pool, 250, 8, taken)
    assert len(first) == 8 and len(second) == 8
    assert not set(first.tolist()) & set(second.tolist())
    assert taken[first].all() and taken[second].all()


def test_study_marks_events_unmatched_when_controls_run_out():
    df, idx = planted_market(plant=False)
    sd = {"EURUSD": SymbolData(df, POINT)}
    ev = events_for(df, idx)
    plenty = study(ev, sd, k_controls=2, n_boot=100, periods={"development": es.PERIODS["development"]})
    scarce = study(ev, sd, k_controls=400, n_boot=100, periods={"development": es.PERIODS["development"]})
    assert scarce["unmatched"].iloc[0] >= plenty["unmatched"].iloc[0]


def test_placebo_events_keep_counts_and_directions():
    df, idx = planted_market(plant=False)
    sd = {"EURUSD": SymbolData(df, POINT)}
    ev = events_for(df, idx)
    ev["direction"] = np.where(np.arange(len(ev)) % 3 == 0, -1, 1)
    fake = placebo_events(ev, sd, {"development": es.PERIODS["development"]}, np.random.default_rng(0))
    dev_ev = ev[pd.to_datetime(ev["time"], utc=True) < es.PERIODS["development"][1]]
    assert len(fake) == len(dev_ev)
    assert sorted(fake["direction"]) == sorted(dev_ev["direction"])


def test_placebo_p_value_separates_a_real_effect_from_noise():
    periods = {"development": es.PERIODS["development"]}
    df, idx = planted_market(plant=True)
    sd = {"EURUSD": SymbolData(df, POINT)}
    ev = events_for(df, idx)
    res = add_placebo_pvalues(study(ev, sd, n_boot=100, periods=periods),
                              placebo_calibration(ev, sd, periods, runs=30))
    assert (res["placebo_p"] <= 0.05).all()
    df2, idx2 = planted_market(plant=False)
    sd2 = {"EURUSD": SymbolData(df2, POINT)}
    ev2 = events_for(df2, idx2)
    res2 = add_placebo_pvalues(study(ev2, sd2, n_boot=100, periods=periods),
                               placebo_calibration(ev2, sd2, periods, runs=30))
    assert (res2["placebo_p"] > 0.05).any()


# ---------------------------------------------------------------- runner and validation gate
def test_validation_can_only_be_viewed_once(tmp_path):
    from backend.research.run_h5 import record_validation, validation_already_viewed
    log = tmp_path / "log.csv"
    assert not validation_already_viewed("H5-v1", log)
    record_validation("H5-v1", log)
    assert validation_already_viewed("H5-v1", log)
    assert not validation_already_viewed("H5-v2", log)


def test_run_study_end_to_end_on_synthetic_data():
    from backend.research.run_h5 import run_study
    df = make_h1(17000, seed=9)
    events, res, verdicts = run_study({"EURUSD": df}, {"EURUSD": POINT}, {}, {"development": es.PERIODS["development"]},
                                      n_boot=100, placebo_runs=5)
    assert set(events["kind"]) == {"BOS", "CHoCH"} and (events["symbol"] == "EURUSD").all()
    assert set(res["kind"]) == {"BOS", "CHoCH"} and set(res["horizon"]) == set(es.HORIZONS)
    assert (res["events"] > 50).all() and "placebo_p" in res.columns
    assert set(verdicts["kind"]) == {"BOS", "CHoCH"}
