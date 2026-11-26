import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from h6_world import PARENT_IDX, flip, planted_short, to_h1                     # noqa: E402

from backend.research import h6_counts as h6                                    # noqa: E402
from backend.research.h6_counts import (NewsCalendar, SymbolContext, analyse, build_h4, data_gate,          # noqa: E402
                                        eligible_swings, entry_confirm, equal_level, reference_swing, find_fvg, find_sweep, pool_confluence,
                                        run_symbol, sample_verdict, sr_zone, structure_confirm)


def ctx_for(df, symbol="EURUSD"):
    return SymbolContext(symbol, df, to_h1(df))


# ---------------------------------------------------------------- data gate (completeness against expected candles)
def week_times(weeks=3, skip_slots=None, drop_days=None, start="2022-01-03"):
    days = []
    for w in range(weeks):
        for d in range(5):
            day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=7 * w + d)
            idx = pd.date_range(day, periods=96, freq="15min")
            days.append(idx)
    times = days[0].append(days[1:])
    if skip_slots is not None:
        times = times[~times.hour.isin(skip_slots)]
    return times


def test_gate_does_not_count_a_daily_break_or_a_market_wide_closure_as_missing():
    full = week_times()
    gold = week_times(skip_slots=[22])                                       # daily one-hour maintenance break
    gap = full[~((full >= pd.Timestamp("2022-01-12", tz="UTC")) & (full < pd.Timestamp("2022-01-13", tz="UTC")))]
    closed_everywhere = pd.Timestamp("2022-01-19", tz="UTC")
    mk = lambda idx: pd.DataFrame({"time": idx[~((idx >= closed_everywhere) & (idx < closed_everywhere + pd.Timedelta(days=1)))]})
    g = data_gate({"A": mk(full), "GOLD": mk(gold), "GAPPY": mk(gap)})
    assert g["A"]["missing_pct"] < 0.5 and g["A"]["admitted"]
    assert g["GOLD"]["missing_pct"] < 0.5 and g["GOLD"]["admitted"]        # the break is part of its schedule
    assert g["GAPPY"]["missing_pct"] > 5 and not g["GAPPY"]["admitted"]    # a real missing day, only for one symbol


# ---------------------------------------------------------------- news calendar
def test_news_window_is_30_minutes_each_side_and_currency_specific():
    cal = NewsCalendar(pd.DataFrame({"event": ["NFP", "BoJ"], "currency": ["USD", "JPY"],
                                     "scheduled_utc": ["2022-01-07 13:30", "2022-01-07 03:00"]}))
    t = lambda s: np.datetime64(s)
    got = cal.excluded("EURUSD", [t("2022-01-07T13:00"), t("2022-01-07T13:30"), t("2022-01-07T14:00"),
                                  t("2022-01-07T14:15"), t("2022-01-07T12:45"), t("2022-01-07T03:00")])
    assert list(got) == [True, True, True, False, False, False]              # JPY event does not touch EURUSD
    assert list(cal.excluded("USDJPY", [t("2022-01-07T03:15")])) == [True]
    assert list(cal.excluded("XAUUSD", [t("2022-01-07T13:45")])) == [True]    # gold follows USD


def test_news_clock_offset_shifts_the_window():
    df = pd.DataFrame({"event": ["NFP"], "currency": ["USD"], "scheduled_utc": ["2022-01-07 13:30"]})
    cal = NewsCalendar(df, clock_offset_hours=2.0)                           # candle clock = UTC + 2h
    assert list(cal.excluded("EURUSD", [np.datetime64("2022-01-07T15:30")])) == [True]
    assert list(cal.excluded("EURUSD", [np.datetime64("2022-01-07T13:30")])) == [False]


# ---------------------------------------------------------------- H4 candles
def test_h4_keeps_only_complete_bars():
    t = pd.date_range("2022-01-03 00:00", periods=12, freq="h", tz="UTC").delete(5)     # hour 05:00 is missing
    h1 = pd.DataFrame({"time": t, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0})
    h4 = build_h4(h1)
    assert list(pd.to_datetime(h4["time"]).dt.hour) == [0, 8]                 # the 04:00-08:00 bar is incomplete


# ---------------------------------------------------------------- layer rules
def test_pool_confluence_types_and_no_double_counting():
    assert pool_confluence(1.1000, 0.001, None, None) == []
    assert pool_confluence(1.1000, 0.001, 1.10020, None) == ["PD"]
    assert pool_confluence(1.1000, 0.001, None, 1.09980) == ["EQ"]
    assert pool_confluence(1.1000, 0.001, 1.10025, 1.09975) == ["EQ", "PD"]          # two distinct pools
    assert pool_confluence(1.1000, 0.001, 1.10010, 1.10005) == ["PD+EQ"]              # the same price: one pool
    assert pool_confluence(1.1000, 0.001, 1.1004, None) == []                         # outside 0.3 x ATR


def test_equal_levels():
    assert equal_level([1.1000, 1.10015], 0.001, True) == 1.10015
    assert equal_level([1.1000, 1.10015], 0.001, False) == 1.1000
    assert equal_level([1.1000, 1.1003], 0.001, True) is None
    assert equal_level([1.1000], 0.001, True) is None


def test_support_resistance_zones():
    assert sr_zone("EURUSD", 1.1001, 0.002, []) == ["ROUND"]                          # within 0.25 x 0.002 of 1.1000
    assert sr_zone("EURUSD", 1.1060, 0.002, []) == []
    assert sr_zone("USDJPY", 150.10, 0.8, []) == ["ROUND"]                            # JPY step 1.00
    assert sr_zone("XAUUSD", 2000.5, 4.0, []) == ["ROUND"]                            # gold step 10.0
    swing = sr_zone("EURUSD", 1.1337, 0.002, [1.1335, 1.1339, 1.1200])
    assert swing == ["SWING"]
    assert sr_zone("EURUSD", 1.1337, 0.002, [1.1335, 1.1200]) == []                   # a single swing is not a cluster


def sweep_arrays(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]


def test_sweep_rules_and_both_sides_exclusion():
    atr = np.full(8, 0.0010)
    # columns: high, low, close.  Level 1.1000, other extreme 1.0900. Reclaim needs close <= 1.1000 - 0.0001
    h, l, c = sweep_arrays([[1.0995, 1.0950, 1.0990], [1.1010, 1.0980, 1.1005], [1.1012, 1.0990, 1.0995]])
    assert find_sweep(h, l, c, atr, 0, 3, 1.1000, 1.0900, True) == ("ok", 2)
    h, l, c = sweep_arrays([[1.0995, 1.0950, 1.0990], [1.0998, 1.0950, 1.0995]])
    assert find_sweep(h, l, c, atr, 0, 2, 1.1000, 1.0900, True) == ("no_breach", None)
    h, l, c = sweep_arrays([[1.1010, 1.0980, 1.1005], [1.1020, 1.1000, 1.1010]])
    assert find_sweep(h, l, c, atr, 0, 2, 1.1000, 1.0900, True) == ("no_reclaim", None)
    h, l, c = sweep_arrays([[1.1010, 1.0890, 1.1005], [1.1012, 1.0990, 1.0995]])      # the low of the parent broke first
    assert find_sweep(h, l, c, atr, 0, 2, 1.1000, 1.0900, True) == ("both_sides", 1)
    # mirror: downside sweep of a parent low at 1.0900
    h, l, c = sweep_arrays([[1.0950, 1.0890, 1.0895], [1.0940, 1.0880, 1.0905]])
    assert find_sweep(h, l, c, atr, 0, 2, 1.0900, 1.1000, False) == ("ok", 1)


def test_structure_confirmation_and_the_buffer():
    atr = np.full(40, 0.001)
    c = np.full(40, 1.0990)
    c[7] = 1.09855                                                   # not beyond 1.0985 - 0.0001
    c[9] = 1.0983                                                    # beyond
    assert structure_confirm(c, atr, 5, 1.0985, True, 40) == 9
    assert structure_confirm(c, atr, 10, 1.0985, True, 40) is None
    c2 = np.full(40, 1.0990); c2[8] = 1.1002
    assert structure_confirm(c2, atr, 5, 1.0990, False, 40) == 8                       # mirror (a long setup)


def fvg_inputs(gap):
    tns = (np.datetime64("2022-01-03T00:00") + np.arange(10) * np.timedelta64(15, "m")).astype("datetime64[ns]")
    high = np.array([1.1000, 1.1000, 1.1000, 1.1000, 1.0990, 1.0980, 1.0970 - gap + gap, 1.0960, 1.0950, 1.0950])
    low = np.array([1.0990, 1.0990, 1.0990, 1.0985, 1.0970, 1.0960, 1.0950, 1.0940, 1.0930, 1.0930])
    return high, low, np.full(10, 0.001), tns


def test_fvg_size_consecutive_candles_and_window():
    tns = (np.datetime64("2022-01-03T00:00") + np.arange(10) * np.timedelta64(15, "m")).astype("datetime64[ns]")
    high = np.array([1.101, 1.101, 1.101, 1.101, 1.101, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000])
    low = np.array([1.100, 1.100, 1.100, 1.100, 1.100, 1.0990, 1.0990, 1.0990, 1.0990, 1.0990])
    atr = np.full(10, 0.001)
    high[5] = 1.0990                                   # c3 high
    low[3] = 1.0994                                    # c1 low: gap 0.0004 >= 0.3 x 0.001
    got = find_fvg(high, low, atr, tns, 3, 5, True)
    assert got == (3, 5, 1.0994, 1.0990)
    low[3] = 1.09935                                   # gap 0.00035 -> still enough; shrink below 0.0003
    low[3] = 1.0992
    assert find_fvg(high, low, atr, tns, 3, 5, True) is None                          # gap 0.0002 < 0.0003
    low[3] = 1.0994
    tns2 = tns.copy(); tns2[4:] += np.timedelta64(15, "m")                              # a missing candle inside
    assert find_fvg(high, low, atr, tns2, 3, 5, True) is None
    assert find_fvg(high, low, atr, tns, 4, 5, True) is None                           # c1 would be before the sweep candle


def test_entry_confirmation_invalidation_and_mirror():
    n = 30
    high = np.full(n, 1.0980); low = np.full(n, 1.0970); close = np.full(n, 1.0975)
    top, bottom = 1.0994, 1.0990                       # midpoint 1.0992
    high[12] = 1.0993; close[12] = 1.0991
    assert entry_confirm(high, low, close, 10, top, bottom, True, n) == ("ok", 12)
    close[11] = 1.0995                                 # a close above the top before the retest kills the zone
    assert entry_confirm(high, low, close, 10, top, bottom, True, n) == ("fvg_invalidated", None)
    high2 = np.full(n, 1.0980); low2 = np.full(n, 1.0970); close2 = np.full(n, 1.0975)
    assert entry_confirm(high2, low2, close2, 10, top, bottom, True, n) == ("no_entry", None)
    low3 = np.full(n, 1.1010); high3 = np.full(n, 1.1020); close3 = np.full(n, 1.1015)   # long setup, zone below price
    low3[12] = 1.0991; close3[12] = 1.0993
    assert entry_confirm(high3, low3, close3, 10, top, bottom, False, n) == ("ok", 12)


# ---------------------------------------------------------------- the whole sequence on the planted world
def test_exactly_one_event_with_the_planted_indices():
    df, keys = planted_short()
    att = run_symbol(ctx_for(df))
    ev = att[att["status"] == "event"]
    assert len(ev) == 1
    r = ev.iloc[0]
    assert r["direction"] == -1 and r["pool"] == "PD" and r["zone"] == "ROUND"
    assert (r["s"], r["k"], r["e"], r["entry"]) == (keys["s"], keys["k"], keys["e"], keys["entry"])


def test_mirrored_market_gives_the_mirrored_long_event():
    df, keys = planted_short()
    ev = run_symbol(ctx_for(flip(df)))
    ev = ev[ev["status"] == "event"]
    assert len(ev) == 1 and ev.iloc[0]["direction"] == 1 and ev.iloc[0]["entry"] == keys["entry"]


def test_prices_after_the_entry_cannot_change_the_event():
    df, keys = planted_short()
    rng = np.random.default_rng(1)
    d2 = df.copy()
    cut = keys["entry"] + 1
    for col in ("open", "high", "low", "close"):
        d2.loc[cut:, col] = d2.loc[cut:, col] * (1 + rng.normal(0, 0.003, len(d2) - cut))
    d2["high"] = d2[["open", "high", "low", "close"]].max(axis=1)
    d2["low"] = d2[["open", "high", "low", "close"]].min(axis=1)
    ev = run_symbol(ctx_for(d2))
    ev = ev[ev["status"] == "event"]
    assert len(ev) == 1 and ev.iloc[0]["entry"] == keys["entry"]


def test_an_incomplete_forward_window_is_reported_as_invalid_data_not_hidden():
    df, keys = planted_short()
    short = df.iloc[: keys["entry"] + 10].reset_index(drop=True)
    att = run_symbol(ctx_for(short))
    assert (att["status"] == "invalid_data").sum() == 1 and (att["status"] == "event").sum() == 0


def test_a_data_gap_inside_the_forward_window_invalidates_the_event():
    df, keys = planted_short()
    holed = df.drop(range(keys["entry"] + 5, keys["entry"] + 12)).reset_index(drop=True)   # a 105 minute hole
    att = run_symbol(ctx_for(holed))
    assert (att["status"] == "event").sum() == 0 and (att["status"] == "invalid_data").sum() == 1


def test_news_filter_removes_the_event_only_when_a_relevant_release_is_near():
    df, keys = planted_short()
    entry_time = df["time"].iloc[keys["entry"]].tz_convert(None)
    near = NewsCalendar(pd.DataFrame({"event": ["NFP"], "currency": ["USD"],
                                      "scheduled_utc": [entry_time + pd.Timedelta(minutes=20)]}))
    far = NewsCalendar(pd.DataFrame({"event": ["NFP"], "currency": ["USD"],
                                     "scheduled_utc": [entry_time + pd.Timedelta(minutes=45)]}))
    other = NewsCalendar(pd.DataFrame({"event": ["BoJ"], "currency": ["JPY"],
                                       "scheduled_utc": [entry_time + pd.Timedelta(minutes=5)]}))
    assert (run_symbol(ctx_for(df), near)["status"] == "news").sum() == 1
    assert (run_symbol(ctx_for(df), far)["status"] == "event").sum() == 1
    assert (run_symbol(ctx_for(df), other)["status"] == "event").sum() == 1


class FakeCtx(SymbolContext):
    """Feeds crafted candidates to run_symbol so the cooldown rule can be tested on its own."""

    def __init__(self, entries):
        self.symbol = "EURUSD"
        self._entries = entries

    def attempts(self):
        rows = []
        for d, entry in self._entries:
            rows.append({"symbol": "EURUSD", "parent_time": np.datetime64("2022-01-05T00:00"),
                         "parent_end": np.datetime64("2022-01-05T04:00"), "direction": d, "stage": "entry",
                         "status": "candidate", "entry": entry, "w0": entry - 40, "pool": "PD", "zone": "ROUND",
                         "event_time": np.datetime64("2022-01-05T08:00"), "entry_time": np.datetime64("2022-01-05T08:15")})
        return pd.DataFrame(rows)

    def base_valid(self, entry, w0, horizon=24):
        return True


def test_cooldown_is_deterministic_per_direction_and_records_rejections():
    ctx = FakeCtx([(-1, 1000), (-1, 1050), (-1, 1096), (-1, 1200), (1, 1010)])
    att = run_symbol(ctx)
    status = dict(zip(zip(att["direction"], att["entry"]), att["status"]))
    assert status[(-1, 1000)] == "event"
    assert status[(-1, 1050)] == "cooldown_duplicate"                 # 50 < 96 after the kept event
    assert status[(-1, 1096)] == "event"                              # exactly 96 after the kept event: allowed
    assert status[(-1, 1200)] == "event"
    assert status[(1, 1010)] == "event"                               # the other direction is independent
    shuffled = run_symbol(FakeCtx([(1, 1010), (-1, 1200), (-1, 1096), (-1, 1050), (-1, 1000)]))
    assert sorted(zip(shuffled["direction"], shuffled["entry"], shuffled["status"])) == \
        sorted(zip(att["direction"], att["entry"], att["status"]))


# ---------------------------------------------------------------- reporting and guarantees
def test_sample_verdict_needs_every_horizon_in_both_periods():
    def make(n_dev, n_val, horizons_ok=(True, True, True)):
        rows = [{"period": "development", **{f"h{h}": ok for h, ok in zip((24, 48, 96), horizons_ok)}}
                for _ in range(n_dev)]
        rows += [{"period": "validation", **{f"h{h}": ok for h, ok in zip((24, 48, 96), horizons_ok)}}
                 for _ in range(n_val)]
        return pd.DataFrame(rows)
    assert sample_verdict(make(300, 150))[1] == "SUFFICIENT"
    assert sample_verdict(make(299, 150))[1] == "INCONCLUSIVE"
    assert sample_verdict(make(300, 149))[1] == "INCONCLUSIVE"
    assert sample_verdict(make(300, 150, horizons_ok=(True, True, False)))[1] == "INCONCLUSIVE"
    assert sample_verdict(pd.DataFrame())[1] == "INCONCLUSIVE"


def test_analyse_end_to_end_and_attrition_is_monotone():
    df, keys = planted_short()
    out = analyse({"EURUSD": df}, {"EURUSD": to_h1(df)})
    att = out["attrition"]
    dev = att["development"].to_numpy()
    assert dev[1] <= dev[0] and dev[3] <= dev[2]
    assert all(dev[i] >= dev[i + 1] for i in range(2, len(dev) - 1))          # never grows after the attempts stage
    assert att.loc["cooldown", "development"] == 1 and att.loc["sweep", "development"] == 1
    assert len(out["events"]) == 1 and out["verdict"] == "INCONCLUSIVE"


def test_the_counting_module_never_touches_returns():
    src = inspect.getsource(h6)
    for forbidden in ("net_moves", "SymbolData", "block_bootstrap", "placebo", "moves["):
        assert forbidden not in src


def test_only_swings_confirmed_by_the_sweep_candle_can_be_the_reference():
    conf = np.array([10, 20, 25])
    price = np.array([1.0950, 1.0960, 1.0940])
    assert reference_swing(conf, price, 22) == 1.0960            # the swing confirmed at 25 is still in the future
    assert reference_swing(conf, price, 25) == 1.0940            # confirmed exactly at the close of candle 25: usable
    assert reference_swing(conf, price, 9) is None


def test_eligible_swings_obey_confirmation_pivot_and_lookback_rules():
    conf = np.array([100, 110, 118, 125])
    piv = conf - 3
    price = np.array([1.10, 1.11, 1.12, 1.13])
    got = eligible_swings(conf, piv, price, i_ref=120, lookback=30, before=105)
    assert list(got) == [1.10]                                   # 125 is unconfirmed; 107 and 115 pivoted after `before`
    assert list(eligible_swings(conf, piv, price, 120, 30, 200)) == [1.10, 1.11, 1.12]
    assert list(eligible_swings(conf, piv, price, 120, 20, 200)) == [1.11, 1.12]        # the pivot at 97 is outside 20 candles
    assert list(eligible_swings(conf, piv, price, 120, 10, 200)) == [1.12]


def test_holdout_data_is_never_used_even_if_present():
    df, keys = planted_short(start="2024-12-30 00:00")                        # the planted Thursday is 2025-01-02
    ctx = ctx_for(df)
    cut = np.datetime64("2025-01-01T00:00")
    assert ctx.mt.max() < cut and ctx.h1t.max() < cut and ctx.h4t.max() < cut
    out = analyse({"EURUSD": df}, {"EURUSD": to_h1(df)})
    assert len(out["events"]) == 0
