import numpy as np
import pandas as pd

from backend.backtesting.backtest_engine import count_swap_nights
from backend.market_data.indicators import add_indicators

HORIZONS = (6, 12, 24)                      # H1 bars held, entry at the next candle's open
SLIPPAGE_ATR = 0.02                         # per fill; two fills per trade
RATIO_BINS = np.array([0.0, 0.8, 1.0, 1.25, 1.6, np.inf])   # fixed volatility buckets (no lookahead)
EXCLUDE_RADIUS = 24                         # controls may not sit near an event of the same kind
ROLLOVER_HOUR = 22                          # candle-clock hour at which swap is charged (to be verified)
UTC = "UTC"

# Frozen allocation. Data after RESEARCH_END (the holdout) is never loaded by this module.
RESEARCH_END = pd.Timestamp("2025-01-01", tz=UTC)
PERIODS = {
    "development": (pd.Timestamp("2022-01-01", tz=UTC), pd.Timestamp("2024-01-01", tz=UTC)),
    "validation": (pd.Timestamp("2024-01-01", tz=UTC), RESEARCH_END),
}


def gap_flags(times):
    """True for a bar that follows an unexpected gap (more than 3h and not the normal weekend break)."""
    t = pd.Series(pd.to_datetime(times))
    hours = t.diff().dt.total_seconds() / 3600.0
    weekend = (hours >= 36) & (hours <= 72) & (t.shift(1).dt.weekday == 4)
    return ((hours > 3) & ~weekend).fillna(False).to_numpy()


def swap_nights_vector(start, end, triple_weekday, rollover_hour=ROLLOVER_HOUR):
    """
    Vectorised version of count_swap_nights: weighted rollovers in (start, end] for arrays of
    naive datetime64 values. Mon-Fri rollovers count once, the triple weekday counts three times.
    """
    shift = np.timedelta64(rollover_hour, "h")
    k_start = (start - shift).astype("datetime64[D]").astype(np.int64)
    k_end = (end - shift).astype("datetime64[D]").astype(np.int64)
    d0 = int(min(k_start.min(), k_end.min()))
    days = np.arange(d0, int(max(k_start.max(), k_end.max())) + 1)
    weekday = (days + 3) % 7                                   # 1970-01-01 was a Thursday (Monday = 0)
    weight = np.where(weekday < 5, np.where(weekday == triple_weekday, 3, 1), 0)
    cum = np.cumsum(weight)
    return cum[k_end - d0] - cum[k_start - d0]


class SymbolData:
    """Everything the event study needs for one symbol, computed once (including every bar's forward moves)."""

    def __init__(self, df, point, swap=None, max_h=max(HORIZONS), cost_mult=1.0):
        df = df[df["time"] < RESEARCH_END].reset_index(drop=True)          # the holdout never gets in
        self.df = add_indicators(df) if "atr_14" not in df.columns else df
        d = self.df
        self.n = len(d)
        self.t = pd.to_datetime(d["time"])
        self.o = d["open"].to_numpy(); self.c = d["close"].to_numpy()
        self.spread = d["spread"].to_numpy(dtype=float) * point * cost_mult
        self.atr = d["atr_14"].to_numpy()
        self.point, self.swap, self.cost_mult = point, swap, cost_mult
        ratio = d["atr_14"] / d["atr_14"].rolling(250, min_periods=250).median()
        self.ratio = ratio.to_numpy()
        self.bin = np.digitize(np.nan_to_num(self.ratio, nan=-1), RATIO_BINS) - 1      # -1 = warm-up
        self.hour = self.t.dt.hour.to_numpy()

        bad = gap_flags(d["time"])
        bad_cum = np.concatenate([[0], np.cumsum(bad)])
        tt = self.t.to_numpy()
        after_weekend = np.zeros(self.n, dtype=bool)
        gap_after_entry = np.zeros(self.n, dtype=bool)
        for i in range(self.n - 1):
            if (tt[i + 1] - tt[i]) / np.timedelta64(1, "h") > 3:
                after_weekend[i] = True
        for i in range(self.n - max_h - 1):
            gap_after_entry[i] = bad_cum[i + max_h + 2] - bad_cum[i + 1] > 0
        enough = np.arange(self.n) < self.n - max_h - 1
        self.valid = enough & ~gap_after_entry & ~after_weekend & (self.bin >= 0) & ~np.isnan(self.atr)
        self._naive = self.t.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
        self.moves = self._precompute(max_h)

    def _precompute(self, max_h):
        """Net move in ATR units for every bar, both directions, every horizon. Shape (n, 2, len(HORIZONS))."""
        n = self.n
        out = np.full((n, 2, len(HORIZONS)), np.nan)
        i = np.arange(n)
        e = np.minimum(i + 1, n - 1)
        for k, h in enumerate(HORIZONS):
            x = np.minimum(e + h - 1, n - 1)                       # exit candle
            slip = 2 * SLIPPAGE_ATR * self.cost_mult * self.atr
            gross = self.c[x] - self.o[e]
            long_cost = self.spread[e] + slip                      # buy at the ask on the ENTRY candle
            short_cost = self.spread[x] + slip                     # sell short, buy back at the ask on the EXIT candle
            long_swap = short_swap = 0.0
            if self.swap is not None:
                nights = swap_nights_vector(self._naive[e], self._naive[x] + np.timedelta64(1, "h"),
                                            self.swap["triple_weekday"])
                long_swap = self.swap["long_pts"] * self.point * nights
                short_swap = self.swap["short_pts"] * self.point * nights
            out[:, 0, k] = (gross - long_cost + long_swap) / self.atr
            out[:, 1, k] = (-gross - short_cost + short_swap) / self.atr
        out[~self.valid] = np.nan
        return out

    def net_moves(self, idx, direction):
        """Net directional move after costs, in ATR units, looked up for candles idx (direction +1 / -1)."""
        idx = np.asarray(idx, dtype=int)
        side = (np.asarray(direction) == -1).astype(int)
        return self.moves[idx, side, :]

    def net_moves_reference(self, idx, direction, horizons=HORIZONS):
        """Slow, obviously-correct version used only to test the vectorised one."""
        out = np.full((len(idx), len(horizons)), np.nan)
        for r, (i, d) in enumerate(zip(idx, direction)):
            if not self.valid[i]:
                continue
            e = i + 1
            for k, h in enumerate(horizons):
                x = e + h - 1
                gross = d * (self.c[x] - self.o[e])
                spread = self.spread[e] if d == 1 else self.spread[x]
                cost = spread + 2 * SLIPPAGE_ATR * self.cost_mult * self.atr[i]
                swap_price = 0.0
                if self.swap is not None:
                    end = self.t.iloc[x] + pd.Timedelta(hours=1)
                    nights = count_swap_nights(self.t.iloc[e], end, ROLLOVER_HOUR, self.swap["triple_weekday"])
                    pts = self.swap["long_pts"] if d == 1 else self.swap["short_pts"]
                    swap_price = pts * self.point * nights
                out[r, k] = (gross - cost + swap_price) / self.atr[i]
        return out


def control_pools(sd, event_idx, period_bounds):
    """Valid control bars per (period, hour, volatility bucket), excluding bars near events of this kind."""
    near = np.zeros(sd.n, dtype=bool)
    for i in event_idx:
        near[max(0, i - EXCLUDE_RADIUS): i + EXCLUDE_RADIUS + 1] = True
    ok = sd.valid & ~near
    pools = {}
    for pname, (a, b) in period_bounds.items():
        in_period = ((sd.t >= a) & (sd.t < b)).to_numpy()
        base = np.flatnonzero(ok & in_period)
        keys = sd.hour[base] * 10 + sd.bin[base]
        for key in np.unique(keys):
            pools[(pname, int(key))] = base[keys == key]
    return pools


def nearest_controls(pool, i, k, taken, base_window=360, widenings=3, minimum=3):
    """
    Up to k UNUSED control bars from `pool` closest in time to bar i (within +-base_window bars, widened
    up to `widenings` times). Chosen bars are marked in `taken`, so no control is ever reused.
    """
    window = base_window
    for _ in range(widenings + 1):
        lo, hi = np.searchsorted(pool, i - window), np.searchsorted(pool, i + window, side="right")
        cand = pool[lo:hi]
        cand = cand[~taken[cand]]
        if len(cand) >= minimum:
            chosen = cand[np.argsort(np.abs(cand - i), kind="stable")[:k]]
            taken[chosen] = True
            return chosen
        window *= 2
    return np.array([], dtype=int)


def block_bootstrap_ci(values, block_ids, n_boot=2000, seed=0, level=95):
    """Mean and confidence interval, resampling whole blocks (for example months) with replacement."""
    values = np.asarray(values, dtype=float)
    ids, inv = np.unique(block_ids, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return values.mean(), lo, hi


def matched_excess(events, symbol_data, kind, periods, k_controls=5):
    """
    Match every event of one kind to controls and return, per period, the net moves and the excess
    over controls: {period: {net, excess, block, half, unmatched, invalid}}.
    Controls: same symbol, period, hour of day, volatility bucket and direction; not within 24 bars of an
    event of this kind; closest in time; each control bar used at most once.
    """
    ev_kind = events[events["kind"] == kind]
    per_period = {p: {"net": [], "excess": [], "block": [], "half": [], "unmatched": 0, "invalid": 0}
                  for p in periods}
    for sym, grp in ev_kind.groupby("symbol"):
        sd = symbol_data[sym]
        pools = control_pools(sd, grp["idx"].to_numpy(), periods)
        taken = np.zeros(sd.n, dtype=bool)
        ev_time = pd.to_datetime(grp["time"], utc=True)
        for pname, (a, b) in periods.items():
            sub = grp[(ev_time >= a) & (ev_time < b)].sort_values("idx")
            if sub.empty:
                continue
            idx = sub["idx"].to_numpy(); dirs = sub["direction"].to_numpy()
            ev_net = sd.net_moves(idx, dirs)
            times = pd.to_datetime(sub["time"], utc=True)
            store = per_period[pname]
            for r in range(len(idx)):
                if np.isnan(ev_net[r]).any():
                    store["invalid"] += 1                  # for example a data gap inside the holding window
                    continue
                pool = pools.get((pname, int(sd.hour[idx[r]] * 10 + sd.bin[idx[r]])))
                chosen = nearest_controls(pool, idx[r], k_controls, taken) if pool is not None else []
                if len(chosen) == 0:
                    store["unmatched"] += 1
                    continue
                ctrl = sd.net_moves(chosen, np.full(len(chosen), dirs[r]))
                ok = ~np.isnan(ctrl).any(axis=1)
                if not ok.any():
                    store["unmatched"] += 1
                    continue
                t = times.iloc[r]
                store["net"].append(ev_net[r]); store["excess"].append(ev_net[r] - ctrl[ok].mean(axis=0))
                store["block"].append(f"{t.year}-{t.month:02d}")
                store["half"].append(f"{t.year}H{1 if t.month <= 6 else 2}")
    return per_period


def study(events, symbol_data, kinds=None, k_controls=5, seed=1, periods=None, n_boot=2000):
    """
    Event study against matched controls. events: DataFrame with symbol, idx, time, kind, direction.
    Confidence intervals resample whole calendar months. Returns one row per kind, period, horizon.
    """
    periods = periods or PERIODS
    rows = []
    for kind in (kinds or sorted(events["kind"].unique())):
        per_period = matched_excess(events, symbol_data, kind, periods, k_controls)
        for pname in periods:
            s = per_period[pname]
            if not s["net"]:
                continue
            net = np.array(s["net"]); exc = np.array(s["excess"])
            halves = pd.Series(s["half"])
            for k, h in enumerate(HORIZONS):
                m_net, net_lo, net_hi = block_bootstrap_ci(net[:, k], s["block"], n_boot, seed)
                m_exc, exc_lo, exc_hi = block_bootstrap_ci(exc[:, k], s["block"], n_boot, seed)
                by_half = pd.Series(exc[:, k]).groupby(halves.values).mean()
                rows.append({
                    "kind": kind, "period": pname, "horizon": h, "events": len(net),
                    "unmatched": s["unmatched"], "invalid": s["invalid"],
                    "net_mean": m_net, "net_lo": net_lo, "net_hi": net_hi,
                    "excess_mean": m_exc, "excess_lo": exc_lo, "excess_hi": exc_hi,
                    "halves_positive": int((by_half > 0).sum()), "halves": int(len(by_half)),
                })
    return pd.DataFrame(rows)


def placebo_events(events, symbol_data, periods, rng):
    """
    Fake events with the same counts, symbols, periods, directions and (hour, volatility) mix as the real
    ones, placed at random valid bars. Used to measure how big an 'excess' pure chance produces.
    """
    parts = []
    for (sym, kind), grp in events.groupby(["symbol", "kind"]):
        sd = symbol_data[sym]
        ev_time = pd.to_datetime(grp["time"], utc=True)
        for pname, (a, b) in periods.items():
            sub = grp[(ev_time >= a) & (ev_time < b)]
            if sub.empty:
                continue
            in_period = ((sd.t >= a) & (sd.t < b)).to_numpy()
            ok = sd.valid & in_period
            keys_all = sd.hour * 10 + sd.bin
            picked = []
            for idx in sub["idx"].to_numpy():
                cand = np.flatnonzero(ok & (keys_all == sd.hour[idx] * 10 + sd.bin[idx]))
                picked.append(rng.choice(cand) if len(cand) else idx)
            picked = np.array(picked)
            parts.append(pd.DataFrame({
                "symbol": sym, "idx": picked, "time": sd.t.iloc[picked].values, "kind": kind,
                "direction": rng.permutation(sub["direction"].to_numpy()),
            }))
    return pd.concat(parts, ignore_index=True) if parts else events.iloc[0:0]


def placebo_calibration(events, symbol_data, periods, runs=200, k_controls=5, seed=11):
    """
    Distribution of the mean excess produced by placebo events. Returns {(kind, period): array (runs, horizons)}.
    A real excess is only believable if it sits far out in this distribution.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for _ in range(runs):
        fake = placebo_events(events, symbol_data, periods, rng)
        for kind in sorted(fake["kind"].unique()):
            per_period = matched_excess(fake, symbol_data, kind, periods, k_controls)
            for pname in periods:
                s = per_period[pname]
                if s["excess"]:
                    out.setdefault((kind, pname), []).append(np.mean(np.array(s["excess"]), axis=0))
    return {key: np.array(v) for key, v in out.items()}


def add_placebo_pvalues(results, calibration):
    """One-sided p-value for each row: share of placebo runs whose mean excess is at least as large."""
    res = results.copy()
    pv, sd = [], []
    for _, r in res.iterrows():
        dist = calibration.get((r["kind"], r["period"]))
        k = HORIZONS.index(int(r["horizon"]))
        if dist is None or len(dist) == 0:
            pv.append(np.nan); sd.append(np.nan)
        else:
            col = dist[:, k]
            pv.append((1 + int((col >= r["excess_mean"]).sum())) / (len(col) + 1))
            sd.append(float(col.std(ddof=1)))
    res["placebo_p"] = pv
    res["placebo_sd"] = sd
    return res
