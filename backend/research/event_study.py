import numpy as np
import pandas as pd

from backend.backtesting.backtest_engine import count_swap_nights
from backend.market_data.indicators import add_indicators

HORIZONS = (6, 12, 24)                      # H1 bars held, entry at the next candle's open
SLIPPAGE_ATR = 0.02                         # per fill; two fills per trade
RATIO_BINS = np.array([0.0, 0.8, 1.0, 1.25, 1.6, np.inf])   # fixed volatility buckets (no lookahead)
EXCLUDE_RADIUS = 24                         # controls may not sit near an event of the same kind
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


class SymbolData:
    """Everything the event study needs for one symbol, computed once."""

    def __init__(self, df, point, swap=None, max_h=max(HORIZONS)):
        df = df[df["time"] < RESEARCH_END].reset_index(drop=True)          # the holdout never gets in
        self.df = add_indicators(df) if "atr_14" not in df.columns else df
        d = self.df
        self.n = len(d)
        self.t = pd.to_datetime(d["time"])
        self.o = d["open"].to_numpy(); self.c = d["close"].to_numpy()
        self.spread = d["spread"].to_numpy(dtype=float) * point
        self.atr = d["atr_14"].to_numpy()
        self.point, self.swap = point, swap
        ratio = d["atr_14"] / d["atr_14"].rolling(250, min_periods=250).median()
        self.ratio = ratio.to_numpy()
        self.bin = np.digitize(np.nan_to_num(self.ratio, nan=-1), RATIO_BINS) - 1      # -1 = warm-up
        self.hour = self.t.dt.hour.to_numpy()

        bad = gap_flags(d["time"])
        bad_cum = np.concatenate([[0], np.cumsum(bad)])
        gap_after_entry = np.zeros(self.n, dtype=bool)
        after_weekend = np.zeros(self.n, dtype=bool)
        tt = self.t.to_numpy()
        for i in range(self.n - 1):
            if (tt[i + 1] - tt[i]) / np.timedelta64(1, "h") > 3:
                after_weekend[i] = True
        for i in range(self.n - max_h - 1):
            gap_after_entry[i] = bad_cum[i + max_h + 2] - bad_cum[i + 1] > 0
        enough = np.arange(self.n) < self.n - max_h - 1
        self.valid = enough & ~gap_after_entry & ~after_weekend & (self.bin >= 0) & ~np.isnan(self.atr)

    def net_moves(self, idx, direction, horizons=HORIZONS):
        """Net directional move after costs, in ATR units, for a trade signalled on candle idx."""
        out = np.full((len(idx), len(horizons)), np.nan)
        for r, (i, d) in enumerate(zip(idx, direction)):
            if not self.valid[i]:
                continue
            e = i + 1
            for k, h in enumerate(horizons):
                exit_i = e + h - 1
                gross = d * (self.c[exit_i] - self.o[e])
                cost = self.spread[e] + 2 * SLIPPAGE_ATR * self.atr[i]
                swap_price = 0.0
                if self.swap is not None:
                    end = self.t.iloc[exit_i] + pd.Timedelta(hours=1)
                    nights = count_swap_nights(self.t.iloc[e], end, 22, self.swap["triple_weekday"])
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


def nearest_controls(pool, i, k, base_window=360, widenings=3, minimum=3):
    """
    Up to k control bars from `pool` that are CLOSEST IN TIME to bar i (within +-base_window bars,
    widened up to `widenings` times). Using nearby controls keeps each event/control difference
    local, so monthly block resampling captures their shared noise honestly.
    """
    window = base_window
    for _ in range(widenings + 1):
        lo, hi = np.searchsorted(pool, i - window), np.searchsorted(pool, i + window, side="right")
        cand = pool[lo:hi]
        if len(cand) >= minimum:
            order = np.argsort(np.abs(cand - i), kind="stable")[:k]
            return cand[order]
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


def study(events, symbol_data, kinds=None, k_controls=10, seed=1, periods=None, n_boot=2000):
    """
    Event study against matched controls.
    events: DataFrame with symbol, idx, time, kind, direction (one row per event).
    Controls: same symbol, same period, same hour of day, same volatility bucket, same direction,
              not within 24 bars of an event of the same kind, and CLOSEST IN TIME to the event.
    Confidence intervals resample whole calendar months, because events and their controls share
    market conditions within a month. Returns one row per kind, period, horizon.
    """
    periods = periods or PERIODS
    rows = []
    for kind in (kinds or sorted(events["kind"].unique())):
        ev_kind = events[events["kind"] == kind]
        per_period = {p: {"net": [], "excess": [], "block": [], "half": []} for p in periods}
        unmatched = {p: 0 for p in periods}
        invalid = {p: 0 for p in periods}
        for sym, grp in ev_kind.groupby("symbol"):
            sd = symbol_data[sym]
            pools = control_pools(sd, grp["idx"].to_numpy(), periods)
            ev_time = pd.to_datetime(grp["time"], utc=True)
            for pname, (a, b) in periods.items():
                sub = grp[(ev_time >= a) & (ev_time < b)]
                if sub.empty:
                    continue
                idx = sub["idx"].to_numpy(); dirs = sub["direction"].to_numpy()
                ev_net = sd.net_moves(idx, dirs)
                for r in range(len(idx)):
                    if np.isnan(ev_net[r]).any():
                        invalid[pname] += 1                      # for example a data gap inside the holding window
                        continue
                    pool = pools.get((pname, int(sd.hour[idx[r]] * 10 + sd.bin[idx[r]])))
                    chosen = nearest_controls(pool, idx[r], k_controls) if pool is not None else []
                    if len(chosen) == 0:
                        unmatched[pname] += 1
                        continue
                    ctrl = sd.net_moves(chosen, np.full(len(chosen), dirs[r]))
                    ctrl = ctrl[~np.isnan(ctrl).any(axis=1)]
                    if len(ctrl) == 0:
                        unmatched[pname] += 1
                        continue
                    t = pd.to_datetime(sub["time"].iloc[r], utc=True)
                    store = per_period[pname]
                    store["net"].append(ev_net[r]); store["excess"].append(ev_net[r] - ctrl.mean(axis=0))
                    store["block"].append(f"{t.year}-{t.month:02d}")
                    store["half"].append(f"{t.year}H{1 if t.month <= 6 else 2}")
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
                    "unmatched": unmatched[pname], "invalid": invalid[pname],
                    "net_mean": m_net, "net_lo": net_lo, "net_hi": net_hi,
                    "excess_mean": m_exc, "excess_lo": exc_lo, "excess_hi": exc_hi,
                    "halves_positive": int((by_half > 0).sum()), "halves": int(len(by_half)),
                    "passes": bool(net_lo > 0 and exc_lo > 0 and len(net) >= 300),
                })
    return pd.DataFrame(rows)
