"""
H6-v1 counts-only stage. It detects the full setup sequence and counts events. It NEVER computes returns.
Every rule below follows research/H6-v1_spec.md.
"""
import numpy as np
import pandas as pd

from backend.market_data.indicators import atr as atr_fn, swings as swings_fn
from backend.research.event_study import PERIODS, RESEARCH_END, completed_atr_reference, gap_flags

# ---- frozen parameters (spec sections 2 to 5)
POOL_TOL = 0.3        # other pool must lie within this x H1 ATR of the parent extreme
EQ_TOL = 0.2          # equal highs/lows tolerance, x H1 ATR
CLUSTER_TOL = 0.5     # H4 swing cluster tolerance, x H4 ATR
ZONE_HALF = 0.25      # S/R zone half-width, x H4 ATR
SWEEP_BUF = 0.10      # reclaim and structure-break buffer, x M15 ATR
FVG_MIN = 0.30        # minimum gap, x M15 ATR
PARENT_MIN_RANGE = 1.0   # parent range, x H4 ATR
WINDOW = 16           # M15 candles in the parent validity window (the following H4 candle)
CONFIRM_WINDOW = 16
ENTRY_WINDOW = 16
COOLDOWN = 96
H1_LOOKBACK = 120
H4_LOOKBACK = 120
SWING_N = 3
HORIZONS = (24, 48, 96)
GATE_LIMIT_PCT = 5.0
MIN_DEV, MIN_VAL = 300, 150
MIN_DAY_CANDLES = 12

STAGES = ["parents", "parent_range", "attempts", "confluence", "sr_zone", "sweep", "structure", "fvg",
          "entry", "news", "base_valid", "cooldown"]
STAGE_LABEL = {
    "parents": "H4 parent candles (complete)", "parent_range": "range >= 1.0 x H4 ATR",
    "attempts": "parent-side attempts (2 per parent)", "confluence": "liquidity pool confluence",
    "sr_zone": "support/resistance zone", "sweep": "M15 sweep + reclaim (not both sides)",
    "structure": "M15 structure confirmation", "fvg": "qualifying FVG", "entry": "retracement entry confirmed",
    "news": "passes news filter", "base_valid": "valid data (shortest horizon)", "cooldown": "after 96-candle cooldown",
}
ROUND_STEP = {"USDJPY": 1.0, "XAUUSD": 10.0}
DEFAULT_ROUND_STEP = 0.01
PAIR_CURRENCIES = {"EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"), "USDJPY": ("USD", "JPY"),
                   "USDCHF": ("USD", "CHF"), "AUDUSD": ("AUD", "USD"), "USDCAD": ("USD", "CAD"),
                   "NZDUSD": ("NZD", "USD"), "XAUUSD": ("XAU", "USD")}


def _naive(times):
    return pd.to_datetime(pd.Series(times), utc=True).dt.tz_convert("UTC").dt.tz_localize(None).dt.as_unit("ns").to_numpy()


# ------------------------------------------------------------------ data gate (spec section 1)
def data_gate(frames, limit_pct=GATE_LIMIT_PCT, presence=0.5):
    """
    Completeness against EXPECTED tradable candles, not calendar time.
    * A time slot (weekday + time of day) belongs to a symbol's normal schedule when the symbol has a candle in
      at least `presence` of the weeks in which any symbol traded that slot (so a daily maintenance break is not
      'missing').
    * A timestamp at which NO symbol has a candle is a market-wide closure and is never counted as missing.
    Returns {symbol: {"expected": n, "missing": n, "missing_pct": x, "admitted": bool}}.
    """
    sets = {s: pd.DatetimeIndex(_naive(df["time"])) for s, df in frames.items()}
    union = sets[next(iter(sets))]
    for idx in sets.values():
        union = union.union(idx)
    slot = (union.day_of_week * 96 + union.hour * 4 + union.minute // 15).to_numpy()
    out = {}
    for sym, idx in sets.items():
        present = union.isin(idx)
        frac = pd.Series(present).groupby(slot).mean()
        in_schedule = pd.Series(slot).map(frac).to_numpy() >= presence
        expected = int(in_schedule.sum())
        missing = int((in_schedule & ~present).sum())
        pct = 100.0 * missing / expected if expected else 100.0
        out[sym] = {"expected": expected, "missing": missing, "missing_pct": round(pct, 3), "admitted": pct <= limit_pct}
    return out


# ------------------------------------------------------------------ news calendar (spec section 6)
class NewsCalendar:
    """Scheduled high-impact releases. Columns: event, currency, scheduled_utc (original scheduled time only)."""

    def __init__(self, df, clock_offset_hours=0.0):
        self.df = df.copy()
        self.df["when"] = _naive(self.df["scheduled_utc"]) + np.timedelta64(int(round(clock_offset_hours * 60)), "m")

    @classmethod
    def load(cls, path, clock_offset_hours=0.0):
        return cls(pd.read_csv(path), clock_offset_hours)

    def excluded(self, symbol, entry_times):
        """Boolean array: True where an entry time is within +-30 minutes of a relevant scheduled release."""
        cur = PAIR_CURRENCIES[symbol]
        rel = self.df[self.df["currency"].isin(cur)]["when"].to_numpy()
        entry = np.asarray(entry_times, dtype="datetime64[ns]")
        if len(rel) == 0 or len(entry) == 0:
            return np.zeros(len(entry), dtype=bool)
        rel = np.sort(rel)
        pos = np.searchsorted(rel, entry)
        half = np.timedelta64(30, "m")
        left = np.where(pos > 0, rel[np.maximum(pos - 1, 0)], np.datetime64("NaT", "ns"))
        right = np.where(pos < len(rel), rel[np.minimum(pos, len(rel) - 1)], np.datetime64("NaT", "ns"))
        near_left = ~np.isnat(left) & (np.abs(entry - left) <= half)
        near_right = ~np.isnat(right) & (np.abs(right - entry) <= half)
        return near_left | near_right


# ------------------------------------------------------------------ building blocks
def build_h4(h1):
    """H4 candles from H1, aligned to the broker day start. Only COMPLETE bars (4 H1 candles) are kept."""
    t = h1.set_index(pd.to_datetime(h1["time"], utc=True))
    g = t.resample("4h", origin="start_day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                                 close=("close", "last"), n=("close", "size"))
    g = g[g["n"] == 4].drop(columns="n").reset_index().rename(columns={"time": "time"})
    return g


def swing_table(df):
    """Confirmed swings: arrays of (confirmation index, pivot index, price) for highs and lows."""
    sw = swings_fn(df, SWING_N, SWING_N)
    out = {}
    for name in ("swing_high", "swing_low"):
        conf = np.flatnonzero(~sw[name].isna().to_numpy())
        out[name] = (conf, conf - SWING_N, sw[name].to_numpy()[conf])
    return out


def pool_confluence(level, h1_atr, pdx, eq_level):
    """
    Liquidity confluence for a parent extreme `level`. pdx = previous day's extreme (or None); eq_level = the
    equal-extreme pool level (or None). Returns the pool types that coincide, deduplicated:
    a previous-day level and an equal-extreme level at the same price count as ONE pool.
    """
    tol = POOL_TOL * h1_atr
    found = {}
    if pdx is not None and abs(pdx - level) <= tol:
        found["PD"] = pdx
    if eq_level is not None and abs(eq_level - level) <= tol:
        found["EQ"] = eq_level
    if len(found) == 2 and abs(found["PD"] - found["EQ"]) <= EQ_TOL * h1_atr:
        found = {"PD+EQ": found["PD"]}                 # the same price level, counted once
    return sorted(found)


def equal_level(kind_prices, h1_atr, upside):
    """The two most recent eligible swings; if within EQ_TOL x ATR their pool level is the more extreme one."""
    if len(kind_prices) < 2:
        return None
    a, b = kind_prices[-2], kind_prices[-1]
    if abs(a - b) > EQ_TOL * h1_atr:
        return None
    return max(a, b) if upside else min(a, b)


def sr_zone(symbol, level, h4_atr, swing_prices):
    """Round-number or H4 swing-cluster zone containing `level`. Returns the zone types found."""
    types = []
    step = ROUND_STEP.get(symbol, DEFAULT_ROUND_STEP)
    if abs(level - round(level / step) * step) <= ZONE_HALF * h4_atr:
        types.append("ROUND")
    sp = np.asarray(swing_prices, dtype=float)
    for x in sp:
        cluster = sp[np.abs(sp - x) <= CLUSTER_TOL * h4_atr]
        if len(cluster) >= 2 and abs(level - cluster.mean()) <= ZONE_HALF * h4_atr:
            types.append("SWING")
            break
    return types


def reference_swing(conf, price, s):
    """The most recent swing CONFIRMED by the close of candle s (confirmation index <= s). None if there is none."""
    pos = int(np.searchsorted(conf, s, "right")) - 1
    return None if pos < 0 else float(price[pos])


def eligible_swings(conf, piv, price, i_ref, lookback, before):
    """Swings usable at reference candle i_ref: confirmed by then, pivoted before `before`, within the lookback."""
    ok = (conf <= i_ref) & (piv < before) & (piv >= i_ref - lookback)
    return price[ok]


def find_sweep(high, low, close, atr, w0, w1, level, other_extreme, upside):
    """
    Spec step 4. Returns (status, s). Upside: first candle breaching `level` is a; the sweep completes at the first
    candle s >= a (inside the window) that closes at least 0.1 x M15 ATR back inside. If the OTHER extreme of the
    parent was breached at any candle up to and including s, the attempt has swept both sides and is rejected.
    """
    a = None
    for j in range(w0, w1):
        if (high[j] > level) if upside else (low[j] < level):
            a = j
            break
    if a is None:
        return "no_breach", None
    for j in range(a, w1):
        if np.isnan(atr[j]):
            continue
        back = (close[j] <= level - SWEEP_BUF * atr[j]) if upside else (close[j] >= level + SWEEP_BUF * atr[j])
        if back:
            both = any((low[x] < other_extreme) if upside else (high[x] > other_extreme) for x in range(w0, j + 1))
            return ("both_sides", j) if both else ("ok", j)
    return "no_reclaim", None


def structure_confirm(close, atr, s, ref, upside, n):
    """First candle after s (within 16) closing 0.1 x ATR beyond the reference swing. Returns k or None."""
    for j in range(s + 1, min(s + CONFIRM_WINDOW, n - 1) + 1):
        if np.isnan(atr[j]):
            continue
        if (close[j] <= ref - SWEEP_BUF * atr[j]) if upside else (close[j] >= ref + SWEEP_BUF * atr[j]):
            return j
    return None


def find_fvg(high, low, atr, tns, s, k, upside):
    """
    Latest qualifying FVG with all three candles inside [s, k] and consecutive in time.
    Upside sweep -> bearish FVG: low[c1] > high[c3]. Returns (c1, c3, top, bottom) or None.
    """
    step = np.timedelta64(15, "m")
    for c3 in range(k, s + 1, -1):
        c1 = c3 - 2
        if c1 < s or np.isnan(atr[c3]):
            continue
        if tns[c3] - tns[c3 - 1] != step or tns[c3 - 1] - tns[c1] != step:
            continue
        gap = (low[c1] - high[c3]) if upside else (low[c3] - high[c1])
        if gap >= FVG_MIN * atr[c3]:
            top, bottom = (low[c1], high[c3]) if upside else (low[c3], high[c1])
            return c1, c3, top, bottom
    return None


def entry_confirm(high, low, close, k, top, bottom, upside, n):
    """Spec step 7. Returns ('ok', e) or (status, None)."""
    mid = (top + bottom) / 2.0
    for j in range(k + 1, min(k + ENTRY_WINDOW, n - 1) + 1):
        if upside:
            if close[j] > top:
                return "fvg_invalidated", None
            if high[j] >= mid and close[j] < mid:
                return "ok", j
        else:
            if close[j] < bottom:
                return "fvg_invalidated", None
            if low[j] <= mid and close[j] > mid:
                return "ok", j
    return "no_entry", None


# ------------------------------------------------------------------ per-symbol detection
class SymbolContext:
    def __init__(self, symbol, m15, h1):
        self.symbol = symbol
        m15 = m15[m15["time"] < RESEARCH_END].reset_index(drop=True)
        h1 = h1[h1["time"] < RESEARCH_END].reset_index(drop=True)
        self.m15 = m15
        self.mt = _naive(m15["time"])
        self.mh, self.ml, self.mc = (m15[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
        self.matr = atr_fn(m15, 14).to_numpy(dtype=float)
        self.msw = swing_table(m15)
        self.h1t = _naive(h1["time"])
        self.h1atr = atr_fn(h1, 14).to_numpy(dtype=float)
        self.h1sw = swing_table(h1)
        self.h1high, self.h1low = h1["high"].to_numpy(dtype=float), h1["low"].to_numpy(dtype=float)
        h4 = build_h4(h1)
        self.h4 = h4
        self.h4t = _naive(h4["time"])
        self.h4atr = atr_fn(h4, 14).to_numpy(dtype=float)
        self.h4sw = swing_table(h4)
        day = pd.to_datetime(self.h1t).normalize()
        daily = pd.DataFrame({"day": day, "h": self.h1high, "l": self.h1low}).groupby("day").agg(
            h=("h", "max"), l=("l", "min"), n=("h", "size"))
        daily = daily[daily["n"] >= MIN_DAY_CANDLES]
        self.day_high, self.day_low = daily["h"].to_dict(), daily["l"].to_dict()
        self.outcome_atr = completed_atr_reference(m15["time"], 15, h1)
        self.gap_cum = np.concatenate([[0], np.cumsum(gap_flags(m15["time"], 1.0))])

    def _swing_prices(self, table, name, i_ref, lookback, before):
        conf, piv, price = table[name]
        return eligible_swings(conf, piv, price, i_ref, lookback, before)

    def attempts(self):
        rows = []
        h4_time = self.h4t
        for p in range(len(h4_time)):
            p_start = h4_time[p]
            p_end = p_start + np.timedelta64(4, "h")
            h4_atr = self.h4atr[p]
            hi_p, lo_p = self.h4["high"].iloc[p], self.h4["low"].iloc[p]
            base = {"symbol": self.symbol, "parent_time": p_start, "parent_end": p_end}
            if np.isnan(h4_atr):
                continue
            ok_range = (hi_p - lo_p) >= PARENT_MIN_RANGE * h4_atr
            rows.append({**base, "direction": 0, "stage": "parents" if not ok_range else "parent_range"})
            if not ok_range:
                continue
            i1_start = np.searchsorted(self.h1t, p_start, "left")
            i1_end = np.searchsorted(self.h1t, p_end, "left")
            if i1_end == 0 or np.isnan(self.h1atr[i1_end - 1]):
                continue
            h1_atr = self.h1atr[i1_end - 1]
            w0 = int(np.searchsorted(self.mt, p_end, "left"))
            w1 = int(np.searchsorted(self.mt, p_end + np.timedelta64(4, "h"), "left"))
            today = pd.Timestamp(p_end).normalize()
            prev = today - pd.Timedelta(days=1)
            for upside in (True, False):
                direction = -1 if upside else 1
                level = hi_p if upside else lo_p
                other = lo_p if upside else hi_p
                rec = {**base, "direction": direction, "stage": "attempts", "pool": "", "zone": ""}
                sw_name = "swing_high" if upside else "swing_low"
                prices = self._swing_prices(self.h1sw, sw_name, i1_end - 1, H1_LOOKBACK, i1_start)
                eq = equal_level(list(prices), h1_atr, upside)
                pdx = self.day_high.get(prev) if upside else self.day_low.get(prev)
                pools = pool_confluence(level, h1_atr, pdx, eq)
                if not pools:
                    rows.append({**rec, "status": "no_confluence"}); continue
                rec["pool"] = "+".join(pools)
                rec["stage"] = "confluence"
                zprices = self._swing_prices(self.h4sw, sw_name, p, H4_LOOKBACK, p)
                zones = sr_zone(self.symbol, level, h4_atr, zprices)
                if not zones:
                    rows.append({**rec, "status": "no_sr_zone"}); continue
                rec["zone"] = "+".join(zones)
                rec["stage"] = "sr_zone"
                status, s = find_sweep(self.mh, self.ml, self.mc, self.matr, w0, w1, level, other, upside)
                if status != "ok":
                    rows.append({**rec, "status": status}); continue
                rec["stage"] = "sweep"
                conf, piv, price = self.msw["swing_low" if upside else "swing_high"]
                ref = reference_swing(conf, price, s)
                if ref is None:
                    rows.append({**rec, "status": "no_reference"}); continue
                k = structure_confirm(self.mc, self.matr, s, ref, upside, len(self.mc))
                if k is None:
                    rows.append({**rec, "status": "no_structure"}); continue
                rec["stage"] = "structure"
                fvg = find_fvg(self.mh, self.ml, self.matr, self.mt, s, k, upside)
                if fvg is None:
                    rows.append({**rec, "status": "no_fvg"}); continue
                rec["stage"] = "fvg"
                st, e = entry_confirm(self.mh, self.ml, self.mc, k, fvg[2], fvg[3], upside, len(self.mc))
                if st != "ok":
                    rows.append({**rec, "status": st}); continue
                entry = e + 1
                if entry >= len(self.mc):
                    rows.append({**rec, "status": "no_entry_candle"}); continue
                rec.update(stage="entry", status="candidate", s=s, k=k, e=e, entry=entry,
                           event_time=self.mt[e] + np.timedelta64(15, "m"), entry_time=self.mt[entry], w0=w0)
                rows.append(rec)
        return pd.DataFrame(rows)

    def base_valid(self, entry, w0, horizon=HORIZONS[0]):
        """Spec section 5: gap-free sequence and forward window, outcome ATR available, complete window."""
        end = entry + horizon - 1                      # last candle of the holding window
        if end >= len(self.mc) or np.isnan(self.outcome_atr[entry - 1]):
            return False
        return self.gap_cum[end + 1] - self.gap_cum[w0] == 0

    def horizon_valid(self, entry, w0):
        return {h: self.base_valid(entry, w0, h) for h in HORIZONS}


def period_of(ts):
    t = pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts)
    for name, (a, b) in PERIODS.items():
        if a <= t < b:
            return name
    return None


def run_symbol(ctx, calendar=None):
    """
    All attempts for one symbol with their final status, including every rejected duplicate and filter.
    Order of the final filters (spec section 5): news, then data validity, then the 96-candle cooldown among
    the eligible events. Nothing here looks at prices after the entry candle.
    """
    att = ctx.attempts()
    if att.empty:
        return att
    att = att.copy()
    if "status" not in att.columns:
        att["status"] = ""
    att["status"] = att["status"].fillna("")
    att["period"] = [period_of(t) for t in att["parent_end"]]
    for i in att.index[att["status"] == "candidate"]:
        r = att.loc[i]
        if calendar is not None and bool(calendar.excluded(ctx.symbol, [r["entry_time"]])[0]):
            att.loc[i, "status"] = "news"
            continue
        att.loc[i, "stage"] = "news"
        if not ctx.base_valid(int(r["entry"]), int(r["w0"])):
            att.loc[i, "status"] = "invalid_data"
            continue
        att.loc[i, "stage"] = "base_valid"
    eligible = att[(att["status"] == "candidate") & (att["stage"] == "base_valid")]
    if eligible.empty:
        return att
    last = {}
    for i, r in eligible.sort_values(["direction", "entry"]).iterrows():
        d = int(r["direction"])
        if d in last and r["entry"] - last[d] < COOLDOWN:
            att.loc[i, "status"] = "cooldown_duplicate"
            continue
        last[d] = r["entry"]
        att.loc[i, "status"] = "event"
        att.loc[i, "stage"] = "cooldown"
    return att


# ------------------------------------------------------------------ reporting
def attrition(att, calendar_applied):
    """Survivors at each stage, split by the parent's period. Stage order follows the spec sequence."""
    if att.empty:
        zero = pd.DataFrame({"development": 0, "validation": 0}, index=STAGES)
        zero.insert(0, "stage", [STAGE_LABEL[s] for s in STAGES])
        return zero
    order = {"parents": 0, "parent_range": 1, "attempts": 2, "confluence": 3, "sr_zone": 4, "sweep": 5, "structure": 6,
             "fvg": 7, "entry": 8, "news": 9, "base_valid": 10, "cooldown": 11}
    rows = []
    for period in ("development", "validation"):
        a = att[att["period"] == period]
        parents = a[a["direction"] == 0]
        sides = a[a["direction"] != 0]
        reached = sides["stage"].map(order)
        counts = {
            "parents": len(parents), "parent_range": int((parents["stage"] == "parent_range").sum()),
            "attempts": len(sides),
        }
        for st in ("confluence", "sr_zone", "sweep", "structure", "fvg", "entry", "news", "base_valid", "cooldown"):
            counts[st] = int((reached >= order[st]).sum())
        rows.append(pd.Series(counts, name=period))
    table = pd.DataFrame(rows).T.reindex(STAGES)
    table.insert(0, "stage", [STAGE_LABEL[s] for s in table.index])
    return table


def event_counts(att, symbol_ctx):
    """Final unique events per symbol, direction, period, half-year and horizon (no prices involved)."""
    ev = att[att["status"] == "event"].copy()
    if ev.empty:
        return ev
    ev["period"] = [period_of(t) for t in ev["event_time"]]
    ev["half"] = [f"{pd.Timestamp(t).year}H{1 if pd.Timestamp(t).month <= 6 else 2}" for t in ev["event_time"]]
    for h in HORIZONS:
        ev[f"h{h}"] = [bool(symbol_ctx[s].base_valid(int(en), int(w0), h)) for s, en, w0 in zip(ev["symbol"], ev["entry"], ev["w0"])]
    return ev


def sample_verdict(ev):
    """INCONCLUSIVE unless every horizon meets the minimum in BOTH periods (spec section 5)."""
    rows = []
    for h in HORIZONS:
        for period, need in (("development", MIN_DEV), ("validation", MIN_VAL)):
            n = int(((ev["period"] == period) & ev[f"h{h}"]).sum()) if not ev.empty else 0
            rows.append({"horizon": h, "period": period, "events": n, "needed": need, "enough": n >= need})
    table = pd.DataFrame(rows)
    return table, ("SUFFICIENT" if table["enough"].all() else "INCONCLUSIVE")


# ------------------------------------------------------------------ orchestration and command line
def analyse(m15_frames, h1_frames, calendar=None):
    """Run every admitted symbol. Returns attempts, final events, attrition and the sample-size verdict."""
    ctxs, parts = {}, []
    for sym in m15_frames:
        ctxs[sym] = SymbolContext(sym, m15_frames[sym], h1_frames[sym])
        parts.append(run_symbol(ctxs[sym], calendar))
    frames = [p for p in parts if not p.empty]
    att = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["symbol", "direction", "stage", "status", "period", "pool", "zone"])
    ev = event_counts(att, ctxs) if not att.empty else att
    sample, verdict_text = sample_verdict(ev)
    return {"attempts": att, "events": ev, "attrition": attrition(att, calendar is not None),
            "sample": sample, "verdict": verdict_text}


def _print_breakdowns(att, ev):
    if att.empty:
        print("\n(no attempts at all)")
        return
    sides = att[att["direction"] != 0]
    print("\n--- Why attempts ended (all symbols, by the parent's period) ---")
    print(sides.groupby(["status", "period"]).size().unstack(fill_value=0).to_string())
    pooled = sides[sides["pool"].fillna("") != ""]
    print("\n--- Liquidity confluence types among attempts that passed it ---")
    print(pooled.groupby("pool").size().to_string() if len(pooled) else "none")
    zoned = sides[sides["zone"].fillna("") != ""]
    print("\n--- Support/resistance zone types among attempts that passed it ---")
    print(zoned.groupby("zone").size().to_string() if len(zoned) else "none")
    dup = sides[sides["status"] == "cooldown_duplicate"]
    kept = sides[sides["status"] == "event"]
    print("\n--- Cooldown: rejected duplicates vs kept events (by symbol and direction) ---")
    table = pd.DataFrame({"kept": kept.groupby(["symbol", "direction"]).size(),
                          "rejected_duplicates": dup.groupby(["symbol", "direction"]).size()}).fillna(0).astype(int)
    print(table.to_string() if len(table) else "none")
    if not ev.empty:
        print("\n--- Final events: symbol x direction x period ---")
        print(ev.groupby(["symbol", "direction", "period"]).size().unstack(fill_value=0).to_string())
        print("\n--- Final events: half-year ---")
        print(ev.groupby(["half", "period"]).size().unstack(fill_value=0).to_string())


def main(argv=None):
    import argparse
    from pathlib import Path
    from backend import config
    from backend.market_data import history

    ap = argparse.ArgumentParser()
    ap.add_argument("--calendar", default="research/news_calendar.csv")
    ap.add_argument("--clock-offset-file", default="research/clock_offset_hours.txt")
    args = ap.parse_args(argv)

    m15 = {s: history.load(s, "M15", require_complete=False) for s in config.SYMBOLS}
    gate = data_gate({s: df[df["time"] < RESEARCH_END] for s, df in m15.items()})
    print("=== DATA GATE (completeness against expected tradable M15 candles; limit "
          f"{GATE_LIMIT_PCT}%) ===")
    for s, g in gate.items():
        print(f"{s:<8} expected={g['expected']:>7} missing={g['missing']:>6} ({g['missing_pct']:>6.2f}%)  "
              f"{'ADMITTED' if g['admitted'] else 'EXCLUDED'}")
    universe = [s for s, g in gate.items() if g["admitted"]]
    if len(universe) < 6:
        raise SystemExit(f"Only {len(universe)} symbols pass the data gate (need at least 6).")
    print("Universe:", ", ".join(universe))

    calendar = None
    cal_path, off_path = Path(args.calendar), Path(args.clock_offset_file)
    if cal_path.exists() and off_path.exists():
        calendar = NewsCalendar.load(cal_path, float(off_path.read_text().strip()))
        print(f"NEWS FILTER APPLIED ({len(calendar.df)} scheduled releases)")
    else:
        print("NEWS FILTER NOT APPLIED: calendar and/or verified clock offset missing. "
              "These counts are PRELIMINARY and are before the news filter.")

    h1 = {s: history.load(s, "H1", require_complete=False) for s in universe}
    out = analyse({s: m15[s] for s in universe}, h1, calendar)
    print("\n=== ATTRITION (survivors at each stage; early stages are by parent date) ===")
    print(out["attrition"].to_string())
    _print_breakdowns(out["attempts"], out["events"])
    print("\n=== SAMPLE-SIZE CHECK (each horizon must meet the minimum in BOTH periods) ===")
    print(out["sample"].to_string(index=False))
    print(f"\nSAMPLE VERDICT: {out['verdict']}"
          + ("   (preliminary: news filter not applied)" if calendar is None else ""))
    print("No returns were calculated. The holdout (2025 onward) was not loaded.")

    Path("research").mkdir(exist_ok=True)
    out["attempts"].to_csv("research/h6_attempts.csv", index=False)
    out["attrition"].to_csv("research/h6_attrition.csv")
    out["sample"].to_csv("research/h6_sample_check.csv", index=False)


if __name__ == "__main__":
    main()
