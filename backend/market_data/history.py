import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend import config
from backend.market_data.mt5_connector import MT5Connector, MT5Error, validate_candles

DATA_DIR = Path("data")
MANIFEST = DATA_DIR / "history_manifest.json"
START = datetime(2022, 1, 1, tzinfo=timezone.utc)        # H1/M15/M1 history is only reliable from here
TF_MINUTES = {"M1": 1, "M15": 15, "H1": 60}
GAP_HOURS = {"M1": 0.5, "M15": 1.0, "H1": 2.0}           # a gap longer than this is worth reporting
COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
MAX_MISSING_PCT = 5.0                                    # research code refuses history with more missing time
MONTH_ALERT_HOURS = 48.0


def path_for(symbol, tf):
    return DATA_DIR / f"{symbol}_{tf}_2022.csv.gz"


def windows(start, end, days=31):
    a = start
    while a < end:
        b = min(a + timedelta(days=days), end)
        yield a, b
        a = b


def download_since(connector, symbol, tf, start=START, end=None, tries=3, sleep=time.sleep, allow_missing=False):
    """
    Download COMPLETED candles from `start` in monthly chunks. Returns (DataFrame, empty_chunks).
    A monthly chunk with no candles at all means history is missing, so by default that is an error.
    """
    api = connector.api
    broker = connector.broker_symbol(symbol)
    api.symbol_select(broker, True)
    tf_const = getattr(api, f"TIMEFRAME_{tf}")
    end = end or datetime.now(timezone.utc)
    frames, empty = [], 0
    for a, b in windows(start, end):
        rates = None
        for attempt in range(tries):
            rates = api.copy_rates_range(broker, tf_const, a, b)
            if rates is not None and len(rates) > 0:
                break
            sleep(1.0 * (attempt + 1))                  # MT5 may still be syncing history from the server
        if rates is None or len(rates) == 0:
            empty += 1
            continue
        frames.append(pd.DataFrame(rates))
    if empty and not allow_missing:
        raise MT5Error(f"{symbol} {tf}: {empty} monthly chunk(s) returned no candles; history is incomplete "
                       "(run again once MT5 has synced, or pass allow_missing=True)")
    if not frames:
        raise MT5Error(f"{symbol} {tf}: no candles returned since {start.date()}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[COLUMNS].drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    cutoff = pd.Timestamp(end) - pd.Timedelta(minutes=TF_MINUTES[tf])
    df = df[df["time"] <= cutoff].reset_index(drop=True)          # drop a candle that is still forming
    return validate_candles(df), empty


def gap_report(df, tf):
    """
    Gaps in a candle series, excluding the normal weekend break. Returns (gaps, by_year):
      gaps    : one row per unexpected gap (start, end, hours)
      by_year : count of gaps and total missing hours per year
    """
    t = df["time"].reset_index(drop=True)
    diff = t.diff()
    hours = diff.dt.total_seconds() / 3600.0
    prev = t.shift(1)
    is_weekend = (hours >= 36) & (hours <= 72) & (prev.dt.weekday == 4)
    unexpected = (hours > GAP_HOURS[tf]) & ~is_weekend
    gaps = pd.DataFrame({"start": prev[unexpected], "end": t[unexpected], "hours": hours[unexpected]})
    if gaps.empty:
        return gaps.reset_index(drop=True), pd.DataFrame(columns=["year", "gaps", "missing_hours"])
    gaps["year"] = gaps["start"].dt.year
    by_year = gaps.groupby("year").agg(gaps=("hours", "size"), missing_hours=("hours", "sum")).reset_index()
    return gaps.drop(columns="year").reset_index(drop=True), by_year


def coverage_stats(df, tf):
    """Measured completeness of a candle series: missing time as a percent of expected trading time, and bad months."""
    gaps, _ = gap_report(df, tf)
    step = TF_MINUTES[tf] / 60.0
    span = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 3600.0
    expected = span * 120.0 / 168.0                    # forex trades about 120 of every 168 hours
    if gaps.empty:
        missing, months = 0.0, []
    else:
        gaps = gaps.assign(missing=gaps["hours"] - step, month=gaps["start"].dt.strftime("%Y-%m"))
        missing = float(gaps["missing"].sum())
        by_month = gaps.groupby("month")["missing"].sum()
        months = sorted(by_month[by_month > MONTH_ALERT_HOURS].index.tolist())
    return {
        "bars": int(len(df)), "first": df["time"].iloc[0].isoformat(), "last": df["time"].iloc[-1].isoformat(),
        "gaps": int(len(gaps)), "missing_hours": round(missing, 1),
        "missing_pct": round(100.0 * missing / expected, 2) if expected > 0 else 100.0,
        "bad_months": months,
    }


def read_manifest(path=None):
    path = Path(path or MANIFEST)
    return json.loads(path.read_text()) if path.exists() else {}


def update_manifest(key, stats, path=None):
    path = Path(path or MANIFEST)
    data = read_manifest(path)
    data[key] = dict(stats, saved=datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save(df, symbol, tf):
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path_for(symbol, tf), index=False, compression="gzip")


def load(symbol, tf, require_complete=True, manifest_path=None):
    """Load saved candles. By default refuses history that was never verified or is too incomplete."""
    p = path_for(symbol, tf)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run: python -m backend.market_data.history")
    if require_complete:
        entry = read_manifest(manifest_path).get(f"{symbol}_{tf}")
        if entry is None:
            raise MT5Error(f"{symbol} {tf}: no completeness record. Re-run: python -m backend.market_data.history {tf}")
        if entry["missing_pct"] > MAX_MISSING_PCT:
            raise MT5Error(f"{symbol} {tf}: {entry['missing_pct']}% of expected time is missing "
                           f"(limit {MAX_MISSING_PCT}%); refusing to use it for research")
    return pd.read_csv(p, parse_dates=["time"])


def report_coverage():
    """Measure completeness of every saved file (not a sample week) and print it."""
    print(f"{'file':<14}{'bars':>9}  {'first':<11}{'last':<11}{'gaps':>6}{'missing%':>10}  months with more than {MONTH_ALERT_HOURS:.0f}h missing")
    for symbol in config.SYMBOLS:
        for tf in ("H1", "M15", "M1"):
            p = path_for(symbol, tf)
            if not p.exists():
                continue
            s = coverage_stats(pd.read_csv(p, parse_dates=["time"]), tf)
            print(f"{symbol + ' ' + tf:<14}{s['bars']:>9}  {s['first'][:10]:<11}{s['last'][:10]:<11}{s['gaps']:>6}"
                  f"{s['missing_pct']:>9.2f}%  {', '.join(s['bad_months']) or '-'}")


def main(argv=None):
    """Usage: python -m backend.market_data.history [H1] [M15] [M1]   or   ... report"""
    args = [a for a in (sys.argv[1:] if argv is None else argv)]
    if args and args[0].lower() == "report":
        return report_coverage()
    tfs = [a.upper() for a in args] or ["H1", "M15", "M1"]
    bad = [t for t in tfs if t not in TF_MINUTES]
    if bad:
        raise SystemExit(f"Unknown timeframe(s): {bad}. Use H1, M15 and/or M1.")
    c = MT5Connector()
    try:
        c.connect()
        for symbol in config.SYMBOLS:
            for tf in tfs:
                try:
                    df, empty = download_since(c, symbol, tf)
                except MT5Error as exc:
                    print(f"{symbol} {tf}: FAILED - {exc}")
                    continue
                save(df, symbol, tf)
                stats = coverage_stats(df, tf)
                update_manifest(f"{symbol}_{tf}", dict(stats, empty_chunks=empty))
                flag = "" if stats["missing_pct"] <= MAX_MISSING_PCT else "   <-- TOO INCOMPLETE FOR RESEARCH"
                print(f"{symbol} {tf}: {stats['bars']:>8} candles {stats['first'][:10]} -> {stats['last'][:10]}"
                      f"  gaps={stats['gaps']}  missing={stats['missing_pct']}%{flag}", flush=True)
    except MT5Error as exc:
        print("REJECTED:", exc)
    finally:
        c.disconnect()


if __name__ == "__main__":
    main()
