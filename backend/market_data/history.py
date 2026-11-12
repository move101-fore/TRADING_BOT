import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend import config
from backend.market_data.mt5_connector import MT5Connector, MT5Error, validate_candles

DATA_DIR = Path("data")
START = datetime(2022, 1, 1, tzinfo=timezone.utc)        # H1/M15/M1 history is only reliable from here
TF_MINUTES = {"M1": 1, "M15": 15, "H1": 60}
GAP_HOURS = {"M1": 0.5, "M15": 1.0, "H1": 2.0}           # a gap longer than this is worth reporting
COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread"]


def path_for(symbol, tf):
    return DATA_DIR / f"{symbol}_{tf}_2022.csv.gz"


def windows(start, end, days=31):
    a = start
    while a < end:
        b = min(a + timedelta(days=days), end)
        yield a, b
        a = b


def download_since(connector, symbol, tf, start=START, end=None, tries=3, sleep=time.sleep):
    """Download COMPLETED candles from `start` in monthly chunks. Returns (DataFrame, empty_chunks)."""
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
    if not frames:
        raise MT5Error(f"{symbol} {tf}: no candles returned since {start.date()}")
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[COLUMNS].drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    cutoff = pd.Timestamp(end) - pd.Timedelta(minutes=TF_MINUTES[tf])
    df = df[df["time"] <= cutoff].reset_index(drop=True)          # drop a candle that is still forming
    return validate_candles(df), empty


def save(df, symbol, tf):
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path_for(symbol, tf), index=False, compression="gzip")


def load(symbol, tf):
    p = path_for(symbol, tf)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run: python -m backend.market_data.history")
    return pd.read_csv(p, parse_dates=["time"])


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


def main(argv=None):
    """Usage: python -m backend.market_data.history [H1] [M15] [M1]   (default: all three)"""
    tfs = [a.upper() for a in (sys.argv[1:] if argv is None else argv)] or ["H1", "M15", "M1"]
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
                gaps, by_year = gap_report(df, tf)
                big = int((gaps["hours"] > 24).sum()) if len(gaps) else 0
                print(f"{symbol} {tf}: {len(df):>8} candles {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()}"
                      f"  empty_chunks={empty}  gaps={len(gaps)}  gaps>24h={big}", flush=True)
    except MT5Error as exc:
        print("REJECTED:", exc)
    finally:
        c.disconnect()


if __name__ == "__main__":
    main()
