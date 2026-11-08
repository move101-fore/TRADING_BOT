from pathlib import Path

import pandas as pd

from backend import config
from backend.market_data import mt5_connector as mt5c

DATA_DIR = Path("data")


def csv_path(symbol, timeframe="H1"):
    return DATA_DIR / f"{symbol}_{timeframe}.csv"


def download(symbol, timeframe="H1", count=40000):
    """Download completed candles from MT5 and save them to CSV."""
    df = mt5c.get_candles(symbol, timeframe, count)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(csv_path(symbol, timeframe), index=False)
    return df


def load(symbol, timeframe="H1"):
    """Load saved candles from CSV (no MT5 needed)."""
    path = csv_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"No saved data for {symbol} {timeframe}. Run the downloader first.")
    df = pd.read_csv(path, parse_dates=["time"])
    return df


if __name__ == "__main__":
    mt5c.connect()
    try:
        for symbol in config.SYMBOLS:
            try:
                df = download(symbol, "H1")
                print(f"{symbol}: {len(df)} candles, {df['time'].iloc[0]} -> {df['time'].iloc[-1]}")
            except Exception as exc:
                print(f"{symbol}: FAILED - {exc}")
    finally:
        mt5c.disconnect()
