
import MetaTrader5 as mt5
import pandas as pd

from backend import config

TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def connect():
    """Connect to the MT5 terminal and log in. Returns account info."""
    ok = mt5.initialize(
        login=config.MT5_LOGIN,
        password=config.MT5_PASSWORD,
        server=config.MT5_SERVER,
    )
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    return mt5.account_info()


def disconnect():
    mt5.shutdown()


def get_candles(symbol, timeframe="H1", count=1000):
    """
    Return the last `count` COMPLETED candles as a DataFrame.
    start_pos=1 skips the candle still forming, so we never use
    incomplete data (no repainting).
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    broker_symbol = config.SYMBOLS.get(symbol, symbol)
    if not mt5.symbol_select(broker_symbol, True):
        raise RuntimeError(f"Symbol not available: {broker_symbol}")

    rates = mt5.copy_rates_from_pos(broker_symbol, TIMEFRAMES[timeframe], 1, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {broker_symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
| Set-Content backend\market_data\mt5_connector.py
