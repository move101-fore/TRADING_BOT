import numpy as np
import pandas as pd


def sma(series, period):
    return series.rolling(period, min_periods=period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close, period=14):
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0)


def true_range(df):
    prev_close = df["close"].shift(1)
    parts = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr(df, period=14):
    """Wilder's Average True Range."""
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df, period=14):
    """Wilder's ADX with +DI and -DI. Returns a DataFrame."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    atr_ = atr(df, period)
    alpha = 1 / period
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_ = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_})


def swings(df, left=3, right=3):
    """
    Swing highs/lows with NO lookahead.
    A pivot at bar i needs `right` later bars to confirm it, so its value is
    reported on bar i+right (the first bar where it is actually knowable).
    Returns two columns: swing_high, swing_low (NaN where no pivot confirmed).
    """
    window = left + right + 1
    center_high = df["high"].shift(right)
    center_low = df["low"].shift(right)
    roll_max = df["high"].rolling(window, min_periods=window).max()
    roll_min = df["low"].rolling(window, min_periods=window).min()
    swing_high = center_high.where(center_high == roll_max)
    swing_low = center_low.where(center_low == roll_min)
    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low})


def add_indicators(df):
    """Return a copy of df with all strategy indicators added."""
    out = df.copy()
    out["ema_20"] = ema(out["close"], 20)
    out["ema_50"] = ema(out["close"], 50)
    out["ema_200"] = ema(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)
    out["atr_14"] = atr(out, 14)
    out = out.join(adx(out, 14))
    out = out.join(swings(out, 3, 3))
    return out
