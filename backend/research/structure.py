import numpy as np
import pandas as pd

from backend.market_data.indicators import add_indicators

STRUCTURE_VERSION = "H5-v1"


def structure_events(d, buffer_atr=0.10, right=3):
    """
    Market-structure events on H1 candles, from CONFIRMED swing pivots only (a pivot is known
    `right` bars after it forms), so nothing here uses future candles.

    Trend state: UP = last two confirmed swing highs AND lows are rising; DOWN = both falling.
      BOS   (continuation): a close beyond the last swing extreme in the trend direction.
      CHoCH (reversal)    : a close beyond the last opposite swing extreme.
    A close must clear the level by `buffer_atr` x ATR. Each swing level can trigger only once.
    Event time = the completed candle; the earliest entry is the NEXT candle's open.
    Returns a DataFrame: idx, time, kind, direction, trend, level.
    """
    if "swing_high" not in d.columns:
        d = add_indicators(d)
    close = d["close"].to_numpy()
    atr = d["atr_14"].to_numpy()
    sh = d["swing_high"].to_numpy()
    sl = d["swing_low"].to_numpy()
    times = d["time"].to_numpy() if "time" in d.columns else d.index.to_numpy()

    highs, lows = [], []                 # (pivot index, price)
    used_hi, used_lo = set(), set()
    rows = []
    for i in range(len(d)):
        if not np.isnan(sh[i]):
            highs.append((i - right, sh[i]))
        if not np.isnan(sl[i]):
            lows.append((i - right, sl[i]))
        if len(highs) < 2 or len(lows) < 2 or np.isnan(atr[i]):
            continue
        (hi_i, hi), (lo_i, lo) = highs[-1], lows[-1]
        up = hi > highs[-2][1] and lo > lows[-2][1]
        down = hi < highs[-2][1] and lo < lows[-2][1]
        buf = buffer_atr * atr[i]
        if up:
            if close[i] > hi + buf and hi_i not in used_hi:
                used_hi.add(hi_i); rows.append((i, times[i], "BOS", 1, 1, hi))
            elif close[i] < lo - buf and lo_i not in used_lo:
                used_lo.add(lo_i); rows.append((i, times[i], "CHoCH", -1, 1, lo))
        elif down:
            if close[i] < lo - buf and lo_i not in used_lo:
                used_lo.add(lo_i); rows.append((i, times[i], "BOS", -1, -1, lo))
            elif close[i] > hi + buf and hi_i not in used_hi:
                used_hi.add(hi_i); rows.append((i, times[i], "CHoCH", 1, -1, hi))
    return pd.DataFrame(rows, columns=["idx", "time", "kind", "direction", "trend", "level"])
