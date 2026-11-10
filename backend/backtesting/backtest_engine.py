import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.market_data.indicators import atr as atr_indicator

SYMBOL_INFO_PATH = Path("data") / "symbols.json"
SWAP_INFO_PATH = Path("data") / "swaps.json"


@dataclass
class BacktestConfig:
    slippage_atr: float = 0.02          # slippage per fill, as a fraction of ATR
    commission_r: float = 0.0           # flat commission per trade, in R
    max_spread_frac_of_sl: float = 0.25  # skip if spread > this fraction of the stop distance
    max_gap_hours: float = 3.0          # skip entries after a weekend/holiday gap
    rollover_hour: int = 22             # hour of the candle clock at which overnight swap is charged


def load_point(symbol):
    """Price value of one point for a symbol, from data/symbols.json."""
    if not SYMBOL_INFO_PATH.exists():
        raise FileNotFoundError(
            "data/symbols.json missing. Run: python -m backend.market_data.symbol_info"
        )
    info = json.loads(SYMBOL_INFO_PATH.read_text())
    return float(info[symbol]["point"])


def load_swap(symbol):
    """Swap settings for a symbol from data/swaps.json (written by swap_info.py). Points mode only."""
    if not SWAP_INFO_PATH.exists():
        raise FileNotFoundError(
            "data/swaps.json missing. Run: python -m backend.market_data.swap_info"
        )
    info = json.loads(SWAP_INFO_PATH.read_text())[symbol]
    if int(info["swap_mode"]) != 1:
        raise NotImplementedError(
            f"{symbol}: swap_mode {info['swap_mode']} is not supported (only 'points' mode); "
            "refusing to guess the cost"
        )
    return {
        "long_pts": float(info["swap_long"]),
        "short_pts": float(info["swap_short"]),
        "triple_weekday": (int(info["triple_day"]) - 1) % 7,   # MT5: 0=Sunday ... 3=Wednesday -> pandas: Wednesday=2
    }


def count_swap_nights(start, end, rollover_hour=22, triple_weekday=2):
    """
    Weighted number of rollovers in (start, end]. Only Monday to Friday rollovers are charged,
    and the triple-swap weekday counts three times (it covers the weekend).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total = 0
    day = start.normalize()
    while day <= end.normalize():
        instant = day + pd.Timedelta(hours=rollover_hour)
        if start < instant <= end and day.weekday() < 5:
            total += 3 if day.weekday() == triple_weekday else 1
        day += pd.Timedelta(days=1)
    return total


def backtest(df, signals, point, cfg=None, atr=None, strategy="", symbol="", swap=None):
    """
    Event-driven backtest of one strategy on one symbol.

    Rules (no lookahead):
      * a signal is judged on a COMPLETED candle i; the trade enters at the
        OPEN of candle i+1.
      * candles are BID prices. Buys fill at ask (open + spread); sells fill at
        bid and close at ask. The recorded per-candle spread is used.
      * slippage (fraction of ATR) is applied against us on entry and on stops.
      * stop-loss and take-profit are checked candle by candle; if BOTH are
        touched inside one candle the stop is assumed to hit first (pessimistic).
      * a gap through the stop fills at the worse price.
      * if neither is hit within max_hold bars, exit at that bar's close.
      * one position at a time per strategy and symbol.
      * optional overnight swap (swap = dict from load_swap): charged per rollover crossed,
        in points per night, signed from the account's point of view (negative = cost).
    Returns a DataFrame of trades with the result in R multiples.
    """
    cfg = cfg or BacktestConfig()
    n = len(df)
    t = df["time"].to_numpy()
    o = df["open"].to_numpy(dtype=float); h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float); c = df["close"].to_numpy(dtype=float)
    spr = df["spread"].to_numpy(dtype=float) * point
    if atr is None:
        atr = atr_indicator(df)
    atr_v = np.asarray(atr, dtype=float)

    sig = signals["signal"].to_numpy()
    sld = signals["sl_dist"].to_numpy(dtype=float)
    rrs = signals["rr"].to_numpy(dtype=float)
    mhs = signals["max_hold"].to_numpy()
    reasons = signals["reason"].to_numpy()

    trades = []
    last_exit = -1
    for i in np.flatnonzero(sig != 0):
        e = i + 1
        if e >= n or e <= last_exit:
            continue
        gap_h = (t[e] - t[i]) / np.timedelta64(1, "h")
        if gap_h > cfg.max_gap_hours:
            continue
        d = int(sig[i]); sl_d = sld[i]; a = atr_v[i]
        if np.isnan(sl_d) or sl_d <= 0 or np.isnan(a):
            continue
        sp = spr[e]
        if sp > cfg.max_spread_frac_of_sl * sl_d:
            continue
        slip = cfg.slippage_atr * a
        rr = rrs[i]
        mid = o[e]

        if d == 1:
            fill = mid + sp + slip
            sl = mid - sl_d
            tp = mid + rr * sl_d
            risk = fill - sl
        else:
            fill = mid - slip
            sl = mid + sl_d
            tp = mid - rr * sl_d
            risk = sl - fill

        end = min(n, e + int(mhs[i]))
        outcome = None
        exit_price = np.nan
        j = e
        for j in range(e, end):
            if d == 1:
                if l[j] <= sl:
                    px = min(sl, o[j]) if j > e else sl
                    exit_price, outcome = px - slip, "sl"
                    break
                if h[j] >= tp:
                    exit_price, outcome = tp, "tp"
                    break
            else:
                s_j = spr[j]
                if h[j] + s_j >= sl:
                    px = max(sl, o[j] + s_j) if j > e else sl
                    exit_price, outcome = px + slip, "sl"
                    break
                if l[j] + s_j <= tp:
                    exit_price, outcome = tp, "tp"
                    break
        if outcome is None:
            j = end - 1
            exit_price = c[j] if d == 1 else c[j] + spr[j]
            outcome = "time"

        pnl = (exit_price - fill) if d == 1 else (fill - exit_price)
        nights, swap_price = 0, 0.0
        if swap is not None:
            exit_eff = pd.Timestamp(t[j]) + pd.Timedelta(hours=1)     # end of the exit candle
            nights = count_swap_nights(t[e], exit_eff, cfg.rollover_hour, swap["triple_weekday"])
            pts = swap["long_pts"] if d == 1 else swap["short_pts"]
            swap_price = pts * point * nights
        r = (pnl + swap_price) / risk - cfg.commission_r
        trades.append({
            "strategy": strategy, "symbol": symbol, "direction": d,
            "signal_time": t[i], "entry_time": t[e], "exit_time": t[j],
            "entry": fill, "sl": sl, "tp": tp, "exit": exit_price,
            "outcome": outcome, "bars": j - e + 1, "r": r, "reason": reasons[i],
            "nights": nights, "swap_r": swap_price / risk,
        })
        last_exit = j

    cols = ["strategy", "symbol", "direction", "signal_time", "entry_time", "exit_time",
            "entry", "sl", "tp", "exit", "outcome", "bars", "r", "reason", "nights", "swap_r"]
    return pd.DataFrame(trades, columns=cols)
