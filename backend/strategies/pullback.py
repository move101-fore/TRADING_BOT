from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.strategies.base_strategy import BaseStrategy


@dataclass
class PullbackConfig:
    adx_min: float = 20.0
    slope_bars: int = 10            # EMA50 must be rising/falling over this many bars
    touch_atr: float = 0.25         # rejection candle must reach EMA20 within this x ATR
    max_pierce_atr: float = 0.50    # ...but not close through EMA50 zone by more than this
    min_wick_ratio: float = 0.40    # rejection wick / candle range
    min_depth_atr: float = 1.00     # pullback depth from the recent extreme
    depth_lookback: int = 10
    rsi_buy_min: float = 40.0
    rsi_buy_max: float = 70.0
    rsi_sell_min: float = 30.0
    rsi_sell_max: float = 60.0
    sr_tolerance_atr: float = 0.50  # confluence with a confirmed swing level
    require_sr_confluence: bool = False
    sl_buffer_atr: float = 0.25
    min_sl_atr: float = 0.50
    max_sl_atr: float = 3.00
    rr: float = 2.0
    max_hold: int = 48


class Pullback(BaseStrategy):
    name = "pullback"

    def __init__(self, config=None):
        self.cfg = config or PullbackConfig()

    def _signals(self, d):
        c = self.cfg

        # Rejection candle = the PREVIOUS completed bar (shift 1).
        # Confirmation candle = the CURRENT completed bar.
        r_open, r_high = d["open"].shift(1), d["high"].shift(1)
        r_low, r_close = d["low"].shift(1), d["close"].shift(1)
        r_range = (r_high - r_low).replace(0, np.nan)
        e20, e50 = d["ema_20"].shift(1), d["ema_50"].shift(1)
        atr1 = d["atr_14"].shift(1)

        lower_wick = (np.minimum(r_open, r_close) - r_low) / r_range
        upper_wick = (r_high - np.maximum(r_open, r_close)) / r_range

        recent_high = d["high"].shift(2).rolling(c.depth_lookback, min_periods=c.depth_lookback).max()
        recent_low = d["low"].shift(2).rolling(c.depth_lookback, min_periods=c.depth_lookback).min()

        # Confirmed swing levels only (already lookahead-free), previous two of each.
        sw_lo = d["swing_low"].dropna()
        sw_hi = d["swing_high"].dropna()
        lo1 = d["swing_low"].ffill().shift(1)
        lo2 = sw_lo.shift(1).reindex(d.index).ffill().shift(1)
        hi1 = d["swing_high"].ffill().shift(1)
        hi2 = sw_hi.shift(1).reindex(d.index).ffill().shift(1)
        tol = c.sr_tolerance_atr * atr1
        support_near = ((r_low - lo1).abs() <= tol) | ((r_low - lo2).abs() <= tol)
        resist_near = ((r_high - hi1).abs() <= tol) | ((r_high - hi2).abs() <= tol)

        bull_trend = (
            (d["ema_50"] > d["ema_200"])
            & (d["ema_50"] > d["ema_50"].shift(c.slope_bars))
            & (d["adx"] >= c.adx_min)
            & (d["plus_di"] > d["minus_di"])
        )
        bear_trend = (
            (d["ema_50"] < d["ema_200"])
            & (d["ema_50"] < d["ema_50"].shift(c.slope_bars))
            & (d["adx"] >= c.adx_min)
            & (d["minus_di"] > d["plus_di"])
        )

        bull = (
            bull_trend
            & (r_low <= e20 + c.touch_atr * atr1)
            & (r_low >= e50 - c.max_pierce_atr * atr1)
            & (r_close > e50)
            & (lower_wick >= c.min_wick_ratio)
            & (r_close >= r_low + 0.5 * r_range)
            & ((recent_high - r_low) >= c.min_depth_atr * atr1)
            & (d["close"] > r_high) & (d["close"] > d["open"])
            & d["rsi_14"].between(c.rsi_buy_min, c.rsi_buy_max)
        )
        bear = (
            bear_trend
            & (r_high >= e20 - c.touch_atr * atr1)
            & (r_high <= e50 + c.max_pierce_atr * atr1)
            & (r_close < e50)
            & (upper_wick >= c.min_wick_ratio)
            & (r_close <= r_high - 0.5 * r_range)
            & ((r_high - recent_low) >= c.min_depth_atr * atr1)
            & (d["close"] < r_low) & (d["close"] < d["open"])
            & d["rsi_14"].between(c.rsi_sell_min, c.rsi_sell_max)
        )
        if c.require_sr_confluence:
            bull = bull & support_near
            bear = bear & resist_near

        bull = bull.fillna(False)
        bear = bear.fillna(False)

        # Stop beyond the pullback extreme (rejection or confirmation candle).
        bull_sl = d["close"] - (np.minimum(r_low, d["low"]) - c.sl_buffer_atr * atr1)
        bear_sl = (np.maximum(r_high, d["high"]) + c.sl_buffer_atr * atr1) - d["close"]
        bull = bull & (bull_sl >= c.min_sl_atr * atr1) & (bull_sl <= c.max_sl_atr * atr1)
        bear = bear & (bear_sl >= c.min_sl_atr * atr1) & (bear_sl <= c.max_sl_atr * atr1)

        bull_event = bull & ~bull.shift(1, fill_value=False)
        bear_event = bear & ~bear.shift(1, fill_value=False)

        out = pd.DataFrame(index=d.index)
        out["signal"] = np.where(bull_event, 1, np.where(bear_event, -1, 0))
        has = out["signal"] != 0
        out["sl_dist"] = np.where(bull_event, bull_sl, np.where(bear_event, bear_sl, np.nan))
        out["rr"] = np.where(has, c.rr, np.nan)
        out["max_hold"] = np.where(has, c.max_hold, 0)
        out["reason"] = ""

        for i in out.index[has]:
            r = d.loc[i]
            if out.at[i, "signal"] == 1:
                text = (
                    f"Pullback in an uptrend: EMA50 above EMA200 and rising, ADX {r['adx']:.1f}. "
                    f"Price retraced to the EMA20 zone and a bullish rejection candle "
                    f"(long lower wick) closed above EMA50; the next candle confirmed by "
                    f"closing above the rejection high. RSI {r['rsi_14']:.1f}."
                )
                if bool(support_near.at[i]):
                    text += " The rejection low also sits on a confirmed swing-low support."
            else:
                text = (
                    f"Pullback in a downtrend: EMA50 below EMA200 and falling, ADX {r['adx']:.1f}. "
                    f"Price retraced to the EMA20 zone and a bearish rejection candle "
                    f"(long upper wick) closed below EMA50; the next candle confirmed by "
                    f"closing below the rejection low. RSI {r['rsi_14']:.1f}."
                )
                if bool(resist_near.at[i]):
                    text += " The rejection high also sits on a confirmed swing-high resistance."
            out.at[i, "reason"] = text
        return out
