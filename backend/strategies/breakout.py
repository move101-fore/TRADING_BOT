from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.strategies.base_strategy import BaseStrategy


@dataclass
class BreakoutConfig:
    range_bars: int = 24            # lookback that defines the range
    max_range_atr: float = 4.0      # range must be tight: width <= this x ATR
    break_buffer_atr: float = 0.10  # close must clear the level by this x ATR
    min_body_ratio: float = 0.50    # candle body / candle range
    min_range_atr: float = 1.20     # breakout candle range >= this x ATR
    close_location: float = 0.70    # close must sit in outer 30% of the candle
    require_retest: bool = True     # enter only after a retest of the level
    retest_window: int = 12         # bars allowed for the retest
    retest_touch_atr: float = 0.25  # how close price must come to the level
    invalidate_atr: float = 0.50    # close back through level by this = failed
    sl_buffer_atr: float = 0.50     # stop sits this x ATR beyond the level
    rr: float = 2.0
    max_hold: int = 48


class Breakout(BaseStrategy):
    name = "breakout"

    def __init__(self, config=None):
        self.cfg = config or BreakoutConfig()

    def _signals(self, d):
        c = self.cfg
        n = len(d)

        # Range built from PRIOR bars only (shift(1)) -> no lookahead.
        rng_high = d["high"].shift(1).rolling(c.range_bars, min_periods=c.range_bars).max()
        rng_low = d["low"].shift(1).rolling(c.range_bars, min_periods=c.range_bars).min()
        atr_prev = d["atr_14"].shift(1)
        consolidated = (rng_high - rng_low) <= c.max_range_atr * atr_prev

        candle_range = (d["high"] - d["low"]).replace(0, np.nan)
        body_ratio = (d["close"] - d["open"]).abs() / candle_range
        close_loc = (d["close"] - d["low"]) / candle_range
        strong = (body_ratio >= c.min_body_ratio) & (candle_range >= c.min_range_atr * atr_prev)

        prev_inside_up = d["close"].shift(1) <= rng_high
        prev_inside_dn = d["close"].shift(1) >= rng_low

        bull_break = (
            consolidated & strong & prev_inside_up
            & (d["close"] > rng_high + c.break_buffer_atr * atr_prev)
            & (close_loc >= c.close_location)
        ).fillna(False).to_numpy()
        bear_break = (
            consolidated & strong & prev_inside_dn
            & (d["close"] < rng_low - c.break_buffer_atr * atr_prev)
            & (close_loc <= 1 - c.close_location)
        ).fillna(False).to_numpy()

        o = d["open"].to_numpy(); h = d["high"].to_numpy()
        l = d["low"].to_numpy(); cl = d["close"].to_numpy()
        atr_v = d["atr_14"].to_numpy()
        up_lvl = rng_high.to_numpy(); dn_lvl = rng_low.to_numpy()
        adx_v = d["adx"].to_numpy()

        signal = np.zeros(n, dtype=int)
        sl_dist = np.full(n, np.nan)
        reasons = [""] * n

        pend_dir = 0; pend_lvl = np.nan; pend_start = -1

        for i in range(n):
            a = atr_v[i]
            if np.isnan(a):
                continue

            if bull_break[i] or bear_break[i]:
                direction = 1 if bull_break[i] else -1
                level = up_lvl[i] if direction == 1 else dn_lvl[i]
                if not c.require_retest:
                    signal[i] = direction
                    sl_dist[i] = abs(cl[i] - level) + c.sl_buffer_atr * a
                    reasons[i] = self._reason(direction, level, a, False, adx_v[i])
                    pend_dir = 0
                else:
                    pend_dir, pend_lvl, pend_start = direction, level, i
                continue

            if pend_dir == 0:
                continue
            age = i - pend_start
            if age > c.retest_window:
                pend_dir = 0
                continue
            if pend_dir == 1:
                if cl[i] < pend_lvl - c.invalidate_atr * a:
                    pend_dir = 0                      # failed breakout
                elif l[i] <= pend_lvl + c.retest_touch_atr * a and cl[i] > pend_lvl and cl[i] > o[i]:
                    signal[i] = 1
                    sl_dist[i] = (cl[i] - pend_lvl) + c.sl_buffer_atr * a
                    reasons[i] = self._reason(1, pend_lvl, a, True, adx_v[i])
                    pend_dir = 0
            else:
                if cl[i] > pend_lvl + c.invalidate_atr * a:
                    pend_dir = 0
                elif h[i] >= pend_lvl - c.retest_touch_atr * a and cl[i] < pend_lvl and cl[i] < o[i]:
                    signal[i] = -1
                    sl_dist[i] = (pend_lvl - cl[i]) + c.sl_buffer_atr * a
                    reasons[i] = self._reason(-1, pend_lvl, a, True, adx_v[i])
                    pend_dir = 0

        out = pd.DataFrame(index=d.index)
        out["signal"] = signal
        has = out["signal"] != 0
        out["sl_dist"] = np.where(has, sl_dist, np.nan)
        out["rr"] = np.where(has, c.rr, np.nan)
        out["max_hold"] = np.where(has, c.max_hold, 0)
        out["reason"] = reasons
        return out

    def _reason(self, direction, level, a, retested, adx_value):
        c = self.cfg
        side = "above resistance" if direction == 1 else "below support"
        text = (
            f"Range breakout: a tight {c.range_bars}-bar consolidation "
            f"(width within {c.max_range_atr:.0f} x ATR) was broken by a strong candle "
            f"that closed {side} at {level:.5f}, with volatility expansion "
            f"(range at least {c.min_range_atr:.1f} x ATR) and a decisive body. "
        )
        if retested:
            text += "Price then retested the broken level and a confirmation candle closed back in the breakout direction. "
        text += f"ATR {a:.5f}; ADX {adx_value:.1f}."
        return text
