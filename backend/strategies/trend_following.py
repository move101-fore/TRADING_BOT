from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.strategies.base_strategy import BaseStrategy


@dataclass
class TrendFollowingConfig:
    adx_min: float = 20.0
    rsi_buy_min: float = 50.0
    rsi_buy_max: float = 70.0
    rsi_sell_min: float = 30.0
    rsi_sell_max: float = 50.0
    sl_atr_mult: float = 1.5
    rr: float = 2.0
    max_hold: int = 48


class TrendFollowing(BaseStrategy):
    name = "trend_following"

    def __init__(self, config=None):
        self.cfg = config or TrendFollowingConfig()

    def _signals(self, d):
        c = self.cfg

        # Market structure from CONFIRMED swings only (no lookahead).
        highs = d["swing_high"].dropna()
        lows = d["swing_low"].dropna()
        last_high = highs.reindex(d.index).ffill()
        prev_high = highs.shift(1).reindex(d.index).ffill()
        last_low = lows.reindex(d.index).ffill()
        prev_low = lows.shift(1).reindex(d.index).ffill()
        higher_highs = last_high > prev_high
        higher_lows = last_low > prev_low
        lower_highs = last_high < prev_high
        lower_lows = last_low < prev_low

        bull = (
            (d["ema_20"] > d["ema_50"])
            & (d["ema_50"] > d["ema_200"])
            & (d["close"] > d["ema_20"])
            & higher_highs
            & higher_lows
            & (d["adx"] >= c.adx_min)
            & (d["plus_di"] > d["minus_di"])
            & d["rsi_14"].between(c.rsi_buy_min, c.rsi_buy_max)
        )
        bear = (
            (d["ema_20"] < d["ema_50"])
            & (d["ema_50"] < d["ema_200"])
            & (d["close"] < d["ema_20"])
            & lower_highs
            & lower_lows
            & (d["adx"] >= c.adx_min)
            & (d["minus_di"] > d["plus_di"])
            & d["rsi_14"].between(c.rsi_sell_min, c.rsi_sell_max)
        )

        # Fire only on the first bar where the setup becomes true.
        bull_event = bull & ~bull.shift(1, fill_value=False)
        bear_event = bear & ~bear.shift(1, fill_value=False)

        out = pd.DataFrame(index=d.index)
        out["signal"] = np.where(bull_event, 1, np.where(bear_event, -1, 0))
        has = out["signal"] != 0
        out["sl_dist"] = np.where(has, d["atr_14"] * c.sl_atr_mult, np.nan)
        out["rr"] = np.where(has, c.rr, np.nan)
        out["max_hold"] = np.where(has, c.max_hold, 0)
        out["reason"] = ""

        for i in out.index[has]:
            r = d.loc[i]
            if out.at[i, "signal"] == 1:
                out.at[i, "reason"] = (
                    f"Bullish trend: EMA20 > EMA50 > EMA200; close above EMA20; "
                    f"higher high and higher low confirmed; ADX {r['adx']:.1f} "
                    f"with +DI above -DI; RSI {r['rsi_14']:.1f} shows momentum."
                )
            else:
                out.at[i, "reason"] = (
                    f"Bearish trend: EMA20 < EMA50 < EMA200; close below EMA20; "
                    f"lower high and lower low confirmed; ADX {r['adx']:.1f} "
                    f"with -DI above +DI; RSI {r['rsi_14']:.1f} shows momentum."
                )
        return out
