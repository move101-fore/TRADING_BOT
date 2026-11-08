from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.strategies.base_strategy import BaseStrategy


@dataclass
class ReversalConfig:
    swing_right: int = 3               # MUST equal `right` used in indicators.swings()
    min_sep: int = 8                   # min bars between the pattern's swing points
    max_sep: int = 80                  # max bars between the pattern's swing points
    window: int = 20                   # bars allowed between pattern completion and the break
    tol_atr: float = 0.5               # double top/bottom: peaks within this x ATR
    shoulder_tol_atr: float = 1.0      # H&S: shoulders within this x ATR
    head_min_atr: float = 0.5          # H&S: head must exceed shoulders by this x ATR
    min_pattern_height_atr: float = 1.5  # peak-to-neckline height, x ATR
    div_min_rsi: float = 3.0           # RSI must be lower at 2nd high by at least this
    div_rsi_level: float = 60.0        # first high must have RSI at least this
    break_buffer_atr: float = 0.10     # close must break the level by this x ATR
    sl_buffer_atr: float = 0.25
    min_sl_atr: float = 0.50
    max_sl_atr: float = 4.00
    rr: float = 2.0
    max_hold: int = 48


class Reversal(BaseStrategy):
    name = "reversal"

    def __init__(self, config=None):
        self.cfg = config or ReversalConfig()

    # ------------------------------------------------------------------
    def _signals(self, d):
        c = self.cfg
        n = len(d)
        hi = d["high"].to_numpy(); lo = d["low"].to_numpy(); cl = d["close"].to_numpy()
        rsi = d["rsi_14"].to_numpy(); atr = d["atr_14"].to_numpy()
        sh = d["swing_high"].to_numpy(); sw_lo = d["swing_low"].to_numpy()

        # Bearish patterns on the real series -> SELL signals.
        bear = self._detect(hi, lo, cl, rsi, atr, sh, sw_lo)
        # Bullish patterns = bearish patterns on the mirrored series -> BUY signals.
        bull = self._detect(-lo, -hi, -cl, 100.0 - rsi, atr, -sw_lo, -sh)

        signal = np.zeros(n, dtype=int)
        sl_dist = np.full(n, np.nan)
        reasons = [""] * n
        for i in range(n):
            b, u = bear[i], bull[i]
            if b is not None and u is not None:
                continue                       # conflicting on the same bar: skip
            if b is not None:
                signal[i] = -1
                sl_dist[i] = b["sl"]
                reasons[i] = self._reason(-1, b, rsi_note=True)
            elif u is not None:
                signal[i] = 1
                sl_dist[i] = u["sl"]
                u = dict(u); u["level"] = -u["level"]
                reasons[i] = self._reason(1, u, rsi_note=True)

        out = pd.DataFrame(index=d.index)
        out["signal"] = signal
        has = out["signal"] != 0
        out["sl_dist"] = np.where(has, sl_dist, np.nan)
        out["rr"] = np.where(has, c.rr, np.nan)
        out["max_hold"] = np.where(has, c.max_hold, 0)
        out["reason"] = reasons
        return out

    # ------------------------------------------------------------------
    def _detect(self, hi, lo, cl, rsi, atr, sh, sw_lo):
        """
        Detect BEARISH reversals. Returns a list (len n) of None or a dict.
        Swing pivots are added only on the bar where they become confirmed
        (pivot index + swing_right), so nothing here uses future candles.
        """
        c = self.cfg
        R = c.swing_right
        n = len(cl)
        found = [None] * n
        highs = []   # (index, price, rsi)
        lows = []    # (index, price)
        used = set()

        for i in range(n):
            if not np.isnan(sh[i]):
                highs.append((i - R, sh[i], rsi[i - R]))
            if not np.isnan(sw_lo[i]):
                lows.append((i - R, sw_lo[i]))
            a = atr[i]
            if np.isnan(a):
                continue
            buf = c.break_buffer_atr * a

            for matcher in (self._head_shoulders, self._double_top, self._divergence):
                m = matcher(i, a, buf, hi, lo, cl, highs, lows, used)
                if m is None:
                    continue
                used.add(m["key"])
                stop = m["stop_ref"] + c.sl_buffer_atr * a - cl[i]
                if c.min_sl_atr * a <= stop <= c.max_sl_atr * a:
                    m["sl"] = stop
                    found[i] = m
                break
        return found

    # ------------------------------------------------------------------
    def _head_shoulders(self, i, a, buf, hi, lo, cl, highs, lows, used):
        c, R = self.cfg, self.cfg.swing_right
        if len(highs) < 3:
            return None
        (ia, pa, _), (ib, pb, _), (ic, pc, _) = highs[-3:]
        key = ("hs", ic)
        if key in used or i - (ic + R) > c.window:
            return None
        if not (c.min_sep <= ib - ia <= c.max_sep and c.min_sep <= ic - ib <= c.max_sep):
            return None
        if pb < max(pa, pc) + c.head_min_atr * a or abs(pa - pc) > c.shoulder_tol_atr * a:
            return None
        t1 = ia + int(np.argmin(lo[ia:ib + 1])); t1v = lo[t1]
        t2 = ib + int(np.argmin(lo[ib:ic + 1])); t2v = lo[t2]
        if t2 <= t1:
            return None
        slope = (t2v - t1v) / (t2 - t1)
        neck = lambda k: t1v + slope * (k - t1)
        if pb - neck(ib) < c.min_pattern_height_atr * a:
            return None
        ks = np.arange(ic + 1, i)
        if len(ks) and np.any(cl[ks] < t1v + slope * (ks - t1) - buf):
            return None                           # neckline already broken earlier
        if cl[i] < neck(i) - buf:
            return {"key": key, "pattern": "head_shoulders", "level": neck(i), "stop_ref": pc}
        return None

    def _double_top(self, i, a, buf, hi, lo, cl, highs, lows, used):
        c, R = self.cfg, self.cfg.swing_right
        if len(highs) < 2:
            return None
        (ia, pa, _), (ib, pb, _) = highs[-2:]
        key = ("dt", ib)
        if key in used or i - (ib + R) > c.window:
            return None
        if not (c.min_sep <= ib - ia <= c.max_sep) or abs(pa - pb) > c.tol_atr * a:
            return None
        neck = lo[ia:ib + 1].min()
        if min(pa, pb) - neck < c.min_pattern_height_atr * a:
            return None
        if i - 1 > ib and np.any(cl[ib + 1:i] < neck - buf):
            return None
        if cl[i] < neck - buf:
            return {"key": key, "pattern": "double_top", "level": neck, "stop_ref": max(pa, pb)}
        return None

    def _divergence(self, i, a, buf, hi, lo, cl, highs, lows, used):
        c, R = self.cfg, self.cfg.swing_right
        if len(highs) < 2:
            return None
        (ia, pa, ra), (ib, pb, rb) = highs[-2:]
        key = ("div", ib)
        if key in used or i - (ib + R) > c.window:
            return None
        if not (c.min_sep <= ib - ia <= c.max_sep) or not pb > pa:
            return None
        if np.isnan(ra) or np.isnan(rb) or rb > ra - c.div_min_rsi or ra < c.div_rsi_level:
            return None
        between = [p for (li, p) in lows if ia <= li <= ib]
        if not between:
            return None
        level = min(between)                       # structure low that must break
        if i - 1 > ib and np.any(cl[ib + 1:i] < level - buf):
            return None
        if cl[i] < level - buf:
            return {"key": key, "pattern": "rsi_divergence", "level": level,
                    "stop_ref": pb, "rsi_a": ra, "rsi_b": rb}
        return None

    # ------------------------------------------------------------------
    def _reason(self, direction, m, rsi_note=True):
        bull = direction == 1
        p = m["pattern"]
        lvl = m["level"]
        if p == "double_top":
            name = "Double bottom" if bull else "Double top"
            text = (f"{name}: two confirmed swing {'lows' if bull else 'highs'} formed at nearly the "
                    f"same price, with a pullback of at least 1.5 x ATR between them. A candle then "
                    f"closed {'above' if bull else 'below'} the neckline at {lvl:.5f}, confirming a break of structure.")
        elif p == "head_shoulders":
            name = "Inverse head and shoulders" if bull else "Head and shoulders"
            text = (f"{name}: three confirmed swing {'lows' if bull else 'highs'} with the middle one "
                    f"the most extreme and similar shoulders. A candle closed "
                    f"{'above' if bull else 'below'} the sloping neckline ({lvl:.5f}), confirming the pattern.")
        else:
            ra, rb = m["rsi_a"], m["rsi_b"]
            if bull:
                ra, rb = 100.0 - ra, 100.0 - rb
            text = (f"{'Bullish' if bull else 'Bearish'} RSI divergence: price made a "
                    f"{'lower low' if bull else 'higher high'} while RSI {'strengthened' if bull else 'weakened'} from {ra:.1f} to {rb:.1f}. "
                    f"A candle then closed {'above' if bull else 'below'} the last structure "
                    f"{'high' if bull else 'low'} at {lvl:.5f} (change of character).")
        return text
