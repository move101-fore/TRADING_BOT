import math

import numpy as np
import pandas as pd


def wilson_interval(wins, n, z=1.96):
    """Wilson score confidence interval for a proportion (default 95%)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _max_streak(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def equity_curves(trades, risk_pct=0.5, start_balance=10000.0):
    """Return (exit times, cumulative R, money balance) ordered by exit time."""
    t = trades.sort_values("exit_time")
    r = t["r"].to_numpy()
    cum_r = np.cumsum(r)
    balance = start_balance * np.cumprod(1.0 + (risk_pct / 100.0) * r)
    return t["exit_time"].to_numpy(), cum_r, balance


def summarize(trades, risk_pct=0.5, start_balance=10000.0):
    """Headline statistics for a table of trades (needs columns r, exit_time, outcome)."""
    n = len(trades)
    empty = {
        "trades": 0, "wins": 0, "losses": 0, "win_rate": float("nan"),
        "win_rate_lo": float("nan"), "win_rate_hi": float("nan"),
        "avg_r": float("nan"), "expectancy_r": float("nan"),
        "profit_factor": float("nan"), "net_r": 0.0, "max_dd_r": 0.0,
        "net_profit": 0.0, "max_dd_pct": 0.0,
        "max_win_streak": 0, "max_loss_streak": 0, "tp_rate": float("nan"),
    }
    if n == 0:
        return empty

    t = trades.sort_values("exit_time")
    r = t["r"].to_numpy(dtype=float)
    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())
    gross_win = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    cum = np.concatenate([[0.0], np.cumsum(r)])
    max_dd_r = float((np.maximum.accumulate(cum) - cum).max())

    bal = start_balance * np.cumprod(1.0 + (risk_pct / 100.0) * r)
    bal = np.concatenate([[start_balance], bal])
    peak = np.maximum.accumulate(bal)
    max_dd_pct = float(((peak - bal) / peak).max() * 100.0)

    lo, hi = wilson_interval(wins, n)
    return {
        "trades": n, "wins": wins, "losses": losses,
        "win_rate": wins / n, "win_rate_lo": lo, "win_rate_hi": hi,
        "avg_r": float(r.mean()), "expectancy_r": float(r.mean()),
        "profit_factor": float(pf), "net_r": float(r.sum()), "max_dd_r": max_dd_r,
        "net_profit": float(bal[-1] - start_balance), "max_dd_pct": max_dd_pct,
        "max_win_streak": _max_streak(r > 0), "max_loss_streak": _max_streak(r <= 0),
        "tp_rate": float((t["outcome"] == "tp").mean()),
    }
