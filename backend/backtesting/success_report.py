import argparse

import numpy as np
import pandas as pd

from backend.backtesting.backtest_engine import BacktestConfig, backtest
from backend.backtesting.performance import summarize, wilson_interval
from backend.backtesting.run_backtest import split


def breakeven_win_rate(r):
    """Win rate at which this set of trades would break even, from its own average win and loss in R."""
    wins = r[r > 0]
    losses = r[r <= 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    avg_win, avg_loss = wins.mean(), -losses.mean()
    return avg_loss / (avg_win + avg_loss)


def mean_ci(r, z=1.96):
    n = len(r)
    if n < 2:
        return float("nan"), float("nan")
    m = r.mean()
    se = r.std(ddof=1) / np.sqrt(n)
    return m - z * se, m + z * se


def success_table(trades, label_col="strategy"):
    """One row per (group, period): outcome mix, win rate with CI, break-even, average R with CI."""
    rows = []
    for period, tp in split(trades).items():
        for name, grp in tp.groupby(label_col):
            r = grp["r"].to_numpy(dtype=float)
            n = len(r)
            if n == 0:
                continue
            wins = int((r > 0).sum())
            wlo, whi = wilson_interval(wins, n)
            lo, hi = mean_ci(r)
            s = summarize(grp)
            rows.append({
                "group": name, "period": period, "trades": n,
                "tp_pct": (grp["outcome"] == "tp").mean() * 100,
                "sl_pct": (grp["outcome"] == "sl").mean() * 100,
                "time_pct": (grp["outcome"] == "time").mean() * 100,
                "win_pct": wins / n * 100, "win_lo": wlo * 100, "win_hi": whi * 100,
                "breakeven_pct": breakeven_win_rate(r) * 100,
                "avg_r": r.mean(), "avg_r_lo": lo, "avg_r_hi": hi,
                "profit_factor": s["profit_factor"],
            })
    return pd.DataFrame(rows)


def random_signals(df, atr, n, rng, rr=2.0, sl_mult=1.5, max_hold=48):
    """Random direction at random bars, same stop/target rules as the trend strategy."""
    size = len(df)
    sig = np.zeros(size, dtype=int)
    idx = rng.choice(np.arange(250, size - max_hold - 2), size=min(n, size - 400), replace=False)
    sig[idx] = rng.choice([1, -1], size=len(idx))
    has = sig != 0
    out = pd.DataFrame(index=df.index)
    out["signal"] = sig
    out["sl_dist"] = np.where(has, np.asarray(atr) * sl_mult, np.nan)
    out["rr"] = np.where(has, rr, np.nan)
    out["max_hold"] = np.where(has, max_hold, 0)
    out["reason"] = ""
    return out


def random_baseline(loader, point_fn, symbols, cfg=None, seeds=(1, 2, 3), n_per_symbol=3000, swap_fn=None):
    """What pure chance earns with the same stops, targets and costs."""
    from backend.market_data.indicators import add_indicators
    cfg = cfg or BacktestConfig()
    parts = []
    for sym in symbols:
        df = loader(sym)
        data = add_indicators(df)
        point = point_fn(sym)
        swap = swap_fn(sym) if swap_fn else None
        for seed in seeds:
            rng = np.random.default_rng(seed)
            sig = random_signals(data, data["atr_14"], n_per_symbol, rng)
            t = backtest(data, sig, point, cfg, atr=data["atr_14"], strategy="RANDOM_BASELINE", symbol=sym, swap=swap)
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def print_table(table):
    for period in ("train", "validation"):
        sub = table[table["period"] == period]
        if sub.empty:
            continue
        print(f"\n=== {period.upper()} ===")
        print(f"{'group':<18}{'trades':>7} {'TP%':>6}{'SL%':>6}{'time%':>7}  "
              f"{'win%':>6} {'(95% CI)':<14}{'breakeven%':>11}  {'avgR':>7} {'(95% CI)':<17}{'PF':>6}")
        for _, r in sub.sort_values("group").iterrows():
            print(f"{r['group']:<18}{int(r['trades']):>7} {r['tp_pct']:>6.1f}{r['sl_pct']:>6.1f}{r['time_pct']:>7.1f}  "
                  f"{r['win_pct']:>6.1f} ({r['win_lo']:4.1f}-{r['win_hi']:4.1f}) {r['breakeven_pct']:>10.1f}%  "
                  f"{r['avg_r']:>+7.3f} ({r['avg_r_lo']:+.3f},{r['avg_r_hi']:+.3f}) {r['profit_factor']:>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-baseline", action="store_true", help="skip the random-entry baseline")
    args = ap.parse_args()

    trades = pd.read_csv("data/trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    table = success_table(trades)

    if not args.no_baseline:
        from backend import config
        from backend.backtesting.backtest_engine import load_point
        from backend.market_data.candle_manager import load
        print("Running random-entry baseline (this takes a minute)...")
        from backend.backtesting.run_backtest import swap_or_none
        base = random_baseline(load, load_point, list(config.SYMBOLS), swap_fn=swap_or_none)
        table = pd.concat([table, success_table(base)], ignore_index=True)

    print_table(table)
    table.to_csv("data/success_report.csv", index=False)
    print("\nSaved data/success_report.csv")
    print("Reading guide: a strategy shows real promise only if its avgR confidence interval is "
          "entirely above 0 AND above the RANDOM_BASELINE row.")


if __name__ == "__main__":
    main()
