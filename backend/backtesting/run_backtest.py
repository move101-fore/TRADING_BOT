import argparse

import pandas as pd

from backend import config
from backend.backtesting.backtest_engine import BacktestConfig, backtest, load_point, load_swap
from backend.backtesting.performance import summarize
from backend.market_data.candle_manager import load
from backend.market_data.indicators import add_indicators
from backend.strategies.breakout import Breakout
from backend.strategies.pullback import Pullback
from backend.strategies.reversal import Reversal
from backend.strategies.trend_following import TrendFollowing

# Fixed chronological splits. The HOLDOUT is only shown with --holdout and
# should be looked at ONCE, at the very end, after all decisions are made.
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
VALID_END = pd.Timestamp("2025-01-01", tz="UTC")


def make_strategies():
    return [TrendFollowing(), Breakout(), Pullback(), Reversal()]


def swap_or_none(symbol):
    """Swap model if data/swaps.json exists; otherwise None (and the caller says so)."""
    try:
        return load_swap(symbol)
    except FileNotFoundError:
        return None


def run_all(loader=load, point_fn=load_point, symbols=None, cfg=None, swap_fn=swap_or_none):
    cfg = cfg or BacktestConfig()
    parts = []
    for sym in (symbols or list(config.SYMBOLS)):
        df = loader(sym)
        point = point_fn(sym)
        swap = swap_fn(sym) if swap_fn else None
        data = add_indicators(df)
        for strat in make_strategies():
            sig = strat.generate(data)
            parts.append(backtest(data, sig, point, cfg, atr=data["atr_14"],
                                  strategy=strat.name, symbol=sym, swap=swap))
        print("done", sym, flush=True)
    return pd.concat(parts, ignore_index=True)


def split(trades, show_holdout=False):
    t = trades["entry_time"]
    out = {
        "train": trades[t < TRAIN_END],
        "validation": trades[(t >= TRAIN_END) & (t < VALID_END)],
    }
    if show_holdout:
        out["holdout"] = trades[t >= VALID_END]
    return out


def _fmt(s):
    if s["trades"] == 0:
        return "no trades"
    return (f"{s['trades']:>5}  win {s['win_rate']*100:5.1f}% (lo {s['win_rate_lo']*100:4.1f}%)  "
            f"avgR {s['avg_r']:+.3f}  PF {s['profit_factor']:5.2f}  "
            f"netR {s['net_r']:+8.1f}  maxDD {s['max_dd_r']:6.1f}R")


def report(trades, show_holdout=False):
    periods = split(trades, show_holdout)
    names = sorted(trades["strategy"].unique())
    for strat in names:
        print(f"\n=== {strat} (all symbols pooled) ===")
        for pname, tp in periods.items():
            s = summarize(tp[tp["strategy"] == strat])
            print(f"  {pname:<11}{_fmt(s)}")
    print("\n=== per symbol, VALIDATION period: net R (trades) ===")
    val = periods["validation"]
    syms = sorted(val["symbol"].unique())
    print("  " + "strategy".ljust(18) + "".join(s.ljust(14) for s in syms))
    for strat in names:
        row = "  " + strat.ljust(18)
        for sym in syms:
            sub = val[(val["strategy"] == strat) & (val["symbol"] == sym)]
            row += f"{sub['r'].sum():+7.1f} ({len(sub):>3}) "
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="also show the final holdout period (look ONCE, at the end)")
    ap.add_argument("--slippage", type=float, default=0.02, help="slippage as a fraction of ATR")
    args = ap.parse_args()

    cfg = BacktestConfig(slippage_atr=args.slippage)
    print("Swap model:", "ON (data/swaps.json)" if swap_or_none("EURUSD") is not None
          else "OFF (data/swaps.json not found; run python -m backend.market_data.swap_info)")
    trades = run_all(cfg=cfg)
    trades.to_csv("data/trades.csv", index=False)
    print(f"Saved {len(trades)} trades to data/trades.csv")
    report(trades, args.holdout)


if __name__ == "__main__":
    main()
