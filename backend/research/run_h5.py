import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend import config
from backend.research.event_study import (PERIODS, SymbolData, add_placebo_pvalues, placebo_calibration,
                                          study)
from backend.research.structure import STRUCTURE_VERSION, structure_events
from backend.research.verdict import verdict

LOG = Path("research") / "validation_log.csv"
OUT_DIR = Path("research")


def validation_already_viewed(hyp_id, log=LOG):
    if not Path(log).exists():
        return False
    with open(log, newline="") as f:
        return any(row and row[0] == hyp_id for row in csv.reader(f))


def record_validation(hyp_id, log=LOG):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", newline="") as f:
        csv.writer(f).writerow([hyp_id, datetime.now(timezone.utc).isoformat()])


def run_study(frames, points, swaps, periods, n_boot=2000, cost_mult=1.0, placebo_runs=100):
    """frames/points/swaps are dicts keyed by symbol. Returns (events, results, verdicts)."""
    symbol_data, parts = {}, []
    for sym, df in frames.items():
        sd = SymbolData(df, points[sym], swaps.get(sym), cost_mult=cost_mult)
        symbol_data[sym] = sd
        ev = structure_events(sd.df)
        ev.insert(0, "symbol", sym)
        parts.append(ev)
    events = pd.concat(parts, ignore_index=True)
    res = study(events, symbol_data, periods=periods, n_boot=n_boot)
    if placebo_runs:
        res = add_placebo_pvalues(res, placebo_calibration(events, symbol_data, periods, runs=placebo_runs))
    return events, res, verdict(res)


def print_results(res):
    pd.set_option("display.width", 220)
    has_p = "placebo_p" in res.columns
    for (kind, period), grp in res.groupby(["kind", "period"]):
        print(f"\n=== {kind}  |  {period} ===")
        print(f"{'horizon':>8}{'events':>8}{'no-ctrl':>8}{'invalid':>8}   {'net after costs (ATR)':<30}{'excess vs controls (ATR)':<30}"
              f"{'halves+':>8}{'placebo p':>11}")
        for _, r in grp.iterrows():
            p = f"{r['placebo_p']:.3f}" if has_p and pd.notna(r["placebo_p"]) else "  n/a"
            print(f"{int(r['horizon']):>7}h{int(r['events']):>8}{int(r['unmatched']):>8}{int(r['invalid']):>8}   "
                  f"{r['net_mean']:+.3f} [{r['net_lo']:+.3f},{r['net_hi']:+.3f}]      "
                  f"{r['excess_mean']:+.3f} [{r['excess_lo']:+.3f},{r['excess_hi']:+.3f}]      "
                  f"{int(r['halves_positive'])}/{int(r['halves'])}{p:>11}")


def main():
    from backend.backtesting.backtest_engine import load_point
    from backend.backtesting.run_backtest import swap_or_none
    from backend.market_data import history

    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true",
                    help="ALSO evaluate the 2024 validation period (allowed once per hypothesis version)")
    ap.add_argument("--force", action="store_true", help="re-view validation (this burns it; the registry must say so)")
    ap.add_argument("--placebo-runs", type=int, default=100, help="placebo repetitions (more = finer p-values)")
    ap.add_argument("--cost-mult", type=float, default=1.0, help="multiply spread and slippage (1.5 and 2.0 are the stress tests)")
    args = ap.parse_args()

    periods = {"development": PERIODS["development"]}
    if args.validation:
        if validation_already_viewed(STRUCTURE_VERSION) and not args.force:
            raise SystemExit(f"Validation for {STRUCTURE_VERSION} was already viewed (see {LOG}). "
                             "Looking again would contaminate it. Define a new version instead.")
        periods["validation"] = PERIODS["validation"]

    frames = {s: history.load(s, "H1") for s in config.SYMBOLS}      # refuses incomplete history
    points = {s: load_point(s) for s in config.SYMBOLS}
    swaps = {s: swap_or_none(s) for s in config.SYMBOLS}
    print("Swap model:", "ON" if any(swaps.values()) else "OFF (data/swaps.json missing)", "| cost multiplier:", args.cost_mult)

    events, res, verdicts = run_study(frames, points, swaps, periods, cost_mult=args.cost_mult,
                                      placebo_runs=args.placebo_runs)
    print(f"Structure engine {STRUCTURE_VERSION}: {len(events)} events "
          f"({', '.join(f'{k}={v}' for k, v in events['kind'].value_counts().items())})")
    print_results(res)
    print("\n=== VERDICT (full registry rule) ===")
    for _, v in verdicts.iterrows():
        print(f"{v['kind']:<8} {v['status']:<34} {v['reasons']}")

    OUT_DIR.mkdir(exist_ok=True)
    tag = f"x{args.cost_mult:g}"
    name = f"h5_dev_validation_{tag}" if args.validation else f"h5_development_{tag}"
    res.to_csv(OUT_DIR / f"{name}.csv", index=False)
    if args.validation:
        record_validation(STRUCTURE_VERSION)
        print(f"\nValidation viewed and logged in {LOG}.")


if __name__ == "__main__":
    main()
