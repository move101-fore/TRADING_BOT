# Research registry (frozen rules)

## 1. Data
- Usable history: 2022-01-01 onward (H1, M15, M1 about 95-98% complete). Before 2022 the history is sparse and EXCLUDED.
- Tick history is not available (EURUSD 2026 only). Backtests use M1 candles with the recorded spread; ask = bid + spread.
- If a stop and a target fall inside the same M1 candle, the stop is assumed to hit first.

## 2. Allocation (frozen, approved)
- Development: 2022-01-01 to 2023-12-31 (design, event studies)
- Validation:  2024-01-01 to 2024-12-31 (looked at ONCE per hypothesis version; enforced by research/validation_log.csv)
- Holdout:     2025-01-01 to latest (SEALED; research code never loads it; opened once, at the very end, for finalists only)
- Changing any rule after seeing validation results creates a NEW version (for example H5-v2) and counts as a new variant.

## 3. Costs (always on)
- Spread from each candle, slippage 0.02 ATR per fill (stress at 1.5x and 2x), swap per symbol from data/swaps.json
  (rollover 22:00 candle clock, Wednesday triple, Mon-Fri), commission 0 (Standard account, to be verified).

## 4. Baselines every candidate must beat
- B0: the old four strategies (git tag baseline-four-strategies, data/baseline/).
- B1: random direction at the same timestamps.
- B2: matched random levels (same count, same ATR distance structure).
- B3: simple prior-day high/low reaction model.

## 5. Trial ledger (cap: 12 variants in total)
| ID | Date | Hypothesis | Frozen definition | Result | Status |
|----|------|------------|-------------------|--------|--------|
| H5-v1 | (fill in) | Market structure: with-trend BOS beats matched controls; CHoCH reversal tested separately | see section 7 | FAIL in development: BOS excess about 0 (12h +0.01 [-0.14,+0.16]); CHoCH net and excess negative; no kind passed at 2 of 3 horizons; validation NOT opened | closed |

## 6. Event-study method (frozen)
- Move measured from the NEXT candle's open to the close 6, 12 or 24 H1 bars later, in ATR units, net of spread, 2 x 0.02 ATR slippage and swap.
- Controls: same symbol, period, hour of day, volatility bucket (ATR / 250-bar median: <0.8, 0.8-1, 1-1.25, 1.25-1.6, >1.6) and direction;
  not within 24 bars of an event of the same kind; the 10 closest in time (within about 15 trading days, widened if needed).
- Confidence intervals: 95%, resampling whole calendar months (2,000 draws).
- Events with a data gap in the holding window are excluded and counted.

## 7. H5-v1 definition (frozen)
- Swings: confirmed pivots, 3 bars each side. Trend UP = last two swing highs and lows rising; DOWN = both falling.
- BOS: close beyond the last swing in the trend direction by at least 0.1 ATR. CHoCH: same for the opposite swing. Each swing level fires once.

## 8. Pass rule for an event study (all required)
- Net mean after costs: lower 95% bound above 0.   AND   Excess over matched controls: lower 95% bound above 0.
- At least 300 events, and the same kind must pass at 2 of the 3 horizons (guards against picking the best of many cells).
- Positive in Development AND, when viewed once, Validation; positive excess in at least 4 of 6 half-year slices overall.
- Only events that pass may be given exits. Each exit rule is a variant in the ledger.


## 9. Symbol exclusions
- GBPUSD excluded from H5 and H1 event studies: H1 history is 8.42% incomplete (every month of 2022 patchy). All other symbols are under 3% missing.

- H5-v1 FINAL (R2.1 hardened re-run, 7 symbols, development only): BOS excess +0.007/-0.007/-0.028 ATR (placebo p 0.43-0.54); CHoCH excess negative (placebo p 0.81-0.92); verdict FAIL for both; validation NOT opened; 1 of 12 variants used.
