# H7-v1 specification: do the layers add predictive value? (dose-response design)

STATUS: DRAFT until frozen. The freeze tool refuses to freeze while any line below still carries the open-item marker.
Registry: variant 3 of 12 (H5-v1 = 1, H6-v1 = 2, closed INCONCLUSIVE). Validation viewed ONCE. The 2025-onward holdout stays sealed.
Why H7 exists: H6-v1 required every layer at once and produced 3 events in 2 years, which no pre-registered test can judge.
H7 asks the same question in a way the data can answer: among MANY sweeps, does excess return rise with the number of layers present?
Objective reminder: a small number of high-quality opportunities with demonstrable positive expectancy. A statistically
significant dose-response is NOT a profitability pass. The top bucket must itself be profitable after costs.

## 1. Reused definitions (frozen from H6-v1, unchanged)
Data gate, universe rule, candle clock, periods, ATR references, swing definition, parent range (H4 candle at least 1.0 x H4 ATR,
valid for the following H4 candle only), liquidity pool tolerances, support/resistance zones, sweep + reclaim (not swept on both
sides), structure reference and break buffer, FVG size and consecutive-candle rule, news filter, cost model and swap. All as in
research/H6-v1_spec.md. Where this file says nothing, H6-v1 governs. Nothing in H6-v1 is re-tuned.

## 2. The event set: ALL qualifying sweeps
- One event per (symbol, parent candle, direction): the first sweep + reclaim (H6 step 4) of a qualifying parent. No requirement for liquidity confluence or a support/resistance zone. [CONFIRM]
- Trade direction is the fade of the sweep: short after an upside sweep, long after a downside sweep.

## 3. The common decision candle (no look-ahead)
- Every layer must be knowable before the outcome is measured, so ALL events share one rule: the decision candle d is the M15 candle 16 candles after the sweep candle s (the end of the H6 confirmation window). Layers are evaluated using candles up to and including d only. [CONFIRM]
- The outcome is measured from the open of the candle after d, exactly as in H6 section 8 (horizons 24, 48, 96 M15 candles, net of spread, slippage 0.02 outcome-ATR per fill, swap).
- If d, or the holding window after it, contains an unexpected data gap, the event is invalid and counted as such.

## 4. The layers (independent yes/no flags, all known by d)
- L1 liquidity confluence: the swept parent extreme coincides with a previous-day extreme or confirmed equal highs/lows (H6 step 2 rules).
- L2 support/resistance location: the swept parent extreme lies within a round-number or H4 swing-cluster zone (H6 step 3 rules).
- L3 structure confirmation: a candle closes at least 0.1 x M15 ATR beyond the reference swing confirmed by s, at any candle from s+1 through d (H6 step 5, same window length).
- L4 imbalance: a qualifying FVG in the fade direction (gap at least 0.3 x M15 ATR, three consecutive candles) formed entirely within [s, d]. It does NOT require L3. [CONFIRM]
- Dose = L1 + L2 + L3 + L4, an integer from 0 to 4. The retest/entry confirmation of H6 step 7 is NOT a layer here.
- News: events whose entry candle (the candle after d) lies within +-30 minutes of a scheduled high-impact release are excluded, as in H6. Until the calendar exists, counts are PRELIMINARY.

## 5. Outcome-blind counting (mandatory before any return is calculated)
- Report the attrition and the number of valid events by dose, symbol, direction, half-year and horizon, for Development and Validation separately. No returns are loaded.
- Minimums (a sample-count pass is not a profitability pass): [CONFIRM]
  - all sweeps: Development at least 1,000 and Validation at least 500;
  - the top bucket (dose of at least 3): Development at least 150 and Validation at least 75, at each horizon;
  - every dose from 0 to 2 must have at least 100 events in Development.
- A miss means INCONCLUSIVE. Thresholds are never relaxed after counts or returns are seen.
- Order (owner rule): freeze, then count. If a minimum is missed, ONE counts-only revision is permitted (H7-v1.1, outcome-blind, own hash). If that also misses, INCONCLUSIVE. Removing a layer is not an allowed revision.

## 6. The primary test (calculated only after counts are accepted and the specification is frozen)
Excess return of each event over matched controls: same symbol, period, hour of day at d, volatility bucket, and direction; not within 96 candles of any event; closest in time; each control used once (5 per event).
PASS needs ALL of the following at at least 2 of the 3 horizons, in Development and then in Validation (viewed once):
1. Slope: the regression slope of excess on dose has a 95% lower bound above 0 (monthly block bootstrap).
2. Permutation: the slope's p-value is at most 0.025 when dose labels are shuffled within symbol-month blocks (at least 2,000 shuffles).
3. Top bucket (dose of at least 3): net return after costs AND excess over controls each have a 95% lower bound above 0, with the minimum counts of section 5.
4. Slope positive in at least four of the six half-year periods.
Verdict: PASS, FAIL or INCONCLUSIVE. Cost stress at 1.5x and 2x is reported.

## 7. Descriptive only (cannot pass or promote anything)
Mean excess by dose, each layer alone, each pair of layers, per-symbol and per-direction results, results by decision anchor at k instead of d, and dropping one layer at a time. No layer is promoted or retained because it looks good here.

## 8. Change control
Frozen via the freeze tool (hash and git commit recorded). Any change after freezing is a new version and a new variant.

## H7 CONTROL MATCHING - FROZEN DECISION

Select up to five eligible controls per event, requiring
at least three valid controls.

Candidate controls must be within +/- 1,440 M15 bar indices
of the event.

The matching window is fixed. It must never be widened
to obtain additional controls.

Controls must match:
- Symbol
- Research period
- Hour of day
- Volatility bucket
- Direction

Controls must be at least 96 M15 bars away from every
qualifying H7 sweep event.

Each control may be used only once.

Events with fewer than three valid controls are excluded
from matched-excess calculations but remain in attrition
reporting.

Each event receives equal weight. Its control benchmark
is the mean return of its eligible controls.

Report the distribution of events with 3, 4, 5 and fewer
than 3 controls.

Run a sensitivity analysis restricted to events with
exactly five valid controls.

