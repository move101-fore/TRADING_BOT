# H7-v1 research specification — foundation draft

**Status: DRAFT — NOT FROZEN.** This document records the research contract and unresolved decisions. It does not authorize outcome analysis, validation access, live trading, or changes to H6-v1.

## 1. Research question

Among qualifying CRT parent-range sweeps and reclaims, does the presence of additional confirmation layers predict better forward returns after transaction costs and relative to matched controls?

H7 is a dose-response experiment, not a ready-to-trade strategy. A positive dose slope alone is not a profitability pass. The highest-dose group must also demonstrate positive net returns after costs and positive excess returns against controls.

## 2. Research allocation and holdout protection

The existing research registry defines:
- Development: 2022-01-01 through 2023-12-31.
- Validation: 2024-01-01 through 2024-12-31; each hypothesis version may view validation once, enforced through the validation log.
- Holdout: 2025-01-01 onward; sealed and not loaded during development or validation.

H7 must receive its own validation-log key. Validation outcome access is prohibited until the development decision is recorded and the one-view gate is passed. Outcome-blind counts may be produced separately, but they must not expose returns or outcome-derived statistics. The 2025+ holdout remains inaccessible until a finalist is explicitly authorized under the registry.

## 3. Parent range and event population

The proposed inherited CRT rules are:
- Parent: completed H4 candle.
- Minimum parent range: 1.0 × H4 ATR.
- Parent validity: following H4 candle only.
- Sweep detection: M15.
- Sweep buffer/reclaim: 0.10 × M15 ATR.
- Reject an attempt if both parent extremes have been breached by or before reclaim.
- One event per (symbol, parent candle, direction): the first qualifying sweep/reclaim.

H7 event discovery must not require liquidity confluence, support/resistance, structure, FVG, or an H6 FVG-retracement entry. Those are measured after event discovery, not used to select the base event population.

The exact meaning of the event timestamp (breach candle versus reclaim candle) and exact M15 window indexing must be frozen before implementation. The current proposed convention is to anchor the observation window to the completed reclaim candle; this remains pending approval.

## 4. Four independently measured layers

Every event receives four binary layer observations, each based only on information available by the common decision cutoff.

- **L1 — Liquidity:** parent extreme coincides with previous-day extreme or confirmed equal highs/lows, using inherited H6 tolerances.
- **L2 — Support/resistance:** swept parent extreme lies within the qualifying round-number or H4 swing-cluster zone, using inherited H6 tolerances.
- **L3 — Structure:** after reclaim, an M15 close breaks the relevant confirmed reference swing by at least 0.10 × M15 ATR.
- **L4 — FVG:** a fade-direction three-candle gap of at least 0.30 × M15 ATR forms wholly within the observation window.

Each layer must be computed independently. L4 must not be inferred from L3. Record L3-only, L4-only, both, and neither frequencies and outcomes as secondary descriptive diagnostics. Do not promote a layer based on exploratory subgroup results.

Dose = L1 + L2 + L3 + L4, with values 0 through 4. Missing or uncomputable layer evidence is not silently encoded as False; it is a data-quality failure requiring explicit accounting.

## 5. Common decision cutoff and entry

Proposed convention:
- Let s be the designated completed sweep/reclaim M15 candle.
- Let d be the completed M15 decision candle 16 bars after s (index offset +16; inclusive count is 17 candles from s through d).
- All four layer observations must be knowable no later than the close of d.
- The earliest outcome entry is the open of the next M15 candle.

The exact timestamp/index convention must be covered by boundary tests, including a layer that becomes true only after d (which must be rejected). No feature may use a pivot before its right-hand confirmation bars have completed.

## 6. Outcomes and transaction costs

Proposed horizons: 24, 48 and 96 M15 bars, measured in the direction fading the sweep.

The outcome unit is the last completed H1 ATR available at the decision cutoff. Entry occurs at the open immediately after d. Net outcome must include recorded spread, slippage of 0.02 outcome ATR per fill, and applicable swap. Cost stress tests use 1.5× and 2.0× base costs.

A gap in the event-classification window invalidates classification; a gap in the holding window invalidates only the affected outcome horizon. Do not discard an otherwise valid event solely because of a gap outside the data required for its classification or that horizon's holding window. No returns are calculated by the outcome-blind count stage.

## 7. News handling

Retain the inherited high-impact scheduled-news exclusion for currencies in the pair, with a ±30-minute window around the entry timestamp. All timestamps must be normalised to UTC before matching. If the calendar or verified time conversion is unavailable, label the affected research output PRELIMINARY; do not silently treat the filter as passed.

News-regime breakdowns are descriptive only and do not become a fifth H7 layer.

## 8. Matched controls — unresolved

The registry's frozen event-study description specifies 10 closest controls, while the current event-study implementation defaults to 5. The earlier H7 draft proposed 5. Reconcile this conflict against the registered method before H7 is frozen.

Controls must match symbol, research period, hour/session, volatility bucket and direction; be outside the exclusion radius around same-kind events; be selected without reuse; and have matching failures reported by dose. Fix the primary control count and matching window before H7 outcomes are examined. Do not silently discard events that cannot be matched.

## 9. Data and market-clock definitions — unresolved

Before event code is frozen:
1. Compare H1-derived H4 candles against actual MT5 H4 candles by symbol and date; do not assume midnight-aligned resampling matches broker H4 boundaries.
2. Define previous-day liquidity as either previous UTC calendar day or previous completed broker trading session. The recommended candidate is previous completed trading session, but the session calendar must be documented and verified.
3. Document MT5 candle timestamps, broker server offset and news timestamp convention. Internal research timestamps must be UTC-aware.
4. Freeze the H7 symbol universe from the approved H6 data-gate report; do not infer it from partial or changing downloads.

## 10. Outcome-blind sample gate — proposed, not frozen

The prior H7 draft proposed:
- All sweeps: at least 1,000 development and 500 validation events.
- Dose 3–4 combined: at least 150 development and 75 validation events.
- Each of doses 0, 1 and 2: at least 100 development events.

Before freezing, decide whether dose 0–2 minima also apply to validation, whether minima apply per horizon after data-quality exclusions, and confirm that the top bucket is dose 3–4 combined. Report raw counts, distinct symbol-month blocks, per-symbol counts, half-year counts, unmatched controls and horizon-specific valid counts. Missing a pre-registered minimum means INCONCLUSIVE; thresholds must not be relaxed after seeing counts or outcomes.

## 11. Statistical plan — proposed, not frozen

Primary questions:
1. Is excess return increasing with dose?
2. Is the dose 3–4 group profitable after costs?
3. Is the result robust across development/validation, half-years, symbols and cost stress?

The earlier H7 draft proposed a positive regression slope with 95% lower bound above zero, a dose-label permutation p-value ≤0.025 using at least 2,000 shuffles, and positive net and excess lower confidence bounds for dose 3–4 at at least two of three horizons in both development and validation. It also proposed positive slopes in at least four of six half-year periods. These criteria require a power and multiplicity review before freeze.

Bootstrap blocks must preserve calendar-time dependence and cross-symbol co-movement. Resample calendar-month blocks jointly across symbols, not independently by symbol. Permutations within symbol-month blocks must report informative blocks and available label variation. A nominal p-value is not sufficient if the permutation design has little effective variation.

All primary hypotheses, horizons, confidence intervals, cost assumptions, matching rules and pass conditions must be frozen before any H7 validation outcome is viewed. Exploratory layer/pair analyses must be clearly separated from the primary decision.

## 12. Registry and change control

H7-v1 is a new hypothesis variant and must receive its own registry row and validation-log key before outcome evaluation. Freeze means all unresolved decisions are approved and replaced with exact rules; no unresolved placeholders remain; specification and implementation commit hashes are recorded; deterministic tests pass; and the development-only run is reproducible.

Any rule change after validation outcomes are viewed requires a new hypothesis version/variant. H6-v1 remains unchanged and reproducible.

## 13. Implementation boundary

The first implementation milestone may add event/layer data contracts and synthetic unit tests only. It must not load or inspect 2024 outcomes, load 2025+ holdout data, calculate H7 returns, alter H6, or place orders. Event discovery and outcome calculation are separate later milestones after this specification is frozen.
