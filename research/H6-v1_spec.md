# H6-v1 specification: CRT + liquidity + support/resistance + structure + FVG (+ news filter)

STATUS: DRAFT until frozen. The freeze tool refuses to freeze while any line below still carries the open-item marker.
After freezing, this file may not change. Any change creates a new version (H6-v2 = variant 3 of 12).

Registry: variant 2 of 12. Validation is viewed ONCE. The 2025-onward holdout stays sealed: the research code never loads it.
H1-v1 (support/resistance alone) is NOT tested separately; it is the location layer here.
Primary question: does the COMPLETE setup produce positive net returns after costs, beyond matched controls?
Objective reminder: a small number of high-quality opportunities with demonstrable positive expectancy, not many signals.

## 0. Approvals (owner decisions, recorded)
1. Gold: excluded if its completeness fails the data gate (section 1). 2. Outcome unit: last completed H1 ATR. 3. Parent range
at least 1.0 x H4 ATR. 4. Parent valid for the following H4 candle only; the window is never extended. 5. Liquidity pool must
coincide with another pool; this requirement may NOT be removed to raise event counts. 6. S/R half-width 0.25 x H4 ATR.
7. 96-candle cooldown per symbol and direction. 8. Minimum sample counts are minimums only and do not replace the pass criteria.
9. News filter by pre-registered event type. Pre-registered experiment: no thresholds changed after results are seen,
no conditions relaxed to manufacture trades, no outcome-based revisions.

## 1. Data, clocks, universe and the data gate
- Candidate symbols: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, XAUUSD. The universe is decided ONLY by the gate below, never by results.
- Data gate: completeness is measured against EXPECTED tradable M15 candles, not calendar time. A symbol is admitted if its missing share is at most 5%. At least 6 symbols must be admitted. Mechanics: a weekday-and-time-of-day slot belongs to a symbol's normal schedule when the symbol has a candle in at least 50% of the weeks in which any symbol traded that slot (so a daily maintenance break is not missing time); a timestamp at which NO symbol has a candle is a market-wide closure and is never counted as missing.
- Candles: M15 as downloaded (completed candles only). H1 from its own file. H4 candles are built from H1 at the broker day start (00, 04, 08, 12, 16, 20). Only COMPLETE H4 bars (4 H1 candles) are used; H4 swing zones look back at most 120 H4 candles and use only swings that pivoted before the parent candle.
- One clock throughout: the broker candle clock. Its offset from UTC must be verified (section 7) before the news filter is used.
- Periods: Development 2022-01-01 to 2023-12-31. Validation 2024-01-01 to 2024-12-31. Holdout 2025-01-01 onward (sealed).
- An event happens at the CLOSE of its last candle. The earliest trade entry is the NEXT M15 candle's open.

## 2. Volatility references (one measure per purpose, all Wilder ATR with period 14)
| Used for | ATR |
|---|---|
| Equal highs/lows tolerance (0.2) | H1 ATR |
| Liquidity confluence tolerance (0.3) | H1 ATR |
| Support/resistance clustering (0.5) and zone half-width (0.25) | H4 ATR |
| Sweep reclaim distance (0.1) and structure break buffer (0.1) | M15 ATR |
| FVG minimum size (0.3) | M15 ATR |
- H1 ATR in step 2 is the value of the last COMPLETED H1 candle at the parent's close. H4 ATR is the parent candle's own value.
- Outcome unit (moves and slippage are expressed in it): the H1 ATR of the last COMPLETED H1 candle at the event time (the H1 candle completes one hour after it opens; an M15 candle is decided when it closes).
- Control matching bucket: M15 ATR divided by the median of the previous 1000 M15 ATR values, with fixed bins <0.8, 0.8-1.0, 1.0-1.25, 1.25-1.6, >1.6.

## 3. Swings (frozen)
- A swing high (low) is a candle whose high (low) is the highest (lowest) of the 3 candles before it, itself, and the 3 candles after it.
- It is CONFIRMED when the third candle after it closes. It may not be used before then, and it is never revised or removed.
- Used on H1 (equal highs/lows), H4 (support/resistance clusters) and M15 (structure). No future candle information; no repainting.

## 4. Event sequence (strict order; a failed step ends that attempt, with no second chance for the same parent and side)
1. Parent range. A completed H4 candle P defines [low_P, high_P] and must be at least 1.0 x H4 ATR tall. It is valid for the FOLLOWING H4 candle only: from P's close until the close of the next H4 candle (16 M15 candles). The window is never extended.
2. Liquidity pool. For an upside sweep the primary pool is the parent HIGH. It counts only if at least one OTHER pool lies within 0.3 x H1 ATR of it:
   - previous-day high: the high of the previous calendar day on the candle clock (a day needs at least 12 H1 candles) , or
   - confirmed equal highs: the two most recent confirmed H1 swing highs within the last 120 H1 candles, no more than 0.2 x H1 ATR apart (level = the higher one). Both swings must have pivoted before the parent candle began, so the parent's own extreme can never be counted as a second pool.
   A previous-day level and an equal-highs level at the same price (within 0.2 x H1 ATR of each other) count as ONE pool. Downside mirror: parent LOW with previous day's low or equal lows (level = the lower one).
3. Support/resistance location. The pool level must lie within a zone: a round number, or an H4 swing cluster. Zone half-width = 0.25 x H4 ATR.
   Round numbers: every 0.0100 for non-JPY pairs, 1.00 for JPY pairs, 10.0 for gold. Swing cluster: for each confirmed H4 swing high (resistance) or low (support), the set of swings within 0.5 x H4 ATR of it; if a set has at least 2 members, its average is a zone centre. Zones must be known before the parent candle began.
4. M15 liquidity sweep. In the validity window, a is the first M15 candle whose high exceeds high_P. The sweep completes at the first candle s (at or after a, inside the window) that CLOSES at least 0.1 x M15 ATR back below high_P. If the parent's low was also breached at any candle up to and including s, the attempt is rejected as swept on both sides. Only information up to s is used.
5. M15 structure confirmation. The reference is the most recent M15 swing low CONFIRMED by the close of s. Within the next 16 M15 candles, the first candle that closes at least 0.1 x M15 ATR below it is the confirmation candle k. No reference or no such candle: no event.
6. Qualifying FVG. A bearish FVG is three consecutive M15 candles c1, c2, c3 (consecutive in time), all within [s, k], with low(c1) > high(c3) and a gap of at least 0.3 x M15 ATR (at c3). If several qualify, use the one with the latest c3. None: no event.
7. Retracement and entry confirmation. Within 16 M15 candles after k, the first candle e whose high reaches the FVG midpoint and which CLOSES below it. If a candle closes above low(c1) first, the zone is invalidated and there is no event. The event time is the close of e.
- Long setups are the exact mirror (downside sweep of the parent low; reference swing high; bullish FVG with high(c1) < low(c3); candle e whose low reaches the midpoint and closes above it).
- At most one attempt per (symbol, parent candle, direction).

## 5. Event counting (outcome-blind, mandatory before any return is calculated)
- A unique valid full-stack event has passed steps 1 to 7, then the final filters in this order: 
  (a) news filter; (b) data validity: no unexpected gap (more than 1 hour) from the start of the parent validity window through the end of the holding window, the outcome ATR available, and a complete holding window inside the research data; (c) the cooldown.
- Cooldown: per symbol and direction, an event is rejected if its entry is fewer than 96 M15 candles after the entry of the previous KEPT event. Rejected duplicates are recorded with their status and reported.
- Counts are reported by instrument, direction, half-year and overall, for Development and Validation separately, plus an attrition table showing the survivors at every stage and why attempts ended. No returns are loaded while counting.
- Minimum counts apply at EACH horizon (24, 48 and 96 candles): Development at least 300, Validation at least 150, overall at least 300. 
- If a minimum is not met the verdict is INCONCLUSIVE (not fail, not pass). Thresholds are never relaxed after counts or returns are seen.
- Order of events (owner decision): freeze H6-v1 first, then count. If the counts miss a minimum, ONE counts-only revision is permitted, recorded as H6-v1.1 with its own hash (still variant 2 of 12; it may use counts and attrition only, never returns). If H6-v1.1 also misses, the verdict is INCONCLUSIVE. The liquidity-confluence requirement (step 2) is not eligible for removal in any revision.
- Counts calculated before the news calendar exists are PRELIMINARY and are labelled as such.

## 6. News filter
- Calendar: scheduled (original) release timestamps from official release schedules only; never actual or revised times. Fields: event name, currency, scheduled local time with its IANA time zone, UTC time, source address, retrieval date.
- High-impact classification is by pre-registered event type, never by market reaction: policy-rate decisions of the Fed, ECB, BoE, BoJ, RBA, BoC, SNB and RBNZ; US Non-Farm Payrolls; US CPI.
- Time zones: local official time converted to UTC with daylight-saving rules, then to the candle clock using the verified offset.
- Currency relevance: a release applies to pairs containing its currency; gold follows USD.
- An event is excluded if its entry candle open lies within 30 minutes before to 30 minutes after a scheduled release.
- The calendar file is built before any outcome is calculated, and its hash is recorded with the frozen specification.

## 7. Clock verification (before using the calendar)
- Using only Development data, check that M15 ranges around US Non-Farm Payroll releases peak at the expected clock time (within one candle) for at least 80% of releases. If not, the offset is wrong and nothing proceeds until it is fixed.

## 8. Outcomes and the primary test (calculated only after counts are accepted and the specification is frozen)
- Move: from the next M15 open to the close of the candle h M15 candles later, h = 24, 48, 96 (6, 12, 24 hours), in outcome-ATR units, net of spread (longs: the entry candle's spread; shorts: the exit candle's), slippage of 0.02 outcome-ATR per fill (two fills) and the account's swap.
- Controls: same symbol, period, hour of day, volatility bucket and direction; not within 96 M15 candles of any event; closest in time; each used once (5 per event).
- Pass (all required): net after costs and excess over matched controls each have a 95% lower confidence bound above 0 (monthly block bootstrap); placebo p-value at most 0.025; positive results at at least two-thirds of the tested horizons (2 of 3); Development and Validation consistent (Validation viewed once); excess positive in at least four of six half-year periods; and the minimum counts of section 5.
- A sample-count pass is not a profitability pass. Verdict: PASS, FAIL or INCONCLUSIVE. Cost stress runs at 1.5x and 2x are reported.

## 9. Descriptive only (cannot pass or promote anything)
- The ladder (adding liquidity, then support/resistance, then structure, then FVG), dropping one component at a time, per-instrument and per-direction results.
- No component is promoted or retained because it looks good in this same experiment.

