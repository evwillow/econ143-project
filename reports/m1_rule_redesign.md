# M1 — Breakout Rule Redesign

**Status:** diagnosis + proposed rewrite. **No code changed.** Awaiting user review before implementation.

**Trigger:** chart QC of 10 random sample setups from `m1_sample_setups.csv` failed 10/10 — none of the dates landed on actual Qullamaggie-style breakout days. The current rule is structurally wrong, not parametrically off.

---

## TL;DR — root cause

The current `m1_universe.py` confuses three things:

| Term | What `m1_universe.py` calls it | What Qullamaggie / breakouts.trade calls it |
|---|---|---|
| **base** | the **leg up** from swing_low → high pivot (10–42 days) | the **sideways consolidation** from high pivot → breakout day (14–60 days) |
| **setup date `t`** | a day **inside the pullback** from the high pivot, where close is 8–30% below pivot | the **breakout day** itself — close > prior high, on volume |
| **the "big move" precondition** | absent | **required** — 35–300% advance over 15+ trading days terminating at the consolidation start |

So the rule is detecting **pullback bottoms**, not breakouts. It also runs its base-quality filters (`higher_low_count`, `range_contraction_ratio`, `pct_closes_above_20ma_in_base`) on the leg-up rather than the consolidation, which is why `pct_closes_above_20ma_in_base ≥ 0.65` keeps passing meaninglessly — of course closes are above the 20MA on the way up.

The CSV proves it. For WRLD on 2016-09-30:

```
base_start_date=2016-06-13, base_end_date=2016-07-27, base_duration_days=31, date=2016-09-30
```

The "base" (per current code) ended on 2016-07-27 (the pivot). The setup date is **45 trading days later**, in a different price regime entirely. For QURE the pivot-to-setup gap is ~3 months. The chart for WRLD shows a big-volume earnings spike (the pivot), then a 2-month downtrend, then `t` lands on a small bounce in that downtrend.

Three downstream gaps compound the structural error: no volume-surge gate on `t`, no requirement that `close[t]` be above the pivot or yesterday's high, and no spacing → multiple `t`s fire inside the same pullback.

---

## Inventory: `breakouts.trade/src/data-processing/`

Only the files that participate in setup detection are listed.

| File | Role |
|---|---|
| `quality_breakouts.py` | **Main detector.** End-to-end pipeline: load bars → vectorized "is this date a breakout candidate" mask → per-candidate evaluation → score + spacing → write per-setup folder. Functions of interest below. |
| `classify_setup.py` | Post-detection: classifies each surviving breakout into HTF / VCP / Flat Base / Cup&Handle and applies `relative_strength_6m ≥ 1.0` vs SPY (126-trading-day return ratio). Setups failing RS are deleted. |
| `episodic_pivots.py` | Separate scanner for **EP gap-up** events (≥10% gap, ≥2× volume, news catalyst). Not relevant to M1's "consolidation breakout" study. Note for later: Qullamaggie EPs are a different setup family and should not be conflated. |
| `context.py` | Enrichment after detection (key levels, MA-touch counts, sector/index context). Doesn't influence pass/fail. |
| `pipeline/compute_indicators.py` | Computes 10/20/50 SMA, MACD, RSI on the daily parquet. Same indicators we already compute in polars. |

### `quality_breakouts.py` — detection flow

The control flow is `process_ticker → identify_quality_breakouts → evaluate_candidate → process_breakout`.

1. **`identify_quality_breakouts` (lines 1652–1860): vectorized first-pass mask** over every date `t` per ticker. A date is a *candidate* iff **all** of:
   - `high[t] > high[t-1]` (higher high vs prior bar)
   - `close[t] > high[t-1]` (close above prior high — the "breakout" definition)
   - `volume[t] > 2 × volume[t-1]` **AND** `volume[t] > 1.5 × avg20(volume)`
   - `(high[t]-low[t])/open[t] ≥ 2.5%` (daily range)
   - `20-day-rolling-ADR ≥ 2.5%`
   - `close[t] > 20sma × 1.01` OR `close[t] > 50sma × 1.01`

2. **`evaluate_candidate` (lines 1213–1375):** per surviving candidate, in order, must pass:
   - `check_pre_breakout_extension` (lines 1168–1210): reject if any of (a) 5-day gain into `t` > 8%, (b) close > 6% above 10SMA, (c) ≥3 consecutive "meaningful up-closes" in the prior 5 bars. Filters out continuation bars masquerading as breakouts.
   - 20-bar window green-candle ratio ≤ 0.85 (anti-spike).
   - `high[t] > prior_high × 1.005` and 10-day price trend ≥ 3%.
   - **`find_big_move` (lines 746–1061):** look back up to 90 trading days from `t`. Find a `high_date` (highest high in the window). Iterate possible `start_idx` values; the leg from `closes[start_idx]` → `high[high_date]` must satisfy:
     - total gain ∈ [35%, 300%]
     - average daily gain ≥ `0.7% × 0.8 = 0.56%`
     - duration ∈ [15, 90] days
     - quality: ≥1 of {price-increase ratio ≥ 0.4, max-single-day contribution ≤ 60%, high-volume-day ratio ≥ 0.25}
     - then snap `start_idx` to the local low in the next 20 bars for cleanliness
   - `days_from_high ∈ [14, 60]` (the **consolidation** length, in calendar days, from high pivot to breakout)
   - `check_consolidation_tightness` (lines 1101–1165): max drop from `high.high` during consolidation ≤ 30%, with up to 2 "exception days" in (30%, 35%] and zero days >35%
   - `check_orderly_pullback` (lines 1377–1429): deepest pullback ≥ 4% (must actually pull back, not just drift sideways at the high)
   - **MA "surfing"** (inline in `evaluate_candidate` lines 1307–1353): 10SMA and 20SMA both rising across the consolidation (≥98% of starting value), and ≥40% of consolidation bars touch within 2% of one of {10, 20, 50}-SMA from above
   - 30-day spacing vs other accepted setups for this ticker

3. **`process_breakout` (lines 1438–1650):** after the candidate is approved, additional integrity gates: ≥7% total gain across the 1-year `D.json` window, breakout high > 1.01× max(prior 5 highs), volume surge ≥ 1.5× avg 10/30-day volume on `t`, `close[t]` within 5% of `high[t]`, `close[t]` above 10SMA or 20SMA. Then assigns setup type via `classify_setup` and applies `passes_relative_strength` (stock_return / SPY_return ≥ 1.0 over 126 days).

4. **`score_breakouts` + `select_with_spacing` (1862–1958):** ranks surviving setups by tightness/pullback/duration/post-breakout gain and picks a maximal subset with ≥50-day spacing.

5. **Intraday gate** (`_load_early_session_bar` + `_confirm_breakout_intraday`, lines 72–163): optional, requires Polygon minute bars. **Not portable to EC143** — Polygon API key is cancelled and we have daily-only data. We must accept the precision loss.

### `classify_setup.py` — RS filter (the only non-classification logic that affects pass/fail)

`compute_relative_strength` (lines 85–144): `(stock_close[t] / stock_close[t-126]) / (SPY_close[t] / SPY_close[t-126])`. Setups with RS < 1.0 are deleted from `ds/all_setups/`. EC143's current cross-sectional momentum percentile is a related but different filter — it ranks within-day across the universe instead of measuring against an index.

---

## Inventory: `src/m1_universe.py` (current EC143 detector)

Per ticker, vectorized in polars + a per-segment numpy pass for pivot/swing-low indices.

1. **Universe filter** (`_apply_variant`, step 1): `close > $5`, `adv_20 > $5M`, ≥252-day history within segment, `mom_12_1 not null`. Segment-aware (BBBY-style ticker recycling).

2. **Security-type filter** (`_load_non_equity_set` → step 2): drops ETFs/ADRs/units/warrants. Independent of breakout logic.

3. **Cross-sectional momentum percentile** (step 3): ranks `mom_12_1` within `in_universe` per `date`.

4. **Endpoint rules** (step 4):
   - `mom_pct ≥ 0.90` (strict) / `≥ 0.80` (loose)
   - `base_duration_days ∈ [10, 42]` (where `base_duration_days = pivot_idx - swing_low_idx`, **i.e. the leg up**)
   - `pullback_pct = (pivot_close - close[t]) / pivot_close ∈ [0.08, 0.30]`
   - `ma_10 > ma_20 > ma_50` at `t`
   - strict only: `dist_52w_high_pct ≤ 15%`

5. **Pivot/swing-low computation** (`_pivot_arrays_for_segment`):
   - pivot = argmax of `closes[t-60 : t-4]` (i.e. somewhere in `[t-60, t-5]`)
   - swing_low = argmin of `closes[max(0, pivot-60) : pivot]`
   - both **close-based**, no high/low wicks

6. **Base-quality shape filters** (`_compute_base_quality_for_rows`, step 5), evaluated on the slice **`[swing_low_idx, pivot_idx]` inclusive** (i.e. on the leg-up):
   - `higher_low_count ≥ 2` via 5-day fractal
   - `range_contraction_ratio < 0.85` (avg range late half / early half)
   - `pct_closes_above_20ma_in_base ≥ 0.65`

---

## Diff: what the working tool checks that we don't

| Check | breakouts.trade | m1_universe.py | Effect |
|---|---|---|---|
| `t` is a **breakout day** (close > prior high, higher high) | ✅ | ❌ | We fire on any pullback day |
| Volume surge on `t` (≥1.5× ADV and ≥2× prior bar) | ✅ | ❌ | No momentum confirmation |
| Daily range ≥ 2.5% on `t` | ✅ | ❌ | Allows nothing-burger bars |
| Big-move precondition (35–300% over ≥15 trading days, terminating at pivot) | ✅ | ❌ | Allows downtrending or sideways stocks |
| **Consolidation** is the sequence from high pivot → `t` (length 14–60 days) | ✅ | ❌ | We treat the **leg up** as the "base" and never measure the consolidation |
| Consolidation max drop ≤ 30% (with ≤2 exception days to 35%) | ✅ | ❌ | No tightness check on the actual consolidation |
| MAs rising **and** price "surfs" MAs across consolidation | ✅ | partial — we only require `ma_10>ma_20>ma_50` at `t` | Misses choppy/wedging consolidations |
| Pre-breakout extension reject (5-day gain ≤8%, etc.) | ✅ | ❌ | Mid-run continuations slip through |
| Per-ticker spacing (30–50 days) | ✅ | ❌ | Same setup fires multiple times |
| Pivot anti-spike (rejects single-day earnings gaps as the "high") | implicit (high must be ≥3 bars from end of window; many pre-checks) | ❌ | QDEL/WRLD-style false pivots |
| Relative strength vs SPY 6m ≥ 1.0 | ✅ | partial — we use within-universe momentum percentile, which is correlated but not identical | Acceptable proxy but ranks instead of absolute |

What we check that they don't:
- Cross-sectional momentum percentile (more rigorous than absolute RS for ranking; keep this).
- 252-day history minimum + recycled-ticker segment IDs (good; keep).
- Security-type filter via yfinance cache (good; keep).
- Strict/loose split (good; keep).

---

## Diagnosis: which gap caused which sample failure

| Ticker (date) | QC complaint | Primary gap |
|---|---|---|
| KFY 2014-04-11 | down day, higher highs over period rather than consolidating, not a breakout day | (1) `t` not required to be breakout day; (2) shape filter applied to leg up not consolidation |
| WRLD 2016-09-30 | long downtrend before, pivot was earnings spike | (3) no anti-spike on pivot; (4) no big-move precondition; (5) `t` is 45 trading days after pivot — no consolidation-length cap |
| NOVT 2018-03-23 | downtrend into the date, no leg up | (4) no big-move precondition; (5) pivot-to-`t` gap unbounded |
| AMN 2018-12-10 | not the breakout day, higher points before, no clear consolidation | (1) `t` not required to be breakout; (6) no consolidation-tightness check |
| UAL 2018-12-11 | downtrend before, higher highs in range, not breakout day | (1) + (4) + (6) |
| QDEL 2020-05-19 | COVID earnings gap caused the apparent pivot | (3) no anti-spike on pivot; (1) `t` is mid-pullback, not breakout |
| JD 2021-02-26 | not a setup, not breakout day | (1) `t` not breakout; (6) no real consolidation present |
| REZI 2021-09-10 | downtrending before, no consolidation | (4) no big-move precondition; (6) no consolidation-tightness |
| INOD 2023-06-28 | expanding range not consolidating | (6) range-contraction is computed on the leg-up (where it passes trivially) instead of the consolidation |
| QURE 2025-08-27 | downtrending before; even where setup might be, not good | (5) pivot-to-`t` gap is ~3 months; (4) no big-move precondition |

Three underlying bugs:

- **Bug A (covers all 10):** `t` is interpreted as "any day in pullback from a recent high pivot," not as the breakout day. The semantics of `t`, `base`, and `pullback_pct` are inverted relative to the working definition.
- **Bug B (covers WRLD, NOVT, REZI, QURE, partly QDEL):** no required prior big-move uptrend → downtrending stocks pass.
- **Bug C (covers KFY, AMN, UAL, INOD, JD):** the shape filters (`higher_low_count`, `range_contraction_ratio`, `pct_closes_above_20ma_in_base`) are applied to the **leg up**, not the consolidation, so they cannot detect bad consolidation shape.

Bug A is the structural one. B and C are downstream consequences of the same misalignment in what "base" means.

---

## Proposed unified rule (pseudocode)

Replaces the entirety of `_pivot_arrays_for_segment`, the endpoint rules, and the base-quality stage. Universe filter, security-type filter, segmentation, and the strict/loose distinction are unchanged.

```text
For each ticker segment, compute rolling features:
    sma_10, sma_20, sma_50, adv_20, mom_12_1, high_252
    daily_range_pct[t] = (high[t] - low[t]) / open[t]
    adr_20[t] = mean(daily_range_pct[t-20..t-1])

# ---------- Stage 1: vectorized "is t a breakout day?" ----------
breakout_candidate[t] iff ALL:
    close[t]  >  high[t-1]                                # close above prior high
    high[t]   >  high[t-1]                                # higher high
    volume[t] >  1.5 * adv_20[t]                          # volume surge vs 20d avg
    daily_range_pct[t] >= 0.025                           # ≥2.5% range
    adr_20[t]          >= 0.025                           # ≥2.5% trailing ADR
    close[t] > sma_20[t]  AND  close[t] > sma_50[t]       # above key MAs

# ---------- Stage 2: prior big-move precondition ----------
For each candidate t, find a high_date:
    Search high_date as argmax(high[t-90 .. t-15])        # high pivot in [t-90, t-15]
    Reject if high[high_date] is a single-bar spike:
        gap_from_prior_close > 0.20 AND high_date not within ±2 bars of any
        other bar with high >= 0.85 * high[high_date]      # "stand-alone" spike
    Determine the leg-up:
        scan low_date over high_date - [15..60] trading days back
        for the earliest start where:
            close-to-high gain ∈ [35%, 300%]
            duration ≥ 15 trading days
            ≥40% of leg bars are up-closes
        snap low_date to local low within next 20 bars
    Reject candidate if no qualifying leg.

# ---------- Stage 3: consolidation tightness (high_date → t-1) ----------
cons_len = trading-day count between high_date and t        # NEW base_duration_days
Reject if cons_len not in [10, 42]                          # 14–60 calendar days ≈ 10–42 trading
cons_window = bars[high_date .. t-1]
max_drop = (high[high_date] - min(low[cons_window])) / high[high_date]
Reject if max_drop > 0.30 AND > 2 bars in (0.30, 0.35] AND any bar > 0.35
deepest_pullback = max_drop                                 # NEW pullback_pct
Reject if deepest_pullback < 0.04
# MA structure across consolidation
sma_10 and sma_20 at end >= 0.98 * value at start           # MAs rising
≥40% of cons bars touch within 2% of any of {sma_10,20,50} from above

# ---------- Stage 4: pre-breakout extension ----------
5-day gain into t (close[t-1]/open[t-5]) <= 0.08
close[t-1] / sma_10[t-1] - 1 <= 0.06
≤ 2 consecutive "meaningful up-closes" in [t-5..t-1]
    where meaningful = (close > open) AND (close > 1.003 * prev close)

# ---------- Stage 5: variant filter (cross-sectional, per date) ----------
mom_pct = within-universe percentile of mom_12_1 on date t
strict: mom_pct >= 0.90  AND  dist_52w_high_pct <= 0.15
loose:  mom_pct >= 0.80

# ---------- Stage 6: per-ticker spacing ----------
Within ticker, drop any candidate within 30 trading days of an already-kept setup
(rank by tightness*0.5 + pullback_quality*0.15 + days*0.1 + post-bo gain*0.25)
```

### Updated output schema

The required columns survive but **two of them change meaning**:

| Column | Old semantics | New semantics |
|---|---|---|
| `date` | day inside the pullback, 8–30% below pivot | **breakout day** (close > prior high, vol surge) |
| `base_start_date` | swing low of the leg-up | **start of consolidation** = `high_date` |
| `base_end_date` | high pivot | **t - 1** (last bar of consolidation; breakout on `t`) |
| `base_duration_days` | leg-up length (pivot − swing_low) | **consolidation length** (t − high_date) in trading days |
| `pullback_pct` | `(pivot_close − close[t]) / pivot_close` | **deepest** drop during consolidation: `(high − min_low_in_cons) / high` |
| `mom_12_1`, `dist_52w_high_pct`, `universe_variant` | unchanged | unchanged |

Additional columns to add (all derivable, useful for QC and M2/M3):
- `leg_low_date`, `leg_gain_pct` (the prior big-move's start date and magnitude)
- `breakout_volume_ratio` (volume[t] / adv_20[t])
- `breakout_range_pct` (daily range on t)
- `cons_max_drop_pct`, `cons_exception_days`
- `ma_touches_pct_in_cons`

The `higher_low_count` / `range_contraction_ratio` / `pct_closes_above_20ma_in_base` columns can stay in the parquet as supplementary shape stats — but **computed over the consolidation window**, not the leg-up — and dropped from the pass/fail rule. The MA-surf check supersedes them.

---

## Estimated impact on setup count

Current: **strict 1,115, loose 2,741**. Each gate compounds; rough order-of-magnitude estimate, applied to the current 1,115/2,741 baseline:

| Gate | Effect on candidate count |
|---|---|
| Restrict `t` to "close > prior high + higher high" | ×0.10 — most days are not breakout days |
| Volume surge ≥1.5× ADV + range ≥2.5% on `t` | ×0.5 |
| Big-move precondition (35–300% over 15+d) | ×0.5 |
| Consolidation tightness (max drop ≤30% + ≥4% pullback + MA-surf ≥40%) | ×0.5 |
| Pre-breakout extension reject | ×0.85 |
| Per-ticker 30-day spacing | ×0.5 (within-ticker dedup) |

Compound multiplier ≈ **0.005**, so a defensible point estimate is:
- **strict: ~50–100 setups**
- **loose: ~150–300 setups**

That's a 90–95% cut. For a study targeting *quality* setups this is the right direction; the current 1,115/2,741 are overwhelmingly junk per QC. If the rule is too strict for the empirical study (need more sample for cross-sectional regressions in M3), we can relax volume threshold (1.5→1.2), pullback minimum (4%→2%), and big-move minimum (35%→20%) to roughly double the count without re-introducing the structural failures.

---

## Open questions for user judgment

These are points where breakouts.trade and the EC143 spec conflict, or where I need a decision before implementation.

1. **Confirm `t` = breakout day.** The QC notes treat it that way and the working tool defines it that way, but the original EC143 §M1 spec language ("breakout candidate") was ambiguous and the current code went with "in-pullback." Are we now committing to `t` = breakout day across M1/M2/M3? (Affects M2 outcome windows: post-breakout returns are measured forward from `t`.)

2. **Schema breaking change.** `base_start_date`, `base_end_date`, `base_duration_days`, `pullback_pct` keep the same names but flip meaning. Downstream M2/M3 will need re-running. Are you OK with the rename-in-place, or should we add new columns (`cons_start_date`, `cons_end_date`, `cons_duration_days`, `cons_max_drop_pct`) and deprecate the old ones?

3. **Relative strength.** breakouts.trade requires `stock_6m_return / SPY_6m_return ≥ 1.0`. EC143 currently uses cross-sectional momentum percentile within universe (no SPY needed). The two are correlated but not identical:
   - **Option A (keep current):** use `mom_pct` thresholds 0.90/0.80. No new data dependency.
   - **Option B (lift breakouts.trade):** add SPY/^GSPC to the daily bars, compute 6m RS, gate at `RS ≥ 1.0`, drop the percentile filter or keep both. Marginal extra work since SPY is already free via yfinance.
   - I'd recommend **A + B**: keep the percentile (it's a good cross-sectional ranker) and add RS as an additional gate. Decide.

4. **Per-ticker spacing.** breakouts.trade uses 30–50 days. With M2 looking at forward returns, redundant nearby setups bias outcome statistics by autocorrelation. Recommend 30 trading days. OK?

5. **Daily-only loses intraday confirmation.** breakouts.trade gates breakouts on the first 30 minutes (volume + range scaled). We can't replicate. Acceptable cost?

6. **Anti-spike pivot rule needs sharpening.** The clearest failures (WRLD, QDEL) had earnings spikes as the pivot. My pseudocode rejects "stand-alone" spikes (gap >20% AND no neighboring bar within 85% of high). Is that the right test? Alternatives: (a) require pivot bar's close-to-high pct < 5%, (b) require ≥2 bars within 5% of pivot high in surrounding ±5 bars, (c) reject any pivot bar whose gap from prior close exceeds 15%. I lean (b) because it tests "did the price level get re-tested" which is what makes a pivot meaningful. Pick one.

7. **EP (Episodic Pivot) family.** Some "missed" patterns may actually be EPs (gap-up + catalyst), which breakouts.trade handles in a separate scanner. EC143 §M1 spec doesn't mention EPs. Decide: scope-limit M1 to consolidation breakouts only (cleaner), or add a parallel EP detector? Recommend **scope-limit M1**, treat EPs as future work — the QC was on consolidation breakouts.

8. **`min_uptrend_days` units.** breakouts.trade uses 30 *calendar* days (≈21 trading) but the comment says "15 trading days minimum." My pseudocode used 15 trading days. Confirm trading vs calendar.

---

## Stop point

This document is the deliverable. No code changed. Awaiting user decisions on Q1–Q8 before rewriting `_pivot_arrays_for_segment`, the endpoint-rules block, and the base-quality stage in `m1_universe.py`.
