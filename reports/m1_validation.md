# M1 — Universe & Breakout Detection Validation (rewrite)

Detector rewritten 2026-05-03 per `reports/m1_rule_redesign.md`. Old rule fired on pullback days inside a not-really-a-base; new rule fires on the breakout day itself after a real big-move + tight consolidation. **Schema changed** (see column list at bottom). Headline test runs on the **loose** universe; strict is a §7.1 robustness check. **Detector ships at Round 2** — see iteration history below.

## QC iteration history

Manual QC of 10 random sample setups was performed three times during M1 development. Each round adjusted the detector based on specific chart-level failures. Documented honestly here so the writeup §8 limitations section can reference the empirical search:

| Round | Date | Detector | QC score | strict / loose totals |
|---|---|---|---|---|
| 1 (broken) | 2026-05-03 | original `m1_universe.py`: `base = leg-up`, `t = day in pullback` | **0/10** (every sample missed the breakout day) | 1,115 / 2,741 |
| 2 (ships) | 2026-05-04 | full rewrite: 6-stage pipeline (see redesign doc); ADR layer; 15%-gap-open pivot reject; 60-day belt-and-suspenders cap; hardcoded ADR fallback for yfinance 404s | **6.5/10** (6 yes, 1 sort-of, 3 no) | **2,214 / 3,075** ← canonical |
| 3 (rolled back) | 2026-05-04 | added cons_low_trend_slope reject (<−0.002), pre_legup_return reject (<−0.20), and tightened cons_min 10→15 trading days | **2.5/10** | 1,735 / 2,371 (rejected) |

Round 3's rejects were over-fitted to the three failed Round 2 samples (PPTA, QTWO, ATYR) and hurt non-flagged setups — the regression to 2.5/10 came from cutting valid setups, not from passing more bad ones. Both Round 3 metrics survive in this parquet as **informational columns** (`cons_low_trend_slope`, `pre_legup_return`) — useful for M2 / M3 feature exploration without affecting M1's setup count.


- Feature table: 38,810,782 rows (30,514 tickers)
- Tickers split into >1 segment by the 60d recycling gap: **9,660** (31.66%)
- **Canonical pass: `default`** (used for `setups.parquet` and the QC sample).
- Canonical strict total: **2,214** (was 1,115 under old rule)
- Canonical loose total:  **3,075** (was 2,741 under old rule)

## Security-type filter (Bug 1 fix, 2026-05-04)

- Source: **yfinance (3,901 tickers, 1,756 non-equity: 1,483 non-EQUITY + 263 non-US + 0 name-match ADR + 10 hardcoded-ADR fallback)**

| Detection rule | Tickers flagged |
|---|---:|
| `quote_type != EQUITY` (existing rule) | 1,483 |
| `country` non-null and != 'United States' (NEW) | 263 |
| `long_name`/`short_name` matches `(?i)\bADR\b\|American Depositary\|Sponsored ADR` (NEW) | 0 |
| Hardcoded ADR fallback (yfinance 404 rescue, e.g. ERJ) | 10 |
| **Union (final non-equity set)** | **1,756** |
| _added by ADR layer vs old EQUITY-only filter_ | _273_ |

Manual QC of the 2026-05-03 sample flagged ERJ (Embraer ADR) and a suspected EMES ETF as having slipped through the old `quote_type != EQUITY`-only filter. yfinance returns `quoteType=EQUITY` for ADRs, so two additional layers were added: country origin and long-name regex match.

## Old rule vs new rule (totals)

| Variant | Old (broken) rule | New rule (default params) | New rule (relaxed params) |
|---|---:|---:|---:|
| strict | 1,115 | 2,214 | (not triggered: loose>=200) |
| loose  | 2,741  | 3,075  | (not triggered: loose>=200) |

## Pipeline dropouts (canonical pass)

| Stage | Rows surviving |
|---|---:|
| Stage 0a — universe (close/ADV/history) pre-security | 11,133,797 |
| Stage 0b — security-type filter (drop ETFs/ADRs/...) | 8,974,061 |
| Stage 1  — vectorized breakout-day mask              | 89,449 |
| Stages 2-4 — pivot+leg, consolidation, pre-extension | 9,073 |
| Stage 5 (strict)  — variant filter                   | 3,240 |
| Stage 6 (strict)  — per-ticker 30-day spacing        | 2,214 |
| Stage 5 (loose)   — variant filter                   | 4,521 |
| Stage 6 (loose)   — per-ticker 30-day spacing        | 3,075 |

## Setups by year x variant (canonical)

| Year | strict | loose |
|---:|---:|---:|
| 2010 | 89 | 116 |
| 2011 | 94 | 122 |
| 2012 | 70 | 96 |
| 2013 | 112 | 138 |
| 2014 | 87 | 101 |
| 2015 | 74 | 94 |
| 2016 | 64 | 84 |
| 2017 | 136 | 174 |
| 2018 | 124 | 149 |
| 2019 | 130 | 155 |
| 2020 | 204 | 327 |
| 2021 | 237 | 414 |
| 2022 | 106 | 140 |
| 2023 | 166 | 220 |
| 2024 | 246 | 362 |
| 2025 | 275 | 383 |

## Top 10 most-frequent setup tickers per variant

### strict

| Ticker | Setups |
|---|---:|
| TWLO | 8 |
| BOOT | 8 |
| CROX | 8 |
| CONN | 8 |
| LNG | 8 |
| MU | 8 |
| RDNT | 8 |
| NTRA | 8 |
| PTCT | 8 |
| LSCC | 8 |

### loose

| Ticker | Setups |
|---|---:|
| LSCC | 12 |
| BOOT | 10 |
| RDNT | 10 |
| TWLO | 9 |
| PTCT | 9 |
| NTRA | 9 |
| HUBS | 9 |
| CROX | 9 |
| INSP | 9 |
| FSLR | 9 |

## Parameters used (canonical pass)

```
anti_spike_min_neighbors = 2
anti_spike_neighbor_pct = 0.95
anti_spike_window = 5
cons_exception_drop_pct = 0.35
cons_max_drop_pct = 0.3
cons_max_exception_days = 2
cons_max_trading_days = 42
cons_min_pullback_pct = 0.04
cons_min_trading_days = 10
leg_max_gain_pct = 3.0
leg_max_trading_days = 60
leg_min_gain_pct = 0.35
leg_min_trading_days = 15
leg_min_up_close_ratio = 0.4
ma_rising_tol = 0.98
ma_touch_close_pct = 0.98
ma_touch_low_pct = 1.02
ma_touch_min_pct = 0.4
max_dist_52w_high_pct = 0.15
max_legup_high_to_t_trading_days = 60
min_adr_20_pct = 0.025
min_daily_range_pct = 0.025
pivot_lookback = 90
pivot_max_gap_open_pct = 0.15
pivot_min_lag = 15
pre_extend_5d_max = 0.08
pre_extend_close_vs_sma10_max = 0.06
pre_extend_consec_up_threshold = 1.003
pre_extend_max_consec_up = 2
pre_legup_lookback_days = 60
snap_low_window = 20
spacing_trading_days = 30
ticker_recycle_gap_days = 60
vol_surge_x = 1.5
```

## Output schema

Schema changed in this rewrite. Previous columns `base_start_date`, `base_end_date`, `base_duration_days`, `pullback_pct` were misnamed: they described the **leg up** before the pivot, not the consolidation. Rename map (decision 2 of the redesign):

| Old | New | Meaning |
|---|---|---|
| `base_start_date` | `legup_low_date` | low of the prior 35–300% advance |
| `base_end_date`   | `legup_high_date` | pivot high terminating the advance |
| `base_duration_days` | `legup_duration_days` | trading days (low→pivot) |
| `pullback_pct` (misnamed) | `legup_gain_pct` | actual leg-up gain |

New columns describing the **consolidation** between pivot and breakout:

- `cons_start_date` (= `legup_high_date`)
- `cons_end_date` (= the day before `date`)
- `cons_duration_days` (trading days, in [10, 42])
- `cons_max_drop_pct` (deepest dip from pivot during consolidation)
- `cons_exception_days` (count of bars in (.30, .35] drop band)
- `ma_touches_pct_in_cons` (fraction of cons bars touching 10/20/50-SMA from above)
- `breakout_volume_ratio` (volume[t] / 20d avg share volume)
- `breakout_range_pct` (daily range on breakout day)

Demoted (kept as supplementary stats over the **consolidation** window, NOT the leg-up — and not in the pass/fail rule): `higher_low_count`, `range_contraction_ratio`, `pct_closes_above_20ma_in_cons` (renamed from `_in_base`).

## Limitations

- **Daily OHLCV only.** breakouts.trade additionally gates on the first 30 minutes of intraday (volume + range scaled to a partial-day bar). We can't replicate this — the Polygon API key was cancelled for this session and only daily files are available. Some marginal-volume breakouts that would fail intraday confirmation will slip through; some valid breakouts whose intraday behavior is clean may be over-rejected by the daily-only filters. Net direction unknown but expected to be small.
- **Relative strength is the within-universe momentum percentile**, not stock_6m_return / SPY_6m_return (the working tool's gate). Decision 3 of the redesign was to keep `mom_pct` only; SPY data is not loaded.
- **No Episodic Pivot detection.** Gap-up/news-catalyst setups (Qullamaggie's other primary family) are out of scope for M1 (decision 7). Future work.
- **Reference-table location deviates from spec.** `data/raw/reference/` is a Windows junction we can't write through (see M0 audit); `yfinance_types.parquet` lives in `data/interim/reference/` instead.
- **yfinance type coverage is partial.** Source: **yfinance (3,901 tickers, 1,756 non-equity: 1,483 non-EQUITY + 263 non-US + 0 name-match ADR + 10 hardcoded-ADR fallback)**. Tickers absent from the cache are assumed EQUITY (safe direction; alternative is silently dropping common stocks the cache hasn't seen).

## 10 random sample setups (in `m1_sample_setups.csv`)

| Ticker | Date | Variant | mom_pct | legup_gain | pre_leg_ret | cons_dur | cons_drop | cons_slope | ma_touch | bo_vol_x | bo_range |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LVS | 2010-10-04 | loose | 0.981 | 0.514 | -0.097 | 15 | 0.041 | +0.00827 | 0.533 | 1.51 | 0.048 |
| NBIX | 2013-08-21 | strict | 0.950 | 0.372 | +0.047 | 26 | 0.093 | +0.00081 | 0.962 | 2.34 | 0.035 |
| EGY | 2014-09-17 | loose | 0.959 | 0.537 | -0.018 | 16 | 0.127 | -0.00456 | 0.688 | 1.91 | 0.059 |
| GWRE | 2016-06-01 | strict | 0.905 | 0.367 | -0.222 | 23 | 0.082 | +0.00197 | 0.957 | 3.91 | 0.045 |
| NGVT | 2017-11-28 | loose | 0.820 | 0.363 | -0.035 | 15 | 0.058 | +0.00327 | 0.933 | 1.50 | 0.027 |
| TDOC | 2020-07-30 | strict | 0.981 | 0.458 | +0.590 | 15 | 0.114 | -0.00080 | 0.867 | 2.33 | 0.134 |
| QTWO | 2020-12-18 | loose | 0.836 | 0.365 | +0.103 | 18 | 0.057 | +0.00301 | 0.611 | 1.57 | 0.031 |
| ANET | 2024-03-18 | strict | 0.907 | 0.372 | +0.126 | 24 | 0.135 | +0.00343 | 0.833 | 1.58 | 0.031 |
| PPTA | 2025-01-06 | strict | 0.983 | 0.538 | +0.388 | 18 | 0.219 | -0.00746 | 0.611 | 2.56 | 0.086 |
| ATYR | 2025-07-23 | strict | 0.988 | 1.282 | -0.283 | 29 | 0.201 | +0.00273 | 0.586 | 4.44 | 0.219 |
