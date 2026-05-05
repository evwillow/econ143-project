# `data/` — flat layout

All committed pipeline outputs and cached reference data live at the top level of `data/`. Files prefixed `m{N}_` are the canonical output of stage M{N} of the pipeline. Files without that prefix are cached source data (factor panel, SPY bars, yfinance metadata).

| File | Stage / source | What it is |
|---|---|---|
| `m0_audit_summary.json` | M0 | survivorship + bad-bar audit summary on the daily-bar source |
| `m1_setups.parquet` | M1 | 5,289 Qullamaggie consolidation-breakout setups (loose + strict variants stacked) |
| `m2_setups_with_features.parquet` | M2 | M1 + 5 per-setup feature columns (vol contraction, ADR, base duration, RS-vs-SPY, sector) |
| `m3_setups_with_residuals.parquet` | M3 | M2 + factor-residualized 20-day forward returns + winsorized vcr; 5,286 rows |
| `m4_results.parquet` | M4 | OLS + QR(τ) coefficient table on the 2010–2017 training window, both panels × both specs |
| `m5_oos_results.parquet` | M5 | expanding-window walk-forward β diffs, 2018–2025, both panels |
| `m6_bootstrap.parquet` | M6 | 3,971 bootstrap draws (real + placebo) of β(0.90) − β(0.50) across three panel × window combinations |
| `ff3_umd_daily.parquet` | M3 (cached) | Ken French daily FF3 + UMD factors, 2008–2026, parsed and joined |
| `spy_daily.parquet` | M2 (cached) | SPY daily closes (auto-adjusted), 2008–2026, used for RS-vs-SPY feature |
| `yfinance_types.parquet` | M1 (cached) | yfinance `.info` cache for the security-type filter (quote_type, country, long_name, sector, industry) |
| `yfinance_failures.csv` | M1 (cached) | tickers for which yfinance returned no data; logged for transparency |
| `F-F_Research_Data_Factors_daily_CSV.zip` | source | Ken French FF3 daily zip (committed for source-version provenance) |
| `F-F_Momentum_Factor_daily_CSV.zip` | source | Ken French UMD daily zip (committed for source-version provenance) |
| `raw/` | gitignored | Polygon flat-file daily bars from the breakoutStudyTool pipeline; not redistributable |
