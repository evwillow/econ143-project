# M2 — Per-Setup Feature Validation

Inputs: M1 setups (`data/interim/setups.parquet`, 5,289 rows: 2,214 strict + 3,075 loose).

Outputs: 5 features added per row (`vol_contraction_ratio`, `adr_pct`, `base_duration_days`, `rs_slope_vs_spy`, `sector`). All M1 columns preserved.

- Output rows: **5,289** (0 dropped — see table at bottom).
- SPY source: `data\interim\spy_daily.parquet` (4,568 bars, 2008-01-02 to 2026-02-27, `auto_adjust=True` to match split-adjusted daily bars).

## NaN counts per feature, by universe variant

| Feature | strict NaN | strict total | strict NaN % | loose NaN | loose total | loose NaN % | overall NaN % |
|---|---:|---:|---:|---:|---:|---:|---:|
| vol_contraction_ratio | 0 | 2,214 | 0.00% | 0 | 3,075 | 0.00% | 0.00% |
| adr_pct | 0 | 2,214 | 0.00% | 0 | 3,075 | 0.00% | 0.00% |
| base_duration_days | 0 | 2,214 | 0.00% | 0 | 3,075 | 0.00% | 0.00% |
| rs_slope_vs_spy | 0 | 2,214 | 0.00% | 0 | 3,075 | 0.00% | 0.00% |
| sector (sector counts 'Unknown' as missing) | 197 | 2,214 | 8.90% | 267 | 3,075 | 8.68% | 8.77% |

## Distribution stats (numeric features)

| Feature | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| vol_contraction_ratio | 0.2076 | 0.6338 | 0.7671 | 0.9433 | 1.1810 | 1.4895 | 15.3902 |
| adr_pct | 0.0174 | 0.0279 | 0.0315 | 0.0379 | 0.0484 | 0.0612 | 0.1343 |
| base_duration_days | 15 | 15 | 16 | 20 | 27 | 34 | 42 |
| rs_slope_vs_spy | -0.0161 | -0.0038 | -0.0018 | 0.0004 | 0.0028 | 0.0059 | 0.0233 |

## Sanity check vs Qullamaggie expected ranges

| Feature | expected | observed (p25–p75) | observed (p10–p90) | verdict |
|---|---|---|---|---|
| vol_contraction_ratio | [0.4, 1.5] | [0.767, 1.181] | [0.634, 1.489] | IQR inside expected |
| adr_pct | [0.02, 0.08] | [0.031, 0.048] | [0.028, 0.061] | IQR inside expected |
| base_duration_days | matches M1 cons range [10, 42] | [16, 27] | [15, 34] | matches |

## Correlation matrix (Pearson, complete-case)

|  | vol_contraction_ratio | adr_pct | base_duration_days | rs_slope_vs_spy |
|---|---:|---:|---:|---:|
| **vol_contraction_ratio** | +1.000 | +0.074 | -0.088 | +0.237 |
| **adr_pct** | +0.074 | +1.000 | -0.075 | +0.130 |
| **base_duration_days** | -0.088 | -0.075 | +1.000 | -0.050 |
| **rs_slope_vs_spy** | +0.237 | +0.130 | -0.050 | +1.000 |

## Sector counts

| Sector | strict | loose | total |
|---|---:|---:|---:|
| Healthcare | 513 | 677 | 1,190 |
| Technology | 445 | 627 | 1,072 |
| Consumer Cyclical | 265 | 404 | 669 |
| Industrials | 267 | 372 | 639 |
| Unknown | 197 | 267 | 464 |
| Financial Services | 131 | 170 | 301 |
| Energy | 129 | 159 | 288 |
| Basic Materials | 95 | 142 | 237 |
| Communication Services | 77 | 123 | 200 |
| Consumer Defensive | 73 | 96 | 169 |
| Real Estate | 15 | 31 | 46 |
| Utilities | 7 | 7 | 14 |

## Dropped setups

- None.

## Notes

- **Half-split rule** (vol_contraction_ratio): always split base in half exactly (`floor(N/2)` / `ceil(N/2)`), regardless of `cons_duration_days`. This interprets the task brief's "`If cons_duration_days < 20: split base in half exactly`" as applying universally — the `<20` qualifier emphasizes that no minimum is required, and no separate rule is specified for `>=20`. If a different rule was intended (e.g. last-week-vs-first-week, or quintile splits) for longer bases, this needs revisiting before M3.
- **rs_slope_vs_spy**: x is the original day index in [0, N-1] over the base window. Days where SPY data is missing (e.g. exchange holidays the stock trades but SPY doesn't) are skipped from y but their indices are preserved in x — the regression sees gapped x's. When fewer than 3 valid bars remain, the slope is NaN.
- **SPY adjustment**: fetched with `auto_adjust=True` so split/dividend adjustments match the breakoutStudyTool daily bars (which are split-adjusted). If the daily-bar pipeline ever changes adjustment convention, refetch SPY.
- **Sector**: from yfinance `.info['sector']` cached at `data/interim/reference/yfinance_types.parquet`. Tickers whose yfinance fetch failed (~119 of 1,773 in the M1 cache) get sector = 'Unknown'. M2 doesn't refetch.
