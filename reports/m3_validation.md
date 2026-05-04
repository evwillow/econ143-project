# M3 — Forward Returns + Factor Residualization Validation

Inputs: M2 setups (`data/interim/setups_with_features.parquet`, 5,289 rows). Outputs: `data/interim/setups_with_residuals.parquet` (5,286 rows after dropping setups missing fwd_ret_20d or factor window data) plus per-row residuals + winsorized `vol_contraction_ratio_w`.

Training window: **2010-01-01 to 2017-12-31**. OOS: 2018-01-01 to 2025-12-31. OOS residuals are computed by applying the trained coefficients (factor betas + sector dummies + intercept). Year fixed-effect dummies are encoded only for training years; OOS rows get all year dummies = 0 (i.e. mapped to the reference year for scoring). This means OOS residuals carry any year-level drift as an additive offset, but cross-sectional variation -- which is what the M4 quantile regression cares about -- is preserved.

## Setup counts by year (before / after dropping for missing fwd_ret_20d)

| Year | Input rows | Kept rows | Dropped |
|---:|---:|---:|---:|
| 2010 | 205 | 205 | 0 |
| 2011 | 216 | 216 | 0 |
| 2012 | 166 | 166 | 0 |
| 2013 | 250 | 250 | 0 |
| 2014 | 188 | 188 | 0 |
| 2015 | 168 | 168 | 0 |
| 2016 | 148 | 148 | 0 |
| 2017 | 310 | 310 | 0 |
| 2018 | 273 | 273 | 0 |
| 2019 | 285 | 285 | 0 |
| 2020 | 531 | 531 | 0 |
| 2021 | 651 | 651 | 0 |
| 2022 | 246 | 246 | 0 |
| 2023 | 386 | 386 | 0 |
| 2024 | 608 | 608 | 0 |
| 2025 | 658 | 658 | 0 |
| **Total** | **5,289** | **5,286** | **3** |

## Dropped setups (rolled up by reason)

- t+20 beyond end of bar series: 3
- factor panel missing date(s) for setup window: 3

## fwd_ret_20d distribution (raw)

| Slice | n | mean | std | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| training (2010-2017) | 1,651 | -0.0084 | 0.1449 | -0.1667 | -0.0068 | +0.1460 |
| OOS (2018-2025) | 3,635 | -0.0007 | 0.1834 | -0.1841 | -0.0146 | +0.1840 |

## fwd_ret_20d_resid distribution (residualized)

| Slice | n | mean | std | p10 | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| training (2010-2017) | 1,651 | +0.0000 | 0.1366 | -0.1488 | -0.0022 | +0.1441 |
| OOS (2018-2025) | 3,635 | +0.0074 | 0.1705 | -0.1618 | -0.0059 | +0.1798 |

**Sanity check**: training residual mean = +0.000000. PASS (threshold |x| <= 1e-3 -- mechanical given OLS fit with intercept on the training window).

## Factor regression on training window

- Model: `fwd_ret_20d ~ const + Mkt-RF + SMB + HML + UMD + sector FE + year FE`
- Sector FE: 11 categories, reference = `Healthcare` (most-populated). Year FE: 8 categories, reference = `2010` (earliest). All factor variables are 20-trading-day cumulated decimal returns over (t, t+20].
- n = 1,651, k = 22, R² = 0.1109, Adj R² = 0.0994

### Factor coefficients (+ intercept)

| Term | Coef | Std err | t | p |
|---|---:|---:|---:|---:|
| `const` | -0.0116 | 0.0135 | -0.86 | 0.39 |
| `mkt_rf_window` | +0.9043 | 0.1169 | +7.73 | 1.84e-14 |
| `smb_window` | +1.2034 | 0.1900 | +6.33 | 3.08e-10 |
| `hml_window` | -0.0540 | 0.1819 | -0.30 | 0.767 |
| `umd_window` | +0.4484 | 0.1347 | +3.33 | 0.000893 |

### Sector fixed effects (deviations from reference)

| Sector | Coef | Std err | t | p |
|---|---:|---:|---:|---:|
| Basic Materials | +0.0034 | 0.0175 | +0.19 | 0.846 |
| Communication Services | -0.0042 | 0.0188 | -0.23 | 0.822 |
| Consumer Cyclical | -0.0063 | 0.0127 | -0.50 | 0.616 |
| Consumer Defensive | -0.0118 | 0.0212 | -0.56 | 0.577 |
| Energy | -0.0134 | 0.0171 | -0.78 | 0.435 |
| Financial Services | +0.0008 | 0.0162 | +0.05 | 0.961 |
| Industrials | +0.0084 | 0.0128 | +0.65 | 0.514 |
| Real Estate | +0.0381 | 0.0395 | +0.96 | 0.335 |
| Technology | -0.0121 | 0.0115 | -1.05 | 0.295 |
| Unknown | -0.0089 | 0.0124 | -0.71 | 0.475 |

### Year fixed effects (deviations from reference)

| Year | Coef | Std err | t | p |
|---|---:|---:|---:|---:|
| 2011 | -0.0088 | 0.0141 | -0.63 | 0.531 |
| 2012 | +0.0003 | 0.0150 | +0.02 | 0.985 |
| 2013 | +0.0120 | 0.0139 | +0.87 | 0.386 |
| 2014 | +0.0010 | 0.0148 | +0.07 | 0.946 |
| 2015 | -0.0192 | 0.0151 | -1.28 | 0.202 |
| 2016 | -0.0052 | 0.0157 | -0.33 | 0.739 |
| 2017 | -0.0031 | 0.0136 | -0.23 | 0.817 |

## Winsorization of vol_contraction_ratio

- Training-window p99 = **2.8519**. Training raw max = 6.2560.
- Capped: 15 training rows (of 1,651, 0.91%) and 31 total rows of 5,286 (0.59%).
- Output column `vol_contraction_ratio_w` = `min(vol_contraction_ratio, 2.8519)`. The raw column is kept alongside for reference.

## Correlation of fwd_ret_20d_resid with M2 features (complete-case Pearson)

| Feature | Pearson ρ |
|---|---:|
| `vol_contraction_ratio` | +0.0256 |
| `vol_contraction_ratio_w` | +0.0424 |
| `adr_pct` | +0.0456 |
| `base_duration_days` | -0.0076 |
| `rs_slope_vs_spy` | +0.0496 |

Interpretation note: ρ for `vol_contraction_ratio_w` is the linear association *at the conditional mean*. The pre-registered hypothesis is about the **τ=0.90 quantile** vs the τ=0.50 quantile, so a near-zero linear ρ does not imply the hypothesis fails -- it implies the mean-effect channel is small.

## Notes / caveats

- **Residualization is two-pass**. We fit one OLS on the training subset (2010-2017) with sector + year FE included; we then apply the fitted coefficient vector to **every** row (training and OOS) to get fwd_ret_20d_resid. Year-FE dummies are not encoded for OOS years, so OOS rows are scored as if they were the reference year (2010). The induced level shift on OOS residuals is constant per OOS year and doesn't affect cross-sectional inferences.
- **Factor windows use cum-sum subtraction** (`mkt_rf_cum[t+20] - mkt_rf_cum[t]`). This is exactly the sum of the daily decimal factor returns over (t, t+20] when the setup date `t` and `fwd_end_date` are both in the factor panel. If either date isn't, the window sum is null and the row is dropped from the residualization frame (reported under "Dropped setups" above).
- **Setups appearing in both strict and loose** (the same ticker+date) get the **same** fwd_ret_20d, factor windows, and residual. They differ only in the universe_variant column. This is intentional -- the variants share the underlying chart event.
- **Determinism**: same input parquets -> same output parquet. The factor panel is not refetched if it already exists at `data/factors/ff3_umd_daily.parquet`.
