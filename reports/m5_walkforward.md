# M5 — Walk-Forward Expanding-Window OOS Parameter Stability (2018-2025)

For each y ∈ {2018, …, 2025}: refit M3 factor + sector + year-FE residualization on the **expanded training window** [2010-01-01, (y-1)-12-31], then fit QR(0.50) and QR(0.90) on the training residuals using the M4 spec: `fwd_ret_20d_resid ~ vol_contraction_ratio_w + adr_pct + base_duration_days + rs_slope_vs_spy`. The recorded β's are **training-window** estimates; the score year y is reported (`n_score`) for context but its data does not enter the regression. This is a parameter-stability check per the §6 prereg's "≥ 5 of 8 expanding-window OOS years" requirement, not OOS prediction performance.

## LOOSE panel

(headline)

| Year | Train end | n_train | n_score | β(τ=0.50) | β(τ=0.90) | β(0.90)-β(0.50) | Sign |
|---:|---|---:|---:|---:|---:|---:|---|
| 2018 | 2017-12-31 | 925 | 149 | +0.0026 | -0.0019 | -0.0045 | − |
| 2019 | 2018-12-31 | 1,074 | 155 | +0.0006 | +0.0095 | +0.0089 | + |
| 2020 | 2019-12-31 | 1,229 | 327 | +0.0032 | -0.0051 | -0.0083 | − |
| 2021 | 2020-12-31 | 1,556 | 414 | +0.0102 | -0.0089 | -0.0191 | − |
| 2022 | 2021-12-31 | 1,970 | 140 | +0.0057 | -0.0068 | -0.0125 | − |
| 2023 | 2022-12-31 | 2,110 | 220 | +0.0092 | -0.0050 | -0.0142 | − |
| 2024 | 2023-12-31 | 2,330 | 362 | +0.0110 | -0.0065 | -0.0175 | − |
| 2025 | 2024-12-31 | 2,692 | 381 | +0.0124 | -0.0086 | -0.0210 | − |

**Sign-consistency**: 7 of 8 years had β(0.90)−β(0.50) < 0 (predicted direction). Pre-registered threshold is ≥ 5 of 8 → **PASS** (headline).

Positive-sign years (against the predicted direction): [2019].

**Magnitude stability** (|β(0.90)−β(0.50)|): first (2018) = 0.0045, last (2025) = 0.0210, trend = **growing**, mean = 0.0132, std = 0.0058.

## STRICT panel

(§7.1 robustness)

| Year | Train end | n_train | n_score | β(τ=0.50) | β(τ=0.90) | β(0.90)-β(0.50) | Sign |
|---:|---|---:|---:|---:|---:|---:|---|
| 2018 | 2017-12-31 | 726 | 124 | -0.0005 | -0.0018 | -0.0013 | − |
| 2019 | 2018-12-31 | 850 | 130 | -0.0041 | +0.0171 | +0.0212 | + |
| 2020 | 2019-12-31 | 980 | 204 | -0.0015 | +0.0041 | +0.0056 | + |
| 2021 | 2020-12-31 | 1,184 | 237 | +0.0129 | +0.0132 | +0.0003 | + |
| 2022 | 2021-12-31 | 1,421 | 106 | +0.0207 | +0.0176 | -0.0031 | − |
| 2023 | 2022-12-31 | 1,527 | 166 | +0.0148 | +0.0176 | +0.0027 | + |
| 2024 | 2023-12-31 | 1,693 | 246 | +0.0266 | +0.0071 | -0.0195 | − |
| 2025 | 2024-12-31 | 1,939 | 274 | +0.0232 | +0.0051 | -0.0181 | − |

**Sign-consistency**: 4 of 8 years had β(0.90)−β(0.50) < 0 (predicted direction). Pre-registered threshold is ≥ 5 of 8 → **FAIL** (robustness).

Positive-sign years (against the predicted direction): [2019, 2020, 2021, 2023].

**Magnitude stability** (|β(0.90)−β(0.50)|): first (2018) = 0.0013, last (2025) = 0.0181, trend = **growing**, mean = 0.0090, std = 0.0090.

## Pre-registration verdict

From writeup §6: "the same sign holds in at least five of eight expanding-window OOS years (2018-2025)." The headline panel for the prereg is **loose**.

- **Loose (headline)**: 7 / 8 negative — **PASS** vs the ≥5/8 bar.
- **Strict (§7.1)**: 4 / 8 negative — **FAIL** vs the ≥5/8 bar.

The prereg's bootstrap CI inference (M6) is independent of this stability check — the two requirements (CI excludes zero with predicted sign **and** sign-consistency in ≥5 of 8 years) are both needed for the hypothesis to survive.

## Notes

- **What "OOS" means here**: each year y *adds* the prior year (y-1) to the training set, then we recompute the M4 headline statistic on that newly-expanded training set. The score year y itself is recorded as `n_score` for context only — its rows do not enter any regression. The check passes if the predicted sign is robust to where you cut the training data.
- **Determinism**: each year's `m3_factors.residualize` and `statsmodels.QuantReg.fit(q=τ)` are deterministic given fixed input data and τ. Re-running this script on the same M2 + factor-panel inputs yields byte-identical `m5_oos_results.parquet`.
- **Re-fit per year**: the factor-OLS, the sector/year-FE design matrix, the p99 winsor cap on `vol_contraction_ratio`, **and** the QR fits are all redone for every y. This means each year's `vol_contraction_ratio_w` cap is the p99 of THAT year's training set, which can drift slightly as the expanding window adds more post-2017 outliers.
- **No bootstrap inference**: this report is point estimates and sign-consistency only. CIs come in M6.
