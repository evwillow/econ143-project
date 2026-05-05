# M4 — OLS + Quantile Regression Estimation (training 2010-2017)

Pre-registered model (writeup §6): `fwd_ret_20d_resid ~ vol_contraction_ratio_w + adr_pct + base_duration_days + rs_slope_vs_spy`. Factor + sector + year FE already absorbed by the M3 residualization, so this stage adds **no** additional controls. OLS is reported alongside QuantReg at τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}. The pre-registered test statistic is **β(0.90) − β(0.50)** on `vol_contraction_ratio_w`, **headline panel = loose**. Bootstrap inference is M6's job — this report is point estimates only.

## Sample sizes (training window, 2010-2017)

| Panel | n |
|---|---:|
| loose | 925 |
| strict | 726 |

## Headline statistic — β(0.90) − β(0.50) on `vol_contraction_ratio_w`

| Panel | β(τ=0.50) | β(τ=0.90) | β(0.90) − β(0.50) |
|---|---:|---:|---:|
| loose (headline) | +0.0026 | -0.0019 | **-0.0045** |
| strict | -0.0005 | -0.0018 | -0.0013 |

Sign convention: `vol_contraction_ratio` = mean(volume, second half) / mean(volume, first half). Lower = more contraction = stronger Qullamaggie signal. The pre-registered prediction is that contraction lifts the **upside tail** of forward returns more than it shifts the median, i.e. a **more-negative β at τ=0.90 than at τ=0.50** — so the predicted sign of β(0.90) − β(0.50) is **negative**. Inference (stationary block bootstrap, mean block length 30) is M6.

## LOOSE panel

### primary (winsorized vcr)

Cells show **β (t-stat)**. R² / pseudo-R² is the OLS R² for the OLS row and Koenker-Machado pseudo-R² for each QR row.

| model/τ | const | vol_contraction_ratio_w | adr_pct | base_duration_days | rs_slope_vs_spy | n | R² / pseudo-R² |
|---|---|---|---|---|---|---|---|
| OLS | +0.0742 (_+3.11_) | -0.0066 (_-0.59_) | -1.7511 (_-4.38_) | -0.0000 (_-0.05_) | -0.1997 (_-0.16_) | 925 | 0.0225 |
| QR τ=0.10 | +0.0612 (_+2.12_) | -0.0235 (_-1.60_) | -5.1747 (_-10.16_) | +0.0007 (_+0.95_) | +3.8956 (_+2.80_) | 925 | 0.1052 |
| QR τ=0.25 | +0.0623 (_+2.51_) | -0.0124 (_-1.08_) | -3.3174 (_-8.13_) | +0.0002 (_+0.26_) | -0.4099 (_-0.34_) | 925 | 0.0642 |
| QR τ=0.50 | +0.0622 (_+2.80_) | +0.0026 (_+0.25_) | -2.0743 (_-5.58_) | +0.0003 (_+0.48_) | +0.2170 (_+0.18_) | 925 | 0.0174 |
| QR τ=0.75 | +0.0638 (_+2.72_) | +0.0106 (_+0.96_) | -0.3946 (_-0.97_) | +0.0001 (_+0.23_) | -0.1827 (_-0.15_) | 925 | 0.0014 |
| QR τ=0.90 | +0.0582 (_+1.66_) | -0.0019 (_-0.11_) | +2.6284 (_+4.62_) | -0.0005 (_-0.53_) | -0.0150 (_-0.01_) | 925 | 0.0175 |

### sensitivity (raw vcr)

Cells show **β (t-stat)**. R² / pseudo-R² is the OLS R² for the OLS row and Koenker-Machado pseudo-R² for each QR row.

| model/τ | const | vol_contraction_ratio | adr_pct | base_duration_days | rs_slope_vs_spy | n | R² / pseudo-R² |
|---|---|---|---|---|---|---|---|
| OLS | +0.0721 (_+3.13_) | -0.0047 (_-0.49_) | -1.7504 (_-4.38_) | -0.0000 (_-0.04_) | -0.2521 (_-0.20_) | 925 | 0.0224 |
| QR τ=0.10 | +0.0613 (_+2.20_) | -0.0227 (_-1.91_) | -5.1856 (_-10.20_) | +0.0007 (_+0.93_) | +3.8978 (_+2.84_) | 925 | 0.1051 |
| QR τ=0.25 | +0.0595 (_+2.50_) | -0.0097 (_-1.02_) | -3.2997 (_-8.11_) | +0.0002 (_+0.25_) | -0.4530 (_-0.38_) | 925 | 0.0641 |
| QR τ=0.50 | +0.0614 (_+2.87_) | +0.0049 (_+0.55_) | -2.1392 (_-5.76_) | +0.0003 (_+0.57_) | +0.1080 (_+0.09_) | 925 | 0.0176 |
| QR τ=0.75 | +0.0638 (_+2.82_) | +0.0106 (_+1.16_) | -0.3919 (_-0.96_) | +0.0001 (_+0.23_) | -0.1851 (_-0.15_) | 925 | 0.0012 |
| QR τ=0.90 | +0.0718 (_+2.05_) | -0.0110 (_-0.66_) | +2.5401 (_+4.42_) | -0.0006 (_-0.60_) | +0.0053 (_+0.00_) | 925 | 0.0179 |

### Interpretation — loose panel

On the **loose** panel, the slope of `vol_contraction_ratio_w` is +0.0026 at the median (τ=0.50) and -0.0019 at the upper tail (τ=0.90). The headline statistic β(0.90) − β(0.50) = **-0.0045** is negative. The point estimate is more negative at τ=0.90 vs τ=0.50; this is the test-statistic the M6 stationary block bootstrap will assign a CI to. A negative diff is the predicted direction (more contraction → stronger upside-tail effect than median-effect), since lower vcr_w = more contraction = stronger Qullamaggie signal.

## STRICT panel

### primary (winsorized vcr)

Cells show **β (t-stat)**. R² / pseudo-R² is the OLS R² for the OLS row and Koenker-Machado pseudo-R² for each QR row.

| model/τ | const | vol_contraction_ratio_w | adr_pct | base_duration_days | rs_slope_vs_spy | n | R² / pseudo-R² |
|---|---|---|---|---|---|---|---|
| OLS | +0.0583 (_+2.07_) | -0.0025 (_-0.19_) | -1.6545 (_-3.61_) | +0.0003 (_+0.37_) | +0.0678 (_+0.05_) | 726 | 0.0187 |
| QR τ=0.10 | +0.0478 (_+1.40_) | -0.0172 (_-0.99_) | -4.8778 (_-8.42_) | +0.0007 (_+0.81_) | +1.1755 (_+0.81_) | 726 | 0.1048 |
| QR τ=0.25 | +0.0592 (_+2.14_) | -0.0157 (_-1.21_) | -3.2229 (_-7.39_) | +0.0002 (_+0.31_) | -0.2182 (_-0.17_) | 726 | 0.0670 |
| QR τ=0.50 | +0.0539 (_+2.18_) | -0.0005 (_-0.04_) | -1.9508 (_-4.85_) | +0.0005 (_+0.82_) | +0.7543 (_+0.60_) | 726 | 0.0173 |
| QR τ=0.75 | +0.0134 (_+0.50_) | +0.0240 (_+1.91_) | +0.2515 (_+0.54_) | +0.0006 (_+0.85_) | +0.6638 (_+0.48_) | 726 | 0.0044 |
| QR τ=0.90 | +0.0594 (_+1.36_) | -0.0018 (_-0.09_) | +2.4773 (_+3.47_) | -0.0000 (_-0.04_) | +1.6676 (_+0.74_) | 726 | 0.0161 |

### sensitivity (raw vcr)

Cells show **β (t-stat)**. R² / pseudo-R² is the OLS R² for the OLS row and Koenker-Machado pseudo-R² for each QR row.

| model/τ | const | vol_contraction_ratio | adr_pct | base_duration_days | rs_slope_vs_spy | n | R² / pseudo-R² |
|---|---|---|---|---|---|---|---|
| OLS | +0.0585 (_+2.12_) | -0.0026 (_-0.22_) | -1.6545 (_-3.61_) | +0.0003 (_+0.37_) | +0.0769 (_+0.05_) | 726 | 0.0188 |
| QR τ=0.10 | +0.0521 (_+1.52_) | -0.0194 (_-1.22_) | -4.9847 (_-8.55_) | +0.0008 (_+0.83_) | +1.9227 (_+1.32_) | 726 | 0.1051 |
| QR τ=0.25 | +0.0576 (_+2.12_) | -0.0106 (_-0.96_) | -3.2747 (_-7.49_) | +0.0002 (_+0.22_) | -0.5350 (_-0.42_) | 726 | 0.0667 |
| QR τ=0.50 | +0.0508 (_+2.09_) | +0.0012 (_+0.11_) | -1.9634 (_-4.89_) | +0.0006 (_+0.94_) | +0.9354 (_+0.75_) | 726 | 0.0173 |
| QR τ=0.75 | +0.0198 (_+0.77_) | +0.0178 (_+1.74_) | +0.3232 (_+0.70_) | +0.0004 (_+0.63_) | +0.6076 (_+0.45_) | 726 | 0.0040 |
| QR τ=0.90 | +0.0594 (_+1.40_) | -0.0018 (_-0.11_) | +2.4769 (_+3.47_) | -0.0000 (_-0.04_) | +1.6674 (_+0.74_) | 726 | 0.0162 |

### Interpretation — strict panel

On the **strict** panel, the slope of `vol_contraction_ratio_w` is -0.0005 at the median (τ=0.50) and -0.0018 at the upper tail (τ=0.90). The headline statistic β(0.90) − β(0.50) = **-0.0013** is negative. The point estimate is more negative at τ=0.90 vs τ=0.50; this is the test-statistic the M6 stationary block bootstrap will assign a CI to. A negative diff is the predicted direction (more contraction → stronger upside-tail effect than median-effect), since lower vcr_w = more contraction = stronger Qullamaggie signal.

## OLS coefficients with |t| > 3 (primary spec, excluding intercept)

| Panel | Term | β | std err | t | p |
|---|---|---:|---:|---:|---:|
| loose | `adr_pct` | -1.7511 | 0.4000 | -4.38 | 1.34e-05 |
| strict | `adr_pct` | -1.6545 | 0.4580 | -3.61 | 0.000324 |

## Notes

- **Why vcr_w not raw vcr in the headline?** M2 flagged vol_contraction_ratio max=15.39 vs median=0.94 (heavy right tail from low first-half volume). Winsorizing at training p99 (2.85) keeps a small number of extreme rows from dominating the QR fit. Raw-vcr results are reported as a sensitivity in the same report.
- **Why no factor / sector / year controls in the regressor list?** They were already partialled out in M3 by residualizing fwd_ret_20d against FF3+UMD (cumulated over the 20-day forward window) plus sector and year fixed effects on the training window. The LHS here is `fwd_ret_20d_resid`, so adding these controls again would be double-counting.
- **t-stats on QR rows** are the asymptotic z-statistics from statsmodels' default kernel/IID covariance. They are useful as a rough guide but the pre-registered inference uses a stationary block bootstrap (M6), not these t-stats.
- **Determinism**: `sm.OLS(...).fit()` and `sm.QuantReg(...).fit(q=τ)` are both deterministic given fixed input and tau. Re-running the script on the same `data/interim/setups_with_residuals.parquet` yields byte-identical `data/interim/m4_results.parquet`.
