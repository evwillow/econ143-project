# M6 — Stationary Block Bootstrap Inference + Placebo Test

Pre-registered design (writeup §6): 95% bootstrap CI on β(τ=0.90) − β(τ=0.50) for `vol_contraction_ratio_w`, **headline panel = loose**, training window 2010-2017, **stationary block bootstrap with mean block length = 30**. Hypothesis survives if the CI excludes zero with negative sign. M5 already showed sign-consistency in 7/8 expanding-window OOS years (loose); this stage handles the second leg of the prereg (the bootstrap CI).

Knobs: `B_real=1,000`, `B_placebo=500`, `mean_block_len=30`, `max_iter=5,000` (per M5's IterationLimitWarning). RNG seed = `20260514`. Total wall time: **8.5 min**.

Convention: lower `vol_contraction_ratio` = more contraction = stronger Qullamaggie signal. Predicted sign of β(0.90) − β(0.50) is **negative** (more contraction lifts the upside tail more than the median). One-sided bootstrap p-value reported = P̂[diff ≥ 0].

## Canonical / loose (headline pre-registered test)

- Training window: **[2010-01-01, 2017-12-31]**, n = **925**
- Point estimate: β(0.50) = +0.0026, β(0.90) = -0.0019, **diff = -0.0045**
- Bootstrap convergence: 992 of 1,000 draws converged in < 5,000 iterations (99.20%) | wall: 0.6 min

### Bootstrap distribution of β(0.90) − β(0.50)

| n | mean | median | std | p2.5 | p97.5 |
|---:|---:|---:|---:|---:|---:|
| 992 | +0.00067 | -0.00041 | 0.02086 | -0.03778 | +0.04247 |

### 95% confidence intervals

| Method | Lower | Upper | Excludes 0 (one-sided neg)? |
|---|---:|---:|---|
| Percentile | -0.03778 | +0.04247 | NO |
| Basic-bootstrap | -0.05153 | +0.02871 | NO |

**One-sided bootstrap p-value** (H₀: diff ≥ 0 vs H₁: diff < 0): `P̂[diff ≥ 0] = 0.4955`.

### Placebo (vcr_w randomly permuted within each draw)

- Placebo draws: 499 converged of 500 attempted (99.80%) | wall: 0.3 min

| n | mean | median | std | p2.5 | p97.5 |
|---:|---:|---:|---:|---:|---:|
| 499 | +0.00019 | -0.00052 | 0.01627 | -0.02972 | +0.03252 |

**Placebo centering**: ✅ near zero — placebo mean = +0.00019. Under the null, this should be ~0; large deviations flag a problem with the procedure or sample asymmetry.

**Where does the point estimate sit on the placebo null?** θ̂ = -0.00453; placebo mean = +0.00019, std = 0.01627. z = **-0.29**. Fraction of placebo draws ≤ θ̂ = **0.3868** (placebo-based one-sided p ≈ 0.3868).

## Canonical / strict (§7.1 robustness)

- Training window: **[2010-01-01, 2017-12-31]**, n = **726**
- Point estimate: β(0.50) = -0.0005, β(0.90) = -0.0018, **diff = -0.0013**
- Bootstrap convergence: 993 of 1,000 draws converged in < 5,000 iterations (99.30%) | wall: 1.2 min

### Bootstrap distribution of β(0.90) − β(0.50)

| n | mean | median | std | p2.5 | p97.5 |
|---:|---:|---:|---:|---:|---:|
| 993 | +0.00690 | +0.00480 | 0.02370 | -0.03291 | +0.05430 |

### 95% confidence intervals

| Method | Lower | Upper | Excludes 0 (one-sided neg)? |
|---|---:|---:|---|
| Percentile | -0.03291 | +0.05430 | NO |
| Basic-bootstrap | -0.05687 | +0.03034 | NO |

**One-sided bootstrap p-value** (H₀: diff ≥ 0 vs H₁: diff < 0): `P̂[diff ≥ 0] = 0.5785`.

### Placebo (vcr_w randomly permuted within each draw)

- Placebo draws: 494 converged of 500 attempted (98.80%) | wall: 1.4 min

| n | mean | median | std | p2.5 | p97.5 |
|---:|---:|---:|---:|---:|---:|
| 494 | -0.00104 | -0.00076 | 0.02009 | -0.04108 | +0.03933 |

**Placebo centering**: ✅ near zero — placebo mean = -0.00104. Under the null, this should be ~0; large deviations flag a problem with the procedure or sample asymmetry.

**Where does the point estimate sit on the placebo null?** θ̂ = -0.00128; placebo mean = -0.00104, std = 0.02009. z = **-0.01**. Fraction of placebo draws ≤ θ̂ = **0.4858** (placebo-based one-sided p ≈ 0.4858).

## Exploratory / loose, training 2010-2024 (§7 — non-pre-registered)

- Training window: **[2010-01-01, 2024-12-31]**, n = **2,692**
- Point estimate: β(0.50) = +0.0124, β(0.90) = -0.0086, **diff = -0.0210**
- Bootstrap convergence: 990 of 1,000 draws converged in < 5,000 iterations (99.00%) | wall: 5.1 min

### Bootstrap distribution of β(0.90) − β(0.50)

| n | mean | median | std | p2.5 | p97.5 |
|---:|---:|---:|---:|---:|---:|
| 990 | -0.02548 | -0.02565 | 0.01322 | -0.05072 | +0.00031 |

### 95% confidence intervals

| Method | Lower | Upper | Excludes 0 (one-sided neg)? |
|---|---:|---:|---|
| Percentile | -0.05072 | +0.00031 | NO |
| Basic-bootstrap | -0.04227 | +0.00877 | NO |

**One-sided bootstrap p-value** (H₀: diff ≥ 0 vs H₁: diff < 0): `P̂[diff ≥ 0] = 0.0293`.

## Pre-registration verdicts

- **Canonical loose (HEADLINE, prereg §6)**: 95% percentile CI = [-0.03778, +0.04247]. **Does NOT exclude zero with predicted negative sign** ❌
- **Canonical strict (§7.1)**: 95% percentile CI = [-0.03291, +0.05430]. Does NOT exclude zero with predicted negative sign

## §7 exploratory finding (NOT pre-registered)

Per M5, the |β(0.90) − β(0.50)| point estimate **grew** from −0.0045 (training 2010-2017) to −0.0210 (training 2010-2024). Re-running the bootstrap on the latter (loose, 2010-2024) is exploratory: it isn't the §6 prereg (which fixed 2010-2017) but it's the result that best characterises what the data looks like with the most information.

- **Exploratory loose 2010-2024**: n = 2,692, point estimate = **-0.02098**. 95% percentile CI = [-0.05072, +0.00031]. Does NOT exclude zero. One-sided p̂ = 0.0293.

## Computational details

| Item | Value |
|---|---|
| B (real bootstrap) | 1,000 draws |
| B (placebo) | 500 draws |
| Mean block length | 30 (Politis-Romano stationary block) |
| QR `max_iter` | 5,000 |
| RNG seed | 20260514 (numpy `default_rng`) |
| Total wall time | 8.5 min |

## Notes

- **Two-stage caveat**: the bootstrap resamples training rows and refits the QR, but the M3 factor / sector / year-FE residualization is held fixed at the canonical (or exploratory) point estimate. First-stage uncertainty is therefore not propagated. The justification is that the residualization is a high-degrees-of-freedom OLS and the QR slope on residualized y is the dominant source of variability.
- **Block bootstrap on cross-sectional data**: the panel is sorted by date before resampling, so blocks of consecutive rows correspond to setups whose breakout days are close in calendar time. Setups from the same calendar period tend to share market-state shocks, so a stationary block resample is preferable to IID for the std error estimate even though we're running a cross-section per draw.
- **Placebo construction**: `vol_contraction_ratio_w` is randomly permuted across the resampled rows on every draw, breaking its association with `fwd_ret_20d_resid` while leaving everything else (other covariates, factor structure, sample size) unchanged. The placebo distribution should be centered near zero — if it isn't, the procedure or the sample is not symmetric and the headline test needs care.
- **One-sided p-value formula**: `(n_draws_with_diff_>=_0 + 1) / (n_draws + 1)` (the +1 is the standard small-sample correction).
- **Determinism**: with seed = `20260514` and the same input parquets, this script's output is byte-stable. statsmodels' QuantReg solver is deterministic.
