# EC143 Final Project — Writeup

## §6 Pre-Registered Hypothesis (locked May 3, 2026)

We pre-register a one-sided test of whether the volume contraction ratio shifts the τ=0.90 conditional quantile of factor-residualized 20-day forward returns more than it shifts the τ=0.50 conditional quantile, after controlling for the 12-1 momentum factor, sector fixed effects, and year fixed effects. The test statistic is β(0.90) − β(0.50), estimated on the 2010–2017 training period. Inference uses a stationary block bootstrap with mean block length 30. The hypothesis survives if (a) the 95% bootstrap CI of the statistic excludes zero with the predicted sign, and (b) the same sign holds in at least five of eight expanding-window OOS years (2018–2025). The headline test runs on the **loose** universe (mom_pct ≥ 0.80) to maximize sample size for quantile regression; the **strict** universe (mom_pct ≥ 0.90 plus the 15%-of-52w-high gate) is reported in §7.1 as a sensitivity analysis.

§7 Results

[3-4 sentences: the headline. The pre-registered test failed. State θ̂, CI, p-value. State that placebo confirms low power.]

§7.1 Strict universe (robustness)
[2-3 sentences: also fails, attenuates under stricter gate, possible reasons.]

§7.2 Expanding-window exploratory analysis  
[4-5 sentences: M5's sign consistency, magnitude trajectory, the 2024-cutoff CI's near-miss. Frame as "consistent with a small real effect the prereg sample couldn't detect," NOT as a post-hoc finding.]

§7.3 Ancillary quantile-regression finding
[3-4 sentences: adr_pct sign flip across τ, t-stats both >4 at the tails, OLS hides it entirely. This is the methodological contribution — QR reveals heterogeneity OLS misses.]

§8 Limitations
[Bullet list, honest: M1 6.5/10 QC, daily-only, no SPY-RS, prereg sample size (n=925), placebo power.]