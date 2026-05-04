# EC143 Final Project — Writeup

## §6 Pre-Registered Hypothesis (locked May 3, 2026)

We pre-register a one-sided test of whether the volume contraction ratio shifts the τ=0.90 conditional quantile of factor-residualized 20-day forward returns more than it shifts the τ=0.50 conditional quantile, after controlling for the 12-1 momentum factor, sector fixed effects, and year fixed effects. The test statistic is β(0.90) − β(0.50), estimated on the 2010–2017 training period. Inference uses a stationary block bootstrap with mean block length 30. The hypothesis survives if (a) the 95% bootstrap CI of the statistic excludes zero with the predicted sign, and (b) the same sign holds in at least five of eight expanding-window OOS years (2018–2025).