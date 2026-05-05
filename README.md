# EC143 Final Project

Pre-registered quantile-regression test of whether Qullamaggie-style volume contraction during a stock's consolidation base raises the upper tail of 20-day forward returns more than it raises the median, after controlling for FF3+UMD factor returns, sector fixed effects, and year fixed effects.

**Deliverable:** [`reports/writeup.md`](reports/writeup.md). Per-stage validation reports live alongside it as `reports/m*_validation.md`. Tables and figures used in the writeup are at [`reports/writeup_assets/`](reports/writeup_assets/).

**Pre-registered hypothesis:** `reports/writeup.md` §6 (locked 2026-05-03, before any QR fits ran).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate     # Windows; on POSIX use `source .venv/bin/activate`
pip install -e .
```

Python ≥ 3.11 required.

## Data

`data/raw/` is gitignored — it sources Polygon daily bars from a separate `breakoutStudyTool` pipeline and is not redistributable. Generated artefacts (`data/interim/`, `data/factors/`, `reports/writeup_assets/`) are committed, so a grader can re-run from M4 onward without rerunning M0–M3.

## Reproduction

Run modules in order; each writes its outputs and a validation report.

1. `python src/m0_audit.py` — survivorship + bad-bar audit. Writes `data/interim/audit_summary.json`, `reports/audit.md`.
2. `python src/m1_universe.py` — Qullamaggie consolidation-breakout detector. Writes `data/interim/setups.parquet`, `reports/m1_validation.md`.
3. `python src/m2_features.py` — per-setup features (vol contraction, ADR, RS-vs-SPY, sector). Writes `data/interim/setups_with_features.parquet`, `reports/m2_validation.md`.
4. `python src/m3_factors.py` — FF3+UMD factor panel + 20-day-forward residualization on 2010–2017 training. Writes `data/factors/ff3_umd_daily.parquet`, `data/interim/setups_with_residuals.parquet`, `reports/m3_validation.md`.
5. `python src/m4_estimation.py` — OLS + QR(τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}) on training. Writes `data/interim/m4_results.parquet`, `reports/m4_estimation.md`.
6. `python src/m5_walkforward.py` — expanding-window OOS sign-consistency check, 2018–2025. Writes `data/interim/m5_oos_results.parquet`, `reports/m5_walkforward.md`.
7. `python src/m6_bootstrap.py` — stationary block bootstrap CI + placebo (~8 min). Writes `data/interim/m6_bootstrap.parquet`, `reports/m6_inference.md`.
8. `python src/m7_assets.py` — writeup tables + figures. Writes under `reports/writeup_assets/`.

`src/_fetch_ticker_types.py` and `src/_fetch_ticker_adr_fields.py` are one-off helpers used to populate `data/interim/reference/yfinance_types.parquet` (the security-type filter cache). They re-fetch from yfinance and shouldn't need to run again.
