# EC143 Final Project

Pre-registered quantile-regression test of whether Qullamaggie-style volume contraction during a stock's consolidation base raises the upper tail of 20-day forward returns more than it raises the median, after controlling for FF3+UMD factor returns, sector fixed effects, and year fixed effects.

**Deliverable:** [`writeup.ipynb`](writeup.ipynb) at the repo root. Open it and run all cells — every figure, table, and number regenerates from the committed parquets in `data/`.

**Pre-registered hypothesis:** `writeup.ipynb` §6 (locked 2026-05-03, before any QR fits ran).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate     # Windows; on POSIX use `source .venv/bin/activate`
pip install -e .
```

Python ≥ 3.11 required.

## Data

All committed pipeline outputs and cached reference data live flat under `data/` — see [`data/README.md`](data/README.md) for a one-line description of each file. `data/raw/` is gitignored: it sources Polygon daily bars from a separate `breakoutStudyTool` pipeline and is not redistributable. Everything else `writeup.ipynb` needs is committed, so the notebook runs without re-executing the full pipeline.

## Reproduction

To verify only the writeup: open `writeup.ipynb` and `Run All`. The notebook loads the committed parquets and regenerates every output inline.

To re-run the full pipeline from scratch: execute the modules in order. Each writes its outputs to `data/` and emits a per-stage validation report to `reports/` as a side effect (`reports/` is gitignored).

1. `python src/m0_audit.py` — survivorship + bad-bar audit. Writes `data/m0_audit_summary.json`.
2. `python src/m1_universe.py` — Qullamaggie consolidation-breakout detector. Writes `data/m1_setups.parquet`.
3. `python src/m2_features.py` — per-setup features (vol contraction, ADR, RS-vs-SPY, sector). Writes `data/m2_setups_with_features.parquet`.
4. `python src/m3_factors.py` — FF3+UMD factor panel + 20-day-forward residualization on 2010–2017 training. Writes `data/ff3_umd_daily.parquet`, `data/m3_setups_with_residuals.parquet`.
5. `python src/m4_estimation.py` — OLS + QR(τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}) on training. Writes `data/m4_results.parquet`.
6. `python src/m5_walkforward.py` — expanding-window OOS sign-consistency check, 2018–2025. Writes `data/m5_oos_results.parquet`.
7. `python src/m6_bootstrap.py` — stationary block bootstrap CI + placebo (~8 min). Writes `data/m6_bootstrap.parquet`.

`src/_fetch_ticker_types.py` and `src/_fetch_ticker_adr_fields.py` are one-off helpers that populated `data/yfinance_types.parquet` (the security-type filter cache). They re-fetch from yfinance and should not need to run again.
