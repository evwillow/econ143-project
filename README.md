# EC143 Final Project

Pre-registered quantile-regression test of whether Qullamaggie-style volume contraction during a stock's consolidation base raises the upper tail of 20-day forward returns more than it raises the median, after controlling for FF3+UMD factor returns, sector fixed effects, and year fixed effects.

**Deliverable:** [`writeup.ipynb`](writeup.ipynb) at the repo root. Open it and run all cells — every figure, table, and number regenerates from the four committed parquets in `data/`.

**Pre-registered hypothesis:** `writeup.ipynb` §6 (locked 2026-05-03, before any QR fits ran).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate     # Windows; on POSIX use `source .venv/bin/activate`
pip install -r requirements.txt
```

Python ≥ 3.11 required.

## Data

Only four parquets are committed under `data/`, one per analysis-stage output:

- `m3_setups_with_residuals.parquet` — main analysis panel (5,286 setup-rows × M1+M2+M3 columns)
- `m4_results.parquet` — OLS + QR coefficient table on the training window
- `m5_oos_results.parquet` — expanding-window walk-forward summary
- `m6_bootstrap.parquet` — stationary block bootstrap draws (real + placebo)

`data/raw/` is gitignored — it sources Polygon daily bars from a separate `breakoutStudyTool` pipeline and is not redistributable. Intermediate stage parquets, the FF3+UMD factor panel, the SPY cache, and the yfinance ticker-type cache are also omitted from the committed tree; they regenerate from upstream sources on a full pipeline re-run (M3/M2 download Ken French and SPY automatically; M1's yfinance ticker-type cache is repopulated by `src/_fetch_ticker_types.py` and `src/_fetch_ticker_adr_fields.py`).

## Reproduction

To verify only the writeup: open `writeup.ipynb` and `Run All`. It loads the four committed parquets and regenerates every output inline.

To re-run the full pipeline from scratch: execute the modules in order. Each writes its outputs to `data/` and emits a per-stage validation report to `reports/` as a side effect (`reports/` is gitignored).

1. `python src/m0_audit.py` — survivorship + bad-bar audit. Writes `data/m0_audit_summary.json`.
2. `python src/m1_universe.py` — Qullamaggie consolidation-breakout detector. Writes `data/m1_setups.parquet`.
3. `python src/m2_features.py` — per-setup features (vol contraction, ADR, RS-vs-SPY, sector). Writes `data/m2_setups_with_features.parquet` and (if absent) caches `data/spy_daily.parquet`.
4. `python src/m3_factors.py` — FF3+UMD factor panel + 20-day-forward residualization on 2010–2017 training. Writes `data/ff3_umd_daily.parquet`, `data/m3_setups_with_residuals.parquet`.
5. `python src/m4_estimation.py` — OLS + QR(τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90}) on training. Writes `data/m4_results.parquet`.
6. `python src/m5_walkforward.py` — expanding-window OOS sign-consistency check, 2018–2025. Writes `data/m5_oos_results.parquet`.
7. `python src/m6_bootstrap.py` — stationary block bootstrap CI + placebo (~8 min). Writes `data/m6_bootstrap.parquet`.
