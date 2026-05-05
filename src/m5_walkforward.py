"""
M5 — walk-forward (expanding window) parameter-stability check on the
2018-2025 OOS years.

For each OOS year y:
  - Training window:  2010-01-01 -> (y-1)-12-31
  - "Score" year:     y-01-01    -> y-12-31  (informational n_score only)
  - Re-fit M3 factor + sector + year-FE residualization on the expanded
    training window.
  - Re-fit QR(0.50) and QR(0.90) on the training residuals using the same
    M4 specification:
        fwd_ret_20d_resid ~ vol_contraction_ratio_w
                          + adr_pct + base_duration_days + rs_slope_vs_spy
  - Record: n_train, n_score, β_train(0.50), β_train(0.90),
            β_train(0.90) - β_train(0.50)

This is a parameter-stability check (per task brief), not OOS prediction:
the recorded β's come from the *training* slice that was just expanded by
adding year y-1's data. We're answering "does the M4 headline statistic
remain in the predicted direction as more training data is added?", which
is the §6 prereg's "≥ 5 of 8 expanding-window OOS years" requirement.

Inputs
------
data/m2_setups_with_features.parquet
data/ff3_umd_daily.parquet
breakoutStudyTool/.../daily/*.parquet  (for fwd-return close lookup)

Outputs
-------
data/m5_oos_results.parquet
reports/m5_walkforward.md
"""
from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_factors  # noqa: E402   (re-uses _load_ticker_closes, _compute_forward_returns,
                   #               _attach_factor_window_sums, residualize)

REPO_ROOT = Path(__file__).resolve().parents[1]
M2_PATH = REPO_ROOT / "data" / "m2_setups_with_features.parquet"
FF_PATH = REPO_ROOT / "data" / "ff3_umd_daily.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "m5_oos_results.parquet"
REPORT_MD   = REPO_ROOT / "reports" / "m5_walkforward.md"

TRAIN_START = _date(2010, 1, 1)
OOS_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
PANELS = ("loose", "strict")  # loose is headline; strict is §7.1
RHS_COLS = ["vol_contraction_ratio_w", "adr_pct", "base_duration_days", "rs_slope_vs_spy"]
TAUS = (0.50, 0.90)
PREREG_PASS_THRESHOLD = 5  # §6: sign holds in >= 5 of 8 windows
FWD_HORIZON = 20

# In the design matrix the constant takes column 0 and the four RHS terms
# follow in RHS_COLS order. vcr_w is at index 1.
VCR_IDX = 1


# ---------------------------------------------------------------------------
# Per-year fit
# ---------------------------------------------------------------------------

def _fit_year(
    setups_with_factors: pl.DataFrame,
    train_start: _date,
    train_end: _date,
    score_year: int,
) -> list[dict]:
    """Refit residualization for [train_start, train_end], then per panel
    fit QR(0.50)/QR(0.90) and record β diff. Returns one row per panel."""
    fit_df, fit_info = m3_factors.residualize(setups_with_factors, train_start, train_end)

    train_mask = (fit_df["date"] >= train_start) & (fit_df["date"] <= train_end)
    score_mask = (
        (fit_df["date"] >= _date(score_year, 1, 1))
        & (fit_df["date"] <= _date(score_year, 12, 31))
    )

    rows: list[dict] = []
    for panel in PANELS:
        train_panel = fit_df.filter(train_mask & (pl.col("universe_variant") == panel))
        score_panel = fit_df.filter(score_mask & (pl.col("universe_variant") == panel))

        n_train = train_panel.height
        n_score = score_panel.height

        # Defensive: drop rows with any null in the RHS or LHS (shouldn't
        # happen given M2 has 0 NaN on these features and residualize() drops
        # the few rows missing fwd-return / factor windows).
        needed = ["fwd_ret_20d_resid"] + RHS_COLS
        train_panel = train_panel.drop_nulls(subset=needed)
        if train_panel.height < n_train:
            print(f"[M5]   {panel} {score_year}: dropped "
                  f"{n_train - train_panel.height} rows for null in {needed}",
                  flush=True)
            n_train = train_panel.height

        if n_train < 50:
            print(f"[M5]   WARN: {panel} {score_year} has only n_train={n_train}; skipping QR fit")
            rows.append({
                "year": score_year, "panel": panel,
                "train_start": train_start, "train_end": train_end,
                "n_train": n_train, "n_score": n_score,
                "beta_50": float("nan"), "beta_90": float("nan"),
                "diff": float("nan"),
                "sign_diff": "n/a",
                "p99_vcr_train": fit_info["p99_vcr_train"],
                "ols_rsquared": fit_info["rsquared"],
                "fit_succeeded": False,
            })
            continue

        X = train_panel.select(RHS_COLS).to_numpy().astype(np.float64)
        y = train_panel["fwd_ret_20d_resid"].to_numpy().astype(np.float64)
        Xc = sm.add_constant(X, has_constant="add")

        qr50 = sm.QuantReg(y, Xc).fit(q=0.50)
        qr90 = sm.QuantReg(y, Xc).fit(q=0.90)

        b50 = float(qr50.params[VCR_IDX])
        b90 = float(qr90.params[VCR_IDX])
        diff = b90 - b50
        sign = "negative" if diff < 0 else ("positive" if diff > 0 else "zero")

        rows.append({
            "year": score_year,
            "panel": panel,
            "train_start": train_start,
            "train_end": train_end,
            "n_train": n_train,
            "n_score": n_score,
            "beta_50": b50,
            "beta_90": b90,
            "diff": diff,
            "sign_diff": sign,
            "p99_vcr_train": fit_info["p99_vcr_train"],
            "ols_rsquared": fit_info["rsquared"],
            "fit_succeeded": True,
        })

    return rows


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _write_report(rows: list[dict]) -> None:
    panels_results: dict[str, list[dict]] = {p: [] for p in PANELS}
    for r in rows:
        panels_results[r["panel"]].append(r)
    for p in PANELS:
        panels_results[p].sort(key=lambda d: d["year"])

    lines: list[str] = []
    lines.append("# M5 — Walk-Forward Expanding-Window OOS Parameter Stability (2018-2025)")
    lines.append("")
    lines.append(
        "For each y ∈ {2018, …, 2025}: refit M3 factor + sector + year-FE "
        "residualization on the **expanded training window** "
        f"[{TRAIN_START}, (y-1)-12-31], then fit QR(0.50) and QR(0.90) on "
        "the training residuals using the M4 spec: `fwd_ret_20d_resid ~ "
        "vol_contraction_ratio_w + adr_pct + base_duration_days + "
        "rs_slope_vs_spy`. The recorded β's are **training-window** "
        "estimates; the score year y is reported (`n_score`) for context "
        "but its data does not enter the regression. This is a parameter-"
        "stability check per the §6 prereg's \"≥ 5 of 8 expanding-window "
        "OOS years\" requirement, not OOS prediction performance."
    )
    lines.append("")

    # --- Per-panel year tables ---
    for panel in PANELS:
        lines.append(f"## {panel.upper()} panel")
        if panel == "loose":
            lines.append("")
            lines.append("(headline)")
        else:
            lines.append("")
            lines.append("(§7.1 robustness)")
        lines.append("")
        lines.append(
            "| Year | Train end | n_train | n_score | β(τ=0.50) | β(τ=0.90) | β(0.90)-β(0.50) | Sign |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
        for r in panels_results[panel]:
            if not r["fit_succeeded"]:
                lines.append(
                    f"| {r['year']} | {r['train_end']} | {r['n_train']:,} | "
                    f"{r['n_score']:,} | — | — | — | n/a |"
                )
                continue
            sign_emoji = "−" if r["diff"] < 0 else ("+" if r["diff"] > 0 else "0")
            lines.append(
                f"| {r['year']} | {r['train_end']} | {r['n_train']:,} | "
                f"{r['n_score']:,} | {r['beta_50']:+.4f} | {r['beta_90']:+.4f} | "
                f"{r['diff']:+.4f} | {sign_emoji} |"
            )
        lines.append("")

        # Sign-consistency
        diffs = [r["diff"] for r in panels_results[panel] if r["fit_succeeded"]]
        n_neg = sum(1 for d in diffs if d < 0)
        n_pos = sum(1 for d in diffs if d > 0)
        n_total = len(diffs)
        verdict = ("PASS" if n_neg >= PREREG_PASS_THRESHOLD else "FAIL") if n_total >= PREREG_PASS_THRESHOLD else "INCOMPLETE"
        lines.append(
            f"**Sign-consistency**: {n_neg} of {n_total} years had β(0.90)−β(0.50) "
            f"< 0 (predicted direction). Pre-registered threshold is "
            f"≥ {PREREG_PASS_THRESHOLD} of 8 → **{verdict}** "
            f"({'headline' if panel == 'loose' else 'robustness'})."
        )
        lines.append("")
        if n_pos:
            pos_years = [r["year"] for r in panels_results[panel] if r["fit_succeeded"] and r["diff"] > 0]
            lines.append(f"Positive-sign years (against the predicted direction): {pos_years}.")
            lines.append("")

        # Magnitude trend
        if n_total >= 2:
            first_diff = diffs[0]
            last_diff  = diffs[-1]
            mag_first = abs(first_diff)
            mag_last  = abs(last_diff)
            mag_mean  = float(np.mean([abs(d) for d in diffs]))
            mag_std   = float(np.std([abs(d) for d in diffs], ddof=1))
            trend = "growing" if mag_last > mag_first else ("shrinking" if mag_last < mag_first else "flat")
            lines.append(
                f"**Magnitude stability** (|β(0.90)−β(0.50)|): first ({panels_results[panel][0]['year']}) "
                f"= {mag_first:.4f}, last ({panels_results[panel][-1]['year']}) = {mag_last:.4f}, "
                f"trend = **{trend}**, mean = {mag_mean:.4f}, std = {mag_std:.4f}."
            )
            lines.append("")

    # --- Overall verdict ---
    lines.append("## Pre-registration verdict")
    lines.append("")
    lines.append(
        "From writeup §6: \"the same sign holds in at least five of eight "
        "expanding-window OOS years (2018-2025).\" The headline panel for "
        "the prereg is **loose**."
    )
    lines.append("")
    h_diffs = [r["diff"] for r in panels_results["loose"] if r["fit_succeeded"]]
    h_neg = sum(1 for d in h_diffs if d < 0)
    h_total = len(h_diffs)
    if h_total >= PREREG_PASS_THRESHOLD:
        lines.append(
            f"- **Loose (headline)**: {h_neg} / {h_total} negative — "
            f"**{'PASS' if h_neg >= PREREG_PASS_THRESHOLD else 'FAIL'}** vs "
            f"the ≥{PREREG_PASS_THRESHOLD}/8 bar."
        )
    s_diffs = [r["diff"] for r in panels_results["strict"] if r["fit_succeeded"]]
    s_neg = sum(1 for d in s_diffs if d < 0)
    s_total = len(s_diffs)
    if s_total >= PREREG_PASS_THRESHOLD:
        lines.append(
            f"- **Strict (§7.1)**: {s_neg} / {s_total} negative — "
            f"**{'PASS' if s_neg >= PREREG_PASS_THRESHOLD else 'FAIL'}** vs "
            f"the ≥{PREREG_PASS_THRESHOLD}/8 bar."
        )
    lines.append("")
    lines.append(
        "The prereg's bootstrap CI inference (M6) is independent of this "
        "stability check — the two requirements (CI excludes zero with "
        "predicted sign **and** sign-consistency in ≥5 of 8 years) are both "
        "needed for the hypothesis to survive."
    )
    lines.append("")

    # --- Notes ---
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **What \"OOS\" means here**: each year y *adds* the prior year "
        "(y-1) to the training set, then we recompute the M4 headline "
        "statistic on that newly-expanded training set. The score year y "
        "itself is recorded as `n_score` for context only — its rows do "
        "not enter any regression. The check passes if the predicted sign "
        "is robust to where you cut the training data."
    )
    lines.append(
        "- **Determinism**: each year's `m3_factors.residualize` and "
        "`statsmodels.QuantReg.fit(q=τ)` are deterministic given fixed "
        "input data and τ. Re-running this script on the same M2 + "
        "factor-panel inputs yields byte-identical "
        "`m5_oos_results.parquet`."
    )
    lines.append(
        "- **Re-fit per year**: the factor-OLS, the sector/year-FE design "
        "matrix, the p99 winsor cap on `vol_contraction_ratio`, **and** "
        "the QR fits are all redone for every y. This means each year's "
        "`vol_contraction_ratio_w` cap is the p99 of THAT year's training "
        "set, which can drift slightly as the expanding window adds more "
        "post-2017 outliers."
    )
    lines.append(
        "- **No bootstrap inference**: this report is point estimates and "
        "sign-consistency only. CIs come in M6."
    )
    lines.append("")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"[M5] reading {M2_PATH}")
    setups = pl.read_parquet(M2_PATH)
    print(f"[M5] {setups.height:,} setups loaded "
          f"({int((setups['universe_variant']=='strict').sum()):,} strict, "
          f"{int((setups['universe_variant']=='loose').sum()):,} loose)")

    print(f"[M5] reading factor panel: {FF_PATH}")
    ff = pl.read_parquet(FF_PATH)

    # ---- Compute fwd_ret_20d and factor windows ONCE; both are
    # training-window-independent. ----
    unique_tickers = sorted(setups["ticker"].unique().to_list())
    ticker_data = m3_factors._load_ticker_closes(unique_tickers)
    print(f"[M5] computing fwd_ret_{FWD_HORIZON}d...", flush=True)
    setups, drops = m3_factors._compute_forward_returns(setups, ticker_data, FWD_HORIZON)
    print(f"[M5] fwd return drops: {len(drops):,}")
    setups, n_missing_factors = m3_factors._attach_factor_window_sums(setups, ff)
    if n_missing_factors:
        print(f"[M5] WARN: {n_missing_factors:,} rows have NaN factor-window sums")

    # ---- Walk-forward across OOS years ----
    all_rows: list[dict] = []
    for y in OOS_YEARS:
        train_end = _date(y - 1, 12, 31)
        print(f"[M5] year={y}: train [{TRAIN_START}, {train_end}]", flush=True)
        rows = _fit_year(setups, TRAIN_START, train_end, y)
        for r in rows:
            print(
                f"[M5]   {r['panel']}: n_train={r['n_train']:,}  "
                f"β50={r['beta_50']:+.4f}  β90={r['beta_90']:+.4f}  "
                f"diff={r['diff']:+.4f}  sign={r['sign_diff']}"
            )
        all_rows.extend(rows)

    # Sign-consistency summary
    for panel in PANELS:
        diffs = [r["diff"] for r in all_rows
                 if r["panel"] == panel and r["fit_succeeded"]]
        n_neg = sum(1 for d in diffs if d < 0)
        n_total = len(diffs)
        verdict = ("PASS" if n_neg >= PREREG_PASS_THRESHOLD else "FAIL")
        print(f"[M5] {panel}: {n_neg}/{n_total} years with negative diff "
              f"(predicted direction). >= {PREREG_PASS_THRESHOLD}/8 → {verdict}")

    # ---- Output ----
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df = pl.DataFrame(all_rows).with_columns(
        pl.col("train_start").cast(pl.Date),
        pl.col("train_end").cast(pl.Date),
    )
    out_df.write_parquet(OUT_PARQUET)
    print(f"[M5] wrote {OUT_PARQUET}")

    _write_report(all_rows)
    print(f"[M5] wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
