"""
M4 — OLS + Quantile Regression estimation on the 2010-2017 training window.

Inputs
------
data/m3_setups_with_residuals.parquet  (M3 output)

Outputs
-------
data/m4_results.parquet  — long-form coefficient table
reports/m4_estimation.md         — readable validation report

Design (per task brief; pre-registered hypothesis in writeup.md §6):
  - LHS:   fwd_ret_20d_resid (factor + sector + year FE already absorbed in M3)
  - RHS:   vol_contraction_ratio_w + adr_pct + base_duration_days + rs_slope_vs_spy
  - Panels: loose (headline) and strict (§7.1 robustness)
  - Specs:  primary  -> uses vol_contraction_ratio_w (winsorized at training p99)
            sensitivity -> uses raw vol_contraction_ratio
  - Models: OLS, plus QuantReg at tau in {0.10, 0.25, 0.50, 0.75, 0.90}
  - Headline statistic: beta(tau=0.90) - beta(tau=0.50) on vcr_w, loose panel.
    No bootstrap CI here -- M6 handles inference.
"""
from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
M3_PATH = REPO_ROOT / "data" / "m3_setups_with_residuals.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "m4_results.parquet"
REPORT_MD = REPO_ROOT / "reports" / "m4_estimation.md"

TRAIN_START = _date(2010, 1, 1)
TRAIN_END   = _date(2017, 12, 31)

TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)

PANEL_VARIANTS = ("loose", "strict")  # loose is headline, strict is §7.1
SPECS = {
    "primary":     ["vol_contraction_ratio_w", "adr_pct", "base_duration_days", "rs_slope_vs_spy"],
    "sensitivity": ["vol_contraction_ratio",   "adr_pct", "base_duration_days", "rs_slope_vs_spy"],
}

# Pretty-print column ordering for the per-panel × spec coefficient tables in
# the markdown report. const always first.
TERM_DISPLAY_ORDER = {
    "primary":     ["const", "vol_contraction_ratio_w", "adr_pct", "base_duration_days", "rs_slope_vs_spy"],
    "sensitivity": ["const", "vol_contraction_ratio",   "adr_pct", "base_duration_days", "rs_slope_vs_spy"],
}

T_FLAG_THRESHOLD = 3.0  # |t| > 3 in OLS gets flagged


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_one(panel: str, spec_name: str, df: pl.DataFrame) -> tuple[list, dict]:
    """Fit OLS + QR on (panel, spec). Return (rows_for_parquet, qr_betas_by_tau)."""
    rhs_cols = SPECS[spec_name]
    X = df.select(rhs_cols).to_numpy().astype(np.float64)
    y = df["fwd_ret_20d_resid"].to_numpy().astype(np.float64)
    Xc = sm.add_constant(X, has_constant="add")
    n = X.shape[0]

    rows: list[dict] = []

    # ---- OLS ----
    ols = sm.OLS(y, Xc).fit()
    for term, coef, se, t, p in zip(
        ["const"] + rhs_cols, ols.params, ols.bse, ols.tvalues, ols.pvalues
    ):
        rows.append({
            "panel": panel, "spec": spec_name, "model": "OLS", "tau": None,
            "term": term,
            "coef": float(coef), "std_err": float(se),
            "t_or_z": float(t), "p": float(p),
            "rsquared": float(ols.rsquared),
            "pseudo_r2": None,
            "n": int(n),
        })

    # ---- QR at each tau ----
    qr_betas_by_tau: dict[float, dict[str, float]] = {}
    for tau in TAUS:
        # statsmodels QuantReg uses an iterative WLS solver (Koenker-Hallock
        # default). Deterministic given fixed input + tau.
        qr = sm.QuantReg(y, Xc).fit(q=tau)
        coefs_for_tau = {}
        for term, coef, se, t, p in zip(
            ["const"] + rhs_cols, qr.params, qr.bse, qr.tvalues, qr.pvalues
        ):
            rows.append({
                "panel": panel, "spec": spec_name, "model": "QR", "tau": float(tau),
                "term": term,
                "coef": float(coef), "std_err": float(se),
                "t_or_z": float(t), "p": float(p),
                "rsquared": None,
                "pseudo_r2": float(qr.prsquared),
                "n": int(n),
            })
            coefs_for_tau[term] = float(coef)
        qr_betas_by_tau[float(tau)] = coefs_for_tau

    return rows, qr_betas_by_tau


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _flag_t_stat(rows_for_parquet: list[dict]) -> list[dict]:
    """Return list of OLS coefficients with |t| > T_FLAG_THRESHOLD on the
    primary spec, excluding the constant."""
    flags = []
    for r in rows_for_parquet:
        if r["model"] != "OLS" or r["spec"] != "primary" or r["term"] == "const":
            continue
        if abs(r["t_or_z"]) > T_FLAG_THRESHOLD:
            flags.append(r)
    return flags


def _format_coef_cell(coef: float, t: float) -> str:
    """Markdown cell: 'coef (t-stat)', with t-stat in italics. NaN-safe."""
    if not (np.isfinite(coef) and np.isfinite(t)):
        return "—"
    return f"{coef:+.4f} (_{t:+.2f}_)"


def _coef_table(
    rows: list[dict],
    panel: str,
    spec: str,
) -> list[str]:
    """Return markdown lines for a single panel × spec coefficient table."""
    terms = TERM_DISPLAY_ORDER[spec]
    lookup: dict[tuple[str, float | None, str], dict] = {}
    for r in rows:
        if r["panel"] != panel or r["spec"] != spec:
            continue
        lookup[(r["model"], r["tau"], r["term"])] = r

    header = ["model/τ"] + terms + ["n", "R² / pseudo-R²"]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    # OLS row
    cells = ["OLS"]
    n_val = "—"
    r2_val = "—"
    for term in terms:
        r = lookup.get(("OLS", None, term))
        if r is None:
            cells.append("—")
        else:
            cells.append(_format_coef_cell(r["coef"], r["t_or_z"]))
            n_val = f"{r['n']:,}"
            if np.isfinite(r["rsquared"]):
                r2_val = f"{r['rsquared']:.4f}"
    cells.append(n_val)
    cells.append(r2_val)
    lines.append("| " + " | ".join(cells) + " |")

    # QR rows (one per tau)
    for tau in TAUS:
        cells = [f"QR τ={tau:.2f}"]
        n_val = "—"
        r2_val = "—"
        for term in terms:
            r = lookup.get(("QR", float(tau), term))
            if r is None:
                cells.append("—")
            else:
                cells.append(_format_coef_cell(r["coef"], r["t_or_z"]))
                n_val = f"{r['n']:,}"
                if r["pseudo_r2"] is not None and np.isfinite(r["pseudo_r2"]):
                    r2_val = f"{r['pseudo_r2']:.4f}"
        cells.append(n_val)
        cells.append(r2_val)
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def _interpret(headline: dict, panel: str) -> str:
    """One short paragraph per panel summarizing the QR sweep on vcr_w."""
    h = headline.get(panel)
    if h is None:
        return ""
    b50 = h["beta_50"]
    b90 = h["beta_90"]
    diff = h["diff"]
    diff_sign = "negative" if diff < 0 else ("positive" if diff > 0 else "zero")
    direction = "more negative" if (b90 < b50) else "less negative / more positive"
    return (
        f"On the **{panel}** panel, the slope of `vol_contraction_ratio_w` is "
        f"{b50:+.4f} at the median (τ=0.50) and {b90:+.4f} at the upper tail "
        f"(τ=0.90). The headline statistic β(0.90) − β(0.50) = "
        f"**{diff:+.4f}** is {diff_sign}. The point estimate is "
        f"{direction} at τ=0.90 vs τ=0.50; this is the test-statistic the "
        "M6 stationary block bootstrap will assign a CI to. A negative diff "
        "is the predicted direction (more contraction → stronger upside-tail "
        "effect than median-effect), since lower vcr_w = more contraction = "
        "stronger Qullamaggie signal."
    )


def _write_report(
    rows_for_parquet: list[dict],
    headline: dict[str, dict],
    panel_sizes: dict[str, int],
    flags: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# M4 — OLS + Quantile Regression Estimation (training 2010-2017)")
    lines.append("")
    lines.append(
        "Pre-registered model (writeup §6): "
        "`fwd_ret_20d_resid ~ vol_contraction_ratio_w + adr_pct + "
        "base_duration_days + rs_slope_vs_spy`. Factor + sector + year FE "
        "already absorbed by the M3 residualization, so this stage adds **no** "
        "additional controls. OLS is reported alongside QuantReg at τ ∈ "
        "{0.10, 0.25, 0.50, 0.75, 0.90}. The pre-registered test statistic "
        "is **β(0.90) − β(0.50)** on `vol_contraction_ratio_w`, **headline "
        "panel = loose**. Bootstrap inference is M6's job — this report is "
        "point estimates only."
    )
    lines.append("")

    # ---- Sample sizes ----
    lines.append("## Sample sizes (training window, 2010-2017)")
    lines.append("")
    lines.append("| Panel | n |")
    lines.append("|---|---:|")
    for p in PANEL_VARIANTS:
        lines.append(f"| {p} | {panel_sizes[p]:,} |")
    lines.append("")

    # ---- Headline statistic ----
    lines.append("## Headline statistic — β(0.90) − β(0.50) on `vol_contraction_ratio_w`")
    lines.append("")
    lines.append("| Panel | β(τ=0.50) | β(τ=0.90) | β(0.90) − β(0.50) |")
    lines.append("|---|---:|---:|---:|")
    for p in PANEL_VARIANTS:
        h = headline.get(p)
        if h is None:
            lines.append(f"| {p} | — | — | — |")
            continue
        bold_diff = f"**{h['diff']:+.4f}**" if p == "loose" else f"{h['diff']:+.4f}"
        lines.append(
            f"| {p}{' (headline)' if p == 'loose' else ''} | "
            f"{h['beta_50']:+.4f} | {h['beta_90']:+.4f} | {bold_diff} |"
        )
    lines.append("")
    lines.append(
        "Sign convention: `vol_contraction_ratio` = mean(volume, second half) "
        "/ mean(volume, first half). Lower = more contraction = stronger "
        "Qullamaggie signal. The pre-registered prediction is that contraction "
        "lifts the **upside tail** of forward returns more than it shifts the "
        "median, i.e. a **more-negative β at τ=0.90 than at τ=0.50** — so the "
        "predicted sign of β(0.90) − β(0.50) is **negative**. Inference "
        "(stationary block bootstrap, mean block length 30) is M6."
    )
    lines.append("")

    # ---- Per-panel × spec tables ----
    for panel in PANEL_VARIANTS:
        lines.append(f"## {panel.upper()} panel")
        lines.append("")
        for spec in ("primary", "sensitivity"):
            spec_label = "primary (winsorized vcr)" if spec == "primary" else "sensitivity (raw vcr)"
            lines.append(f"### {spec_label}")
            lines.append("")
            lines.append(
                "Cells show **β (t-stat)**. R² / pseudo-R² is the OLS R² for "
                "the OLS row and Koenker-Machado pseudo-R² for each QR row."
            )
            lines.append("")
            lines.extend(_coef_table(rows_for_parquet, panel, spec))
            lines.append("")
        # Per-panel interpretation
        lines.append(f"### Interpretation — {panel} panel")
        lines.append("")
        lines.append(_interpret(headline, panel))
        lines.append("")

    # ---- t-stat flags ----
    lines.append(f"## OLS coefficients with |t| > {T_FLAG_THRESHOLD:g} (primary spec, excluding intercept)")
    lines.append("")
    if not flags:
        lines.append(f"- None. No covariate exceeded the |t| > {T_FLAG_THRESHOLD:g} flag threshold "
                     "in either panel's OLS primary spec.")
    else:
        lines.append("| Panel | Term | β | std err | t | p |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for f in flags:
            lines.append(
                f"| {f['panel']} | `{f['term']}` | {f['coef']:+.4f} | "
                f"{f['std_err']:.4f} | {f['t_or_z']:+.2f} | {f['p']:.3g} |"
            )
    lines.append("")

    # ---- Notes ----
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Why vcr_w not raw vcr in the headline?** M2 flagged vol_contraction_ratio "
        "max=15.39 vs median=0.94 (heavy right tail from low first-half "
        "volume). Winsorizing at training p99 (2.85) keeps a small number of "
        "extreme rows from dominating the QR fit. Raw-vcr results are "
        "reported as a sensitivity in the same report."
    )
    lines.append(
        "- **Why no factor / sector / year controls in the regressor list?** "
        "They were already partialled out in M3 by residualizing fwd_ret_20d "
        "against FF3+UMD (cumulated over the 20-day forward window) plus "
        "sector and year fixed effects on the training window. The LHS here "
        "is `fwd_ret_20d_resid`, so adding these controls again would be "
        "double-counting."
    )
    lines.append(
        "- **t-stats on QR rows** are the asymptotic z-statistics from "
        "statsmodels' default kernel/IID covariance. They are useful as a "
        "rough guide but the pre-registered inference uses a stationary "
        "block bootstrap (M6), not these t-stats."
    )
    lines.append(
        "- **Determinism**: `sm.OLS(...).fit()` and `sm.QuantReg(...).fit(q=τ)` "
        "are both deterministic given fixed input and tau. Re-running the "
        "script on the same `data/m3_setups_with_residuals.parquet` "
        "yields byte-identical `data/m4_results.parquet`."
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

    print(f"[M4] reading {M3_PATH}")
    df_all = pl.read_parquet(M3_PATH)

    train_filter = (pl.col("date") >= TRAIN_START) & (pl.col("date") <= TRAIN_END)

    panel_sizes: dict[str, int] = {}
    rows_for_parquet: list[dict] = []
    headline: dict[str, dict] = {}

    for panel in PANEL_VARIANTS:
        df_panel = df_all.filter(train_filter & (pl.col("universe_variant") == panel))
        # Drop any rows missing residual or RHS (defensive; M3 should not have them)
        needed = ["fwd_ret_20d_resid"] + sorted(set(c for cs in SPECS.values() for c in cs))
        df_panel = df_panel.drop_nulls(subset=needed)
        n = df_panel.height
        panel_sizes[panel] = n
        print(f"[M4] panel='{panel}': n={n:,}")

        for spec in SPECS:
            print(f"[M4]   spec='{spec}': fitting OLS + QR(τ ∈ {list(TAUS)})")
            rows, qr_betas = _fit_one(panel, spec, df_panel)
            rows_for_parquet.extend(rows)

            # Capture the headline statistic from the primary spec only.
            if spec == "primary":
                vcr_term = "vol_contraction_ratio_w"
                b50 = qr_betas[0.50][vcr_term]
                b90 = qr_betas[0.90][vcr_term]
                headline[panel] = {
                    "beta_50": b50,
                    "beta_90": b90,
                    "diff": b90 - b50,
                }

    h_loose = headline.get("loose")
    if h_loose is not None:
        print(
            f"[M4] HEADLINE (loose): β(0.50)={h_loose['beta_50']:+.4f}, "
            f"β(0.90)={h_loose['beta_90']:+.4f}, "
            f"diff={h_loose['diff']:+.4f}"
        )
    h_strict = headline.get("strict")
    if h_strict is not None:
        print(
            f"[M4] strict (§7.1): β(0.50)={h_strict['beta_50']:+.4f}, "
            f"β(0.90)={h_strict['beta_90']:+.4f}, "
            f"diff={h_strict['diff']:+.4f}"
        )

    flags = _flag_t_stat(rows_for_parquet)
    if flags:
        print(f"[M4] {len(flags)} OLS-primary coefficient(s) with |t| > {T_FLAG_THRESHOLD}:")
        for f in flags:
            print(f"[M4]   {f['panel']} {f['term']}: β={f['coef']:+.4f}, t={f['t_or_z']:+.2f}")
    else:
        print(f"[M4] no OLS-primary coefficients with |t| > {T_FLAG_THRESHOLD}")

    out_df = pl.DataFrame(rows_for_parquet)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(OUT_PARQUET)
    print(f"[M4] wrote {OUT_PARQUET} ({out_df.height} rows)")

    _write_report(rows_for_parquet, headline, panel_sizes, flags)
    print(f"[M4] wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
