"""
M7 — Generate writeup data assets (tables + figures only). NO interpretation.

Outputs (under reports/writeup_assets/):
  tables/
    m4_coef_table.md          — OLS + QR(τ) coefficient tables, both panels (primary spec)
    m5_oos_yearly.md          — year × {n_train, β(0.50), β(0.90), diff, sign}, both panels
    m6_inference.md           — bootstrap inference summary, three panels
  figures/
    m5_magnitude_trajectory.png  — |β(0.90)−β(0.50)| vs OOS year, both panels
    m6_bootstrap_dist_loose.png  — canonical-loose real + placebo histograms
    m4_qr_sweep_adr.png          — adr_pct β vs τ for loose panel, with ±1.96·SE bands

Inputs:
  data/interim/m4_results.parquet
  data/interim/m5_oos_results.parquet
  data/interim/m6_bootstrap.parquet

Numeric formatting:
  4 decimal places for coefficients, CIs, and p-values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
M4_PATH = REPO_ROOT / "data" / "interim" / "m4_results.parquet"
M5_PATH = REPO_ROOT / "data" / "interim" / "m5_oos_results.parquet"
M6_PATH = REPO_ROOT / "data" / "interim" / "m6_bootstrap.parquet"

ASSETS_DIR = REPO_ROOT / "reports" / "writeup_assets"
TABLES_DIR = ASSETS_DIR / "tables"
FIGS_DIR = ASSETS_DIR / "figures"

PRIMARY_TERMS = [
    "const",
    "vol_contraction_ratio_w",
    "adr_pct",
    "base_duration_days",
    "rs_slope_vs_spy",
]
TAU_LIST = (0.10, 0.25, 0.50, 0.75, 0.90)
PANELS = ("loose", "strict")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fmt(x, p=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{p}f}"


def _fmt_unsigned(x, p=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{p}f}"


def _coef_cell(coef, t):
    if coef is None or t is None:
        return "—"
    if not (np.isfinite(coef) and np.isfinite(t)):
        return "—"
    return f"{coef:+.4f} ({t:+.2f})"


# ---------------------------------------------------------------------------
# Table 1 — M4 coefficient tables (primary spec, both panels)
# ---------------------------------------------------------------------------

def _build_m4_panel_table(m4: pl.DataFrame, panel: str) -> list[str]:
    df = m4.filter((pl.col("panel") == panel) & (pl.col("spec") == "primary"))
    lookup: dict[tuple[str, float | None, str], dict] = {}
    for r in df.iter_rows(named=True):
        lookup[(r["model"], r["tau"], r["term"])] = r

    header = ["model/τ"] + PRIMARY_TERMS + ["n", "R² / pseudo-R²"]
    out: list[str] = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")

    def emit_row(model_label: str, model_key: str, tau_key):
        cells = [model_label]
        n_val = "—"
        r2_val = "—"
        for term in PRIMARY_TERMS:
            r = lookup.get((model_key, tau_key, term))
            if r is None:
                cells.append("—")
            else:
                cells.append(_coef_cell(r["coef"], r["t_or_z"]))
                n_val = f"{int(r['n']):,}"
                if model_key == "OLS":
                    r2_val = _fmt_unsigned(r["rsquared"], 4)
                else:
                    r2_val = _fmt_unsigned(r["pseudo_r2"], 4)
        cells.append(n_val)
        cells.append(r2_val)
        out.append("| " + " | ".join(cells) + " |")

    emit_row("OLS", "OLS", None)
    for tau in TAU_LIST:
        emit_row(f"QR τ={tau:.2f}", "QR", float(tau))
    return out


def write_m4_table(m4: pl.DataFrame) -> None:
    lines: list[str] = []
    for i, panel in enumerate(PANELS):
        if i > 0:
            lines.append("")
        lines.append(f"### {panel.capitalize()} panel — primary spec (vol_contraction_ratio_w)")
        lines.append("")
        lines.append("Cells: β (t-stat). Last two columns: sample size and OLS R² / Koenker–Machado pseudo-R².")
        lines.append("")
        lines.extend(_build_m4_panel_table(m4, panel))
    (TABLES_DIR / "m4_coef_table.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Table 2 — M5 year × stats
# ---------------------------------------------------------------------------

def write_m5_table(m5: pl.DataFrame) -> None:
    lines: list[str] = []
    for i, panel in enumerate(PANELS):
        if i > 0:
            lines.append("")
        lines.append(f"### {panel.capitalize()} panel — expanding-window OOS (M5)")
        lines.append("")
        lines.append("| Year | Train end | n_train | β(τ=0.50) | β(τ=0.90) | β(0.90)−β(0.50) | Sign |")
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        sub = m5.filter(pl.col("panel") == panel).sort("year")
        for r in sub.iter_rows(named=True):
            if not r["fit_succeeded"]:
                lines.append(
                    f"| {r['year']} | {r['train_end']} | {r['n_train']:,} | — | — | — | n/a |"
                )
                continue
            sign = "−" if r["diff"] < 0 else ("+" if r["diff"] > 0 else "0")
            lines.append(
                f"| {r['year']} | {r['train_end']} | {r['n_train']:,} | "
                f"{_fmt(r['beta_50'])} | {_fmt(r['beta_90'])} | {_fmt(r['diff'])} | {sign} |"
            )
    (TABLES_DIR / "m5_oos_yearly.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Table 3 — M6 inference summary
# ---------------------------------------------------------------------------

def _m6_panel_stats(m6: pl.DataFrame, label: str, panel: str) -> dict:
    sub = m6.filter((pl.col("label") == label) & (pl.col("panel") == panel))
    point = sub.filter(pl.col("mode") == "point")
    real  = sub.filter(pl.col("mode") == "real")
    placebo = sub.filter(pl.col("mode") == "placebo")

    theta_hat = float(point["diff"][0]) if point.height else float("nan")

    real_arr = real["diff"].to_numpy()
    if real_arr.size:
        p25  = float(np.quantile(real_arr, 0.025))
        p975 = float(np.quantile(real_arr, 0.975))
        # One-sided p̂ for H1: diff < 0
        p_one_sided = float((np.sum(real_arr >= 0) + 1) / (real_arr.size + 1))
    else:
        p25 = p975 = p_one_sided = float("nan")

    placebo_arr = placebo["diff"].to_numpy()
    if placebo_arr.size:
        placebo_mean = float(placebo_arr.mean())
        placebo_std  = float(placebo_arr.std(ddof=1)) if placebo_arr.size > 1 else float("nan")
        # Fraction of placebo draws ≤ θ̂ (i.e. percentile of θ̂ in null)
        if np.isfinite(theta_hat):
            theta_pct_in_null = float((placebo_arr <= theta_hat).mean())
        else:
            theta_pct_in_null = float("nan")
    else:
        placebo_mean = placebo_std = theta_pct_in_null = float("nan")

    return {
        "label": label,
        "panel": panel,
        "n_train": int(point["n_train"][0]) if point.height else 0,
        "theta_hat": theta_hat,
        "ci_perc_lo": p25,
        "ci_perc_hi": p975,
        "ci_basic_lo": 2 * theta_hat - p975 if np.isfinite(p975) else float("nan"),
        "ci_basic_hi": 2 * theta_hat - p25  if np.isfinite(p25)  else float("nan"),
        "p_one_sided": p_one_sided,
        "placebo_mean": placebo_mean,
        "placebo_std":  placebo_std,
        "theta_pct_in_null": theta_pct_in_null,
        "n_real_draws": int(real_arr.size),
        "n_placebo_draws": int(placebo_arr.size),
    }


def write_m6_table(m6: pl.DataFrame) -> None:
    rows = [
        _m6_panel_stats(m6, "canonical",   "loose"),
        _m6_panel_stats(m6, "canonical",   "strict"),
        _m6_panel_stats(m6, "exploratory", "loose"),
    ]
    lines: list[str] = []
    lines.append("### Bootstrap inference summary (M6)")
    lines.append("")
    lines.append(
        "Predicted sign on β(0.90)−β(0.50) is negative; one-sided p̂ = "
        "P̂[diff ≥ 0] under the bootstrap (smaller is stronger evidence "
        "against the null)."
    )
    lines.append("")
    header = [
        "Panel",
        "n",
        "θ̂",
        "Percentile 95% CI",
        "Basic 95% CI",
        "One-sided p̂",
        "Placebo mean",
        "Placebo std",
        "Pct of null ≤ θ̂",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        if r["label"] == "canonical" and r["panel"] == "loose":
            label = "Canonical / loose (HEADLINE)"
        elif r["label"] == "canonical" and r["panel"] == "strict":
            label = "Canonical / strict (§7.1)"
        elif r["label"] == "exploratory":
            label = "Exploratory / loose 2010–2024 (§7)"
        else:
            label = f"{r['label']} / {r['panel']}"
        ci_perc = f"[{_fmt(r['ci_perc_lo'])}, {_fmt(r['ci_perc_hi'])}]"
        ci_basic = f"[{_fmt(r['ci_basic_lo'])}, {_fmt(r['ci_basic_hi'])}]"
        placebo_mean = _fmt(r["placebo_mean"]) if np.isfinite(r["placebo_mean"]) else "—"
        placebo_std  = _fmt_unsigned(r["placebo_std"], 4) if np.isfinite(r["placebo_std"]) else "—"
        pct_null     = _fmt_unsigned(r["theta_pct_in_null"], 4) if np.isfinite(r["theta_pct_in_null"]) else "—"
        lines.append(
            f"| {label} | {r['n_train']:,} | {_fmt(r['theta_hat'])} | {ci_perc} | "
            f"{ci_basic} | {_fmt_unsigned(r['p_one_sided'], 4)} | "
            f"{placebo_mean} | {placebo_std} | {pct_null} |"
        )
    (TABLES_DIR / "m6_inference.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 1 — M5 magnitude trajectory
# ---------------------------------------------------------------------------

def fig_m5_trajectory(m5: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for panel, color in (("loose", "C0"), ("strict", "C1")):
        sub = m5.filter(pl.col("panel") == panel).sort("year")
        years = sub["year"].to_numpy()
        diffs = sub["diff"].to_numpy()
        mags = np.abs(diffs)
        ax.plot(years, mags, marker="o", color=color, label=f"{panel} (|β diff|)", linewidth=2)
        # Mark sign with marker color
        for x, d, m in zip(years, diffs, mags):
            if d >= 0:
                ax.scatter([x], [m], color="white", edgecolor=color,
                           zorder=5, s=70, linewidths=1.6)

    # Mark the exploratory 2024-cutoff point (= year 2025 with train_end 2024)
    loose_2025 = m5.filter((pl.col("panel") == "loose") & (pl.col("year") == 2025))
    if loose_2025.height:
        x = int(loose_2025["year"][0])
        y = abs(float(loose_2025["diff"][0]))
        ax.annotate(
            "exploratory\n(train→2024)",
            xy=(x, y),
            xytext=(x - 1.5, y + 0.004),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("OOS year (training window ends y−1)")
    ax.set_ylabel("|β(τ=0.90) − β(τ=0.50)|")
    ax.set_title("M5 walk-forward magnitude trajectory")
    ax.set_xticks(sorted(set(int(y) for y in m5["year"].to_list())))
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "m5_magnitude_trajectory.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — M6 bootstrap distribution (canonical loose)
# ---------------------------------------------------------------------------

def fig_m6_bootstrap_dist(m6: pl.DataFrame) -> None:
    sub = m6.filter((pl.col("label") == "canonical") & (pl.col("panel") == "loose"))
    point = sub.filter(pl.col("mode") == "point")
    real_arr  = sub.filter(pl.col("mode") == "real")["diff"].to_numpy()
    plac_arr  = sub.filter(pl.col("mode") == "placebo")["diff"].to_numpy()
    theta_hat = float(point["diff"][0]) if point.height else float("nan")
    p25  = float(np.quantile(real_arr, 0.025)) if real_arr.size else float("nan")
    p975 = float(np.quantile(real_arr, 0.975)) if real_arr.size else float("nan")

    lo = float(min(real_arr.min(), plac_arr.min(), theta_hat) if real_arr.size and plac_arr.size else 0.0)
    hi = float(max(real_arr.max(), plac_arr.max(), theta_hat) if real_arr.size and plac_arr.size else 0.0)
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    bins = np.linspace(lo - pad, hi + pad, 50)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(real_arr, bins=bins, density=True, alpha=0.55, color="C0",
            label=f"Real bootstrap (n={real_arr.size:,})", edgecolor="C0")
    ax.hist(plac_arr, bins=bins, density=True, alpha=0.45, color="C2",
            label=f"Placebo (n={plac_arr.size:,})", edgecolor="C2")
    ax.axvline(theta_hat, color="black", linestyle="-", linewidth=1.6,
               label=f"θ̂ = {theta_hat:+.4f}")
    ax.axvline(p25,  color="C0", linestyle="--", linewidth=1.0,
               label=f"95% percentile CI: [{p25:+.4f}, {p975:+.4f}]")
    ax.axvline(p975, color="C0", linestyle="--", linewidth=1.0)
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("β(τ=0.90) − β(τ=0.50)")
    ax.set_ylabel("Density")
    ax.set_title("M6 bootstrap distribution — canonical loose (training 2010–2017)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "m6_bootstrap_dist_loose.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — M4 QR sweep on adr_pct (loose panel)
# ---------------------------------------------------------------------------

def fig_m4_qr_sweep_adr(m4: pl.DataFrame) -> None:
    df = m4.filter(
        (pl.col("panel") == "loose")
        & (pl.col("spec") == "primary")
        & (pl.col("model") == "QR")
        & (pl.col("term") == "adr_pct")
    ).sort("tau")
    taus  = df["tau"].to_numpy()
    coefs = df["coef"].to_numpy()
    ses   = df["std_err"].to_numpy()

    # OLS reference
    ols = m4.filter(
        (pl.col("panel") == "loose")
        & (pl.col("spec") == "primary")
        & (pl.col("model") == "OLS")
        & (pl.col("term") == "adr_pct")
    )
    ols_coef = float(ols["coef"][0]) if ols.height else float("nan")
    ols_se   = float(ols["std_err"][0]) if ols.height else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    lo = coefs - 1.96 * ses
    hi = coefs + 1.96 * ses
    ax.fill_between(taus, lo, hi, alpha=0.20, color="C0",
                    label="QR ±1.96·SE band")
    ax.plot(taus, coefs, marker="o", color="C0", linewidth=2,
            label="QR β(adr_pct)")
    if np.isfinite(ols_coef):
        ax.axhline(ols_coef, color="C3", linestyle="--", linewidth=1.2,
                   label=f"OLS β = {ols_coef:+.4f}")
        if np.isfinite(ols_se):
            ax.axhspan(ols_coef - 1.96 * ols_se, ols_coef + 1.96 * ols_se,
                       color="C3", alpha=0.10)
    ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("τ (quantile)")
    ax.set_ylabel("β on adr_pct")
    ax.set_title("M4 — adr_pct β across quantiles (loose panel, primary spec)")
    ax.set_xticks(taus)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "m4_qr_sweep_adr.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[M7] reading {M4_PATH}")
    m4 = pl.read_parquet(M4_PATH)
    print(f"[M7] reading {M5_PATH}")
    m5 = pl.read_parquet(M5_PATH)
    print(f"[M7] reading {M6_PATH}")
    m6 = pl.read_parquet(M6_PATH)

    print("[M7] writing tables/m4_coef_table.md")
    write_m4_table(m4)
    print("[M7] writing tables/m5_oos_yearly.md")
    write_m5_table(m5)
    print("[M7] writing tables/m6_inference.md")
    write_m6_table(m6)

    print("[M7] writing figures/m5_magnitude_trajectory.png")
    fig_m5_trajectory(m5)
    print("[M7] writing figures/m6_bootstrap_dist_loose.png")
    fig_m6_bootstrap_dist(m6)
    print("[M7] writing figures/m4_qr_sweep_adr.png")
    fig_m4_qr_sweep_adr(m4)

    print("[M7] done")


if __name__ == "__main__":
    main()
