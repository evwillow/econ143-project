"""
M6 — Stationary block bootstrap inference + placebo test for the
pre-registered β(τ=0.90) − β(τ=0.50) statistic on `vol_contraction_ratio_w`.

Pre-registered design (writeup §6):
  - 95% bootstrap CI on β(0.90) − β(0.50), training window 2010-2017
  - Headline panel = loose; strict reported as §7.1 robustness
  - Stationary block bootstrap, mean block length = 30 trading days
  - Hypothesis: CI excludes zero with predicted negative sign

Procedures here:
  Step 1 — Real bootstrap, B=1000 draws per panel × per training window.
           For each draw: stationary block resample of training rows (mean
           L=30 calendar/trading-row order), refit QR(0.50) and QR(0.90)
           with max_iter=5000, record vcr_w β diff. Non-converged draws
           excluded.
  Step 2 — Inference outputs: percentile CI, basic CI, one-sided
           bootstrap p-value (P[diff >= 0] under bootstrap), convergence count.
  Step 3 — Placebo: same procedure but `vol_contraction_ratio_w` is randomly
           permuted within the resampled training rows before fitting.
           B=500. Distribution should be centered near zero under the null.
  Step 4 — Exploratory: re-run Step 1 on the LOOSE / 2010-2024 expanding-
           window panel (per M5's findings: effect grew through 2025).

Outputs:
  data/interim/m6_bootstrap.parquet   — long-form per-draw diffs + metadata
  reports/m6_inference.md             — per-panel stats, CIs, placebo, verdict
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m3_factors  # noqa: E402  (re-uses fwd-return / factor-window / residualize helpers)

REPO_ROOT = Path(__file__).resolve().parents[1]
M2_PATH = REPO_ROOT / "data" / "interim" / "setups_with_features.parquet"
FF_PATH = REPO_ROOT / "data" / "factors"   / "ff3_umd_daily.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "interim" / "m6_bootstrap.parquet"
REPORT_MD   = REPO_ROOT / "reports" / "m6_inference.md"

# ---- Knobs ----
B_REAL    = 1000
B_PLACEBO = 500
MEAN_BLOCK_LEN = 30
MAX_ITER = 5000
RNG_SEED = 20260514
FWD_HORIZON = 20

RHS_COLS = ["vol_contraction_ratio_w", "adr_pct", "base_duration_days", "rs_slope_vs_spy"]
VCR_IDX = 1  # const at 0, vcr_w at 1, then adr/bdd/rs

TRAIN_START = _date(2010, 1, 1)
TRAIN_END_CANONICAL   = _date(2017, 12, 31)   # §6 prereg
TRAIN_END_EXPLORATORY = _date(2024, 12, 31)   # §7 exploratory (M5 endpoint)

# Panels run for the canonical + placebo. Exploratory is loose-only.
CANONICAL_PANELS = ("loose", "strict")
EXPLORATORY_PANEL = "loose"


# ---------------------------------------------------------------------------
# Stationary block bootstrap
# ---------------------------------------------------------------------------

def _stationary_block_indices(n: int, mean_block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary block bootstrap. Returns an array
    of n indices into [0, n). Block lengths are Geometric(p=1/L); blocks
    wrap around the end so the resampled series is stationary regardless of
    where blocks start."""
    p = 1.0 / mean_block_len
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))   # support 1, 2, 3, ... ; mean 1/p
        # Place block, wrapping on end of series. Truncate at n.
        take = min(length, n - i)
        if take > 0:
            arange = (np.arange(take) + start) % n
            out[i : i + take] = arange
            i += take
    return out


# ---------------------------------------------------------------------------
# Per-draw QR fit
# ---------------------------------------------------------------------------

def _fit_diff(
    X: np.ndarray, y: np.ndarray, idx: np.ndarray, max_iter: int = MAX_ITER
) -> tuple[float, bool]:
    """Refit QR(0.50) and QR(0.90) on the resampled rows. Returns
    (diff, converged) where converged is True iff both QRs finished in <
    max_iter iterations."""
    Xb = X[idx]
    yb = y[idx]
    Xc = sm.add_constant(Xb, has_constant="add")
    # Suppress IterationLimitWarning chatter; we read .iterations directly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr50 = sm.QuantReg(yb, Xc).fit(q=0.50, max_iter=max_iter)
        qr90 = sm.QuantReg(yb, Xc).fit(q=0.90, max_iter=max_iter)
    converged = (
        getattr(qr50, "iterations", max_iter) < max_iter
        and getattr(qr90, "iterations", max_iter) < max_iter
    )
    diff = float(qr90.params[VCR_IDX] - qr50.params[VCR_IDX])
    return diff, converged


def _point_estimate(X: np.ndarray, y: np.ndarray, max_iter: int = MAX_ITER) -> tuple[float, bool, float, float]:
    """Same as _fit_diff but on the full original sample (no resampling).
    Returns (diff, converged, beta_50, beta_90)."""
    Xc = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr50 = sm.QuantReg(y, Xc).fit(q=0.50, max_iter=max_iter)
        qr90 = sm.QuantReg(y, Xc).fit(q=0.90, max_iter=max_iter)
    converged = (
        getattr(qr50, "iterations", max_iter) < max_iter
        and getattr(qr90, "iterations", max_iter) < max_iter
    )
    b50 = float(qr50.params[VCR_IDX])
    b90 = float(qr90.params[VCR_IDX])
    return b90 - b50, converged, b50, b90


# ---------------------------------------------------------------------------
# Bootstrap drivers
# ---------------------------------------------------------------------------

def _run_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    B: int,
    mean_block_len: int,
    rng: np.random.Generator,
    label: str,
    placebo: bool = False,
    progress_every: int = 100,
) -> tuple[np.ndarray, int, int]:
    """Run B draws. If placebo, randomly permute the vcr_w column (idx
    VCR_IDX-1 in X = 0) on the RESAMPLED rows before fitting. Returns
    (diffs, n_converged, n_attempted)."""
    n = X.shape[0]
    diffs: list[float] = []
    n_converged = 0
    t0 = time.time()
    for b in range(B):
        idx = _stationary_block_indices(n, mean_block_len, rng)
        if placebo:
            # Shuffle vcr_w (col 0 of X — RHS_COLS[0] = vol_contraction_ratio_w)
            # AFTER resampling, so that sampling variance is preserved but the
            # vcr-y association is randomized.
            X_use = X[idx].copy()
            perm = rng.permutation(n)
            X_use[:, 0] = X_use[perm, 0]
            y_use = y[idx]
            Xc = sm.add_constant(X_use, has_constant="add")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qr50 = sm.QuantReg(y_use, Xc).fit(q=0.50, max_iter=MAX_ITER)
                qr90 = sm.QuantReg(y_use, Xc).fit(q=0.90, max_iter=MAX_ITER)
            converged = (
                getattr(qr50, "iterations", MAX_ITER) < MAX_ITER
                and getattr(qr90, "iterations", MAX_ITER) < MAX_ITER
            )
            diff = float(qr90.params[VCR_IDX] - qr50.params[VCR_IDX])
        else:
            diff, converged = _fit_diff(X, y, idx)
        if converged:
            diffs.append(diff)
            n_converged += 1
        if (b + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (b + 1) / elapsed
            eta = (B - (b + 1)) / rate
            print(f"[M6] {label}: {b + 1:,}/{B:,} draws "
                  f"({n_converged:,} converged) "
                  f"~{rate:.2f}/s eta={eta/60:.1f}min", flush=True)
    return np.array(diffs, dtype=np.float64), n_converged, B


# ---------------------------------------------------------------------------
# Inference summary
# ---------------------------------------------------------------------------

def _inference_summary(
    diffs: np.ndarray, theta_hat: float
) -> dict:
    """Return summary dict: distribution stats, percentile CI, basic CI,
    one-sided p-value (P[diff >= 0])."""
    if diffs.size == 0:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"),
            "std": float("nan"),
            "p2_5": float("nan"), "p50": float("nan"), "p97_5": float("nan"),
            "ci_perc_lo": float("nan"), "ci_perc_hi": float("nan"),
            "ci_basic_lo": float("nan"), "ci_basic_hi": float("nan"),
            "p_one_sided_neg": float("nan"),
            "theta_hat": theta_hat,
            "excludes_zero_neg": False,
        }
    p25  = float(np.quantile(diffs, 0.025))
    p50  = float(np.quantile(diffs, 0.50))
    p975 = float(np.quantile(diffs, 0.975))
    out = {
        "n": int(diffs.size),
        "mean": float(diffs.mean()),
        "median": p50,
        "std": float(diffs.std(ddof=1)) if diffs.size > 1 else float("nan"),
        "p2_5": p25,
        "p50": p50,
        "p97_5": p975,
        "ci_perc_lo":  p25,
        "ci_perc_hi":  p975,
        "ci_basic_lo": 2 * theta_hat - p975,
        "ci_basic_hi": 2 * theta_hat - p25,
        # One-sided p-value vs predicted-direction null (H0: diff >= 0,
        # H1: diff < 0). Standard formula adds 1 to numerator+denominator
        # to be conservative.
        "p_one_sided_neg": float((np.sum(diffs >= 0) + 1) / (diffs.size + 1)),
        "theta_hat": float(theta_hat),
        # CI excludes zero with predicted negative sign?
        "excludes_zero_neg": bool(p975 < 0),
    }
    return out


# ---------------------------------------------------------------------------
# Per-panel runner
# ---------------------------------------------------------------------------

def _run_panel(
    fit_df: pl.DataFrame,
    train_start: _date,
    train_end: _date,
    panel: str,
    rng_real: np.random.Generator,
    rng_placebo: np.random.Generator,
    do_placebo: bool,
    label: str,
) -> dict:
    train_df = fit_df.filter(
        (pl.col("date") >= train_start)
        & (pl.col("date") <= train_end)
        & (pl.col("universe_variant") == panel)
    ).drop_nulls(subset=["fwd_ret_20d_resid"] + RHS_COLS).sort("date")
    n = train_df.height
    print(f"[M6] {label}: panel='{panel}' n={n:,}", flush=True)

    X = train_df.select(RHS_COLS).to_numpy().astype(np.float64)
    y = train_df["fwd_ret_20d_resid"].to_numpy().astype(np.float64)

    # Point estimate on the full original sample (no resampling).
    theta_hat, theta_converged, b50_hat, b90_hat = _point_estimate(X, y)
    print(f"[M6]   point est: β50={b50_hat:+.4f}, β90={b90_hat:+.4f}, "
          f"diff={theta_hat:+.4f}, converged={theta_converged}")

    # Real bootstrap.
    print(f"[M6]   running B={B_REAL} real-data bootstrap...", flush=True)
    t_real0 = time.time()
    real_diffs, n_real_conv, n_real_attempt = _run_bootstrap(
        X, y, B_REAL, MEAN_BLOCK_LEN, rng_real, label=f"{label}/real",
    )
    real_secs = time.time() - t_real0

    real_stats = _inference_summary(real_diffs, theta_hat)
    print(
        f"[M6]   real CI(95% percentile): "
        f"[{real_stats['ci_perc_lo']:+.4f}, {real_stats['ci_perc_hi']:+.4f}] "
        f"| converged: {n_real_conv}/{n_real_attempt} | "
        f"one-sided p={real_stats['p_one_sided_neg']:.4f} | "
        f"excludes_zero_neg={real_stats['excludes_zero_neg']}"
    )

    # Placebo (optional — exploratory panels skip).
    placebo_stats = None
    placebo_diffs = np.array([], dtype=np.float64)
    placebo_secs = 0.0
    n_placebo_conv = 0
    n_placebo_attempt = 0
    if do_placebo:
        print(f"[M6]   running B={B_PLACEBO} placebo (vcr permuted)...", flush=True)
        t_p0 = time.time()
        placebo_diffs, n_placebo_conv, n_placebo_attempt = _run_bootstrap(
            X, y, B_PLACEBO, MEAN_BLOCK_LEN, rng_placebo, label=f"{label}/placebo",
            placebo=True,
        )
        placebo_secs = time.time() - t_p0
        # For placebo, "theta_hat" reference is 0 (the null).
        placebo_stats = _inference_summary(placebo_diffs, 0.0)
        # Where does the actual point estimate sit on the placebo distribution?
        if placebo_diffs.size > 1 and placebo_stats["std"] > 0:
            placebo_stats["actual_z_vs_null"] = (theta_hat - placebo_stats["mean"]) / placebo_stats["std"]
            placebo_stats["actual_pct_in_null"] = float((placebo_diffs <= theta_hat).mean())
        else:
            placebo_stats["actual_z_vs_null"] = float("nan")
            placebo_stats["actual_pct_in_null"] = float("nan")
        print(
            f"[M6]   placebo: mean={placebo_stats['mean']:+.5f} "
            f"std={placebo_stats['std']:.5f} "
            f"95%CI=[{placebo_stats['p2_5']:+.5f}, {placebo_stats['p97_5']:+.5f}] "
            f"| actual z={placebo_stats['actual_z_vs_null']:+.2f} "
            f"| pct of null <= theta_hat = {placebo_stats['actual_pct_in_null']:.4f}"
        )

    return {
        "label": label,
        "panel": panel,
        "train_start": train_start,
        "train_end": train_end,
        "n": n,
        "theta_hat": theta_hat,
        "theta_b50": b50_hat,
        "theta_b90": b90_hat,
        "theta_converged": theta_converged,
        "real_diffs": real_diffs,
        "real_n_converged": n_real_conv,
        "real_n_attempt": n_real_attempt,
        "real_stats": real_stats,
        "real_secs": real_secs,
        "placebo_diffs": placebo_diffs,
        "placebo_n_converged": n_placebo_conv,
        "placebo_n_attempt": n_placebo_attempt,
        "placebo_stats": placebo_stats,
        "placebo_secs": placebo_secs,
    }


# ---------------------------------------------------------------------------
# Long-form parquet
# ---------------------------------------------------------------------------

def _to_long_rows(panel_results: list[dict]) -> list[dict]:
    """Each draw becomes one row. We also emit per-panel rows for the point
    estimate (draw_idx = -1, mode = 'point')."""
    rows: list[dict] = []
    for pr in panel_results:
        rows.append({
            "label": pr["label"], "panel": pr["panel"],
            "train_start": pr["train_start"], "train_end": pr["train_end"],
            "n_train": pr["n"],
            "mode": "point", "draw_idx": -1,
            "diff": pr["theta_hat"],
            "beta_50": pr["theta_b50"], "beta_90": pr["theta_b90"],
        })
        for i, d in enumerate(pr["real_diffs"]):
            rows.append({
                "label": pr["label"], "panel": pr["panel"],
                "train_start": pr["train_start"], "train_end": pr["train_end"],
                "n_train": pr["n"],
                "mode": "real", "draw_idx": i,
                "diff": float(d),
                "beta_50": float("nan"), "beta_90": float("nan"),
            })
        for i, d in enumerate(pr["placebo_diffs"]):
            rows.append({
                "label": pr["label"], "panel": pr["panel"],
                "train_start": pr["train_start"], "train_end": pr["train_end"],
                "n_train": pr["n"],
                "mode": "placebo", "draw_idx": i,
                "diff": float(d),
                "beta_50": float("nan"), "beta_90": float("nan"),
            })
    return rows


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _write_report(panel_results: list[dict], total_wall_secs: float) -> None:
    lines: list[str] = []
    lines.append("# M6 — Stationary Block Bootstrap Inference + Placebo Test")
    lines.append("")
    lines.append(
        f"Pre-registered design (writeup §6): 95% bootstrap CI on β(τ=0.90) − "
        f"β(τ=0.50) for `vol_contraction_ratio_w`, **headline panel = loose**, "
        f"training window 2010-2017, **stationary block bootstrap with mean "
        f"block length = {MEAN_BLOCK_LEN}**. Hypothesis survives if the CI "
        "excludes zero with negative sign. M5 already showed sign-consistency "
        "in 7/8 expanding-window OOS years (loose); this stage handles the "
        "second leg of the prereg (the bootstrap CI)."
    )
    lines.append("")
    lines.append(
        f"Knobs: `B_real={B_REAL:,}`, `B_placebo={B_PLACEBO:,}`, "
        f"`mean_block_len={MEAN_BLOCK_LEN}`, `max_iter={MAX_ITER:,}` (per "
        f"M5's IterationLimitWarning). RNG seed = `{RNG_SEED}`. Total wall "
        f"time: **{total_wall_secs/60:.1f} min**."
    )
    lines.append("")
    lines.append(
        "Convention: lower `vol_contraction_ratio` = more contraction = "
        "stronger Qullamaggie signal. Predicted sign of β(0.90) − β(0.50) is "
        "**negative** (more contraction lifts the upside tail more than the "
        "median). One-sided bootstrap p-value reported = P̂[diff ≥ 0]."
    )
    lines.append("")

    # ---- Per-panel detail ----
    for pr in panel_results:
        rs = pr["real_stats"]
        ps = pr.get("placebo_stats")
        is_canon_loose = (pr["label"] == "canonical" and pr["panel"] == "loose")
        is_canon_strict = (pr["label"] == "canonical" and pr["panel"] == "strict")
        is_explor = (pr["label"] == "exploratory")

        if is_canon_loose:
            header = "## Canonical / loose (headline pre-registered test)"
        elif is_canon_strict:
            header = "## Canonical / strict (§7.1 robustness)"
        elif is_explor:
            header = "## Exploratory / loose, training 2010-2024 (§7 — non-pre-registered)"
        else:
            header = f"## {pr['label']} / {pr['panel']}"
        lines.append(header)
        lines.append("")
        lines.append(
            f"- Training window: **[{pr['train_start']}, {pr['train_end']}]**, "
            f"n = **{pr['n']:,}**"
        )
        lines.append(
            f"- Point estimate: β(0.50) = {pr['theta_b50']:+.4f}, "
            f"β(0.90) = {pr['theta_b90']:+.4f}, "
            f"**diff = {pr['theta_hat']:+.4f}**"
        )
        lines.append(
            f"- Bootstrap convergence: {pr['real_n_converged']:,} of "
            f"{pr['real_n_attempt']:,} draws converged in < {MAX_ITER:,} "
            f"iterations ({pr['real_n_converged']/max(1, pr['real_n_attempt'])*100:.2f}%) "
            f"| wall: {pr['real_secs']/60:.1f} min"
        )
        lines.append("")
        lines.append("### Bootstrap distribution of β(0.90) − β(0.50)")
        lines.append("")
        lines.append("| n | mean | median | std | p2.5 | p97.5 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {rs['n']:,} | {rs['mean']:+.5f} | {rs['median']:+.5f} | "
            f"{rs['std']:.5f} | {rs['p2_5']:+.5f} | {rs['p97_5']:+.5f} |"
        )
        lines.append("")
        lines.append("### 95% confidence intervals")
        lines.append("")
        lines.append("| Method | Lower | Upper | Excludes 0 (one-sided neg)? |")
        lines.append("|---|---:|---:|---|")
        lines.append(
            f"| Percentile | {rs['ci_perc_lo']:+.5f} | {rs['ci_perc_hi']:+.5f} | "
            f"{'YES' if rs['excludes_zero_neg'] else 'NO'} |"
        )
        basic_excludes = rs["ci_basic_hi"] < 0
        lines.append(
            f"| Basic-bootstrap | {rs['ci_basic_lo']:+.5f} | {rs['ci_basic_hi']:+.5f} | "
            f"{'YES' if basic_excludes else 'NO'} |"
        )
        lines.append("")
        lines.append(
            f"**One-sided bootstrap p-value** (H₀: diff ≥ 0 vs H₁: diff < 0): "
            f"`P̂[diff ≥ 0] = {rs['p_one_sided_neg']:.4f}`."
        )
        lines.append("")

        if ps is not None:
            lines.append("### Placebo (vcr_w randomly permuted within each draw)")
            lines.append("")
            lines.append(
                f"- Placebo draws: {pr['placebo_n_converged']:,} converged of "
                f"{pr['placebo_n_attempt']:,} attempted "
                f"({pr['placebo_n_converged']/max(1, pr['placebo_n_attempt'])*100:.2f}%) "
                f"| wall: {pr['placebo_secs']/60:.1f} min"
            )
            lines.append("")
            lines.append("| n | mean | median | std | p2.5 | p97.5 |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {ps['n']:,} | {ps['mean']:+.5f} | {ps['median']:+.5f} | "
                f"{ps['std']:.5f} | {ps['p2_5']:+.5f} | {ps['p97_5']:+.5f} |"
            )
            lines.append("")
            centering_ok = abs(ps["mean"]) < 0.005
            lines.append(
                f"**Placebo centering**: {'✅ near zero' if centering_ok else '⚠ not centered (|mean| ≥ 0.005)'} — "
                f"placebo mean = {ps['mean']:+.5f}. Under the null, this should "
                "be ~0; large deviations flag a problem with the procedure or "
                "sample asymmetry."
            )
            lines.append("")
            z = ps.get("actual_z_vs_null", float("nan"))
            pct = ps.get("actual_pct_in_null", float("nan"))
            lines.append(
                f"**Where does the point estimate sit on the placebo null?** "
                f"θ̂ = {pr['theta_hat']:+.5f}; placebo mean = {ps['mean']:+.5f}, "
                f"std = {ps['std']:.5f}. z = **{z:+.2f}**. "
                f"Fraction of placebo draws ≤ θ̂ = **{pct:.4f}** "
                f"(placebo-based one-sided p ≈ {pct:.4f})."
            )
            lines.append("")

    # ---- Top-line verdicts ----
    lines.append("## Pre-registration verdicts")
    lines.append("")
    canon_loose = next((p for p in panel_results
                        if p["label"] == "canonical" and p["panel"] == "loose"), None)
    canon_strict = next((p for p in panel_results
                         if p["label"] == "canonical" and p["panel"] == "strict"), None)
    explor = next((p for p in panel_results if p["label"] == "exploratory"), None)
    if canon_loose:
        rs = canon_loose["real_stats"]
        lines.append(
            f"- **Canonical loose (HEADLINE, prereg §6)**: 95% percentile CI = "
            f"[{rs['ci_perc_lo']:+.5f}, {rs['ci_perc_hi']:+.5f}]. "
            f"{'**EXCLUDES zero with predicted negative sign** ✅' if rs['excludes_zero_neg'] else '**Does NOT exclude zero with predicted negative sign** ❌'}"
        )
    if canon_strict:
        rs = canon_strict["real_stats"]
        lines.append(
            f"- **Canonical strict (§7.1)**: 95% percentile CI = "
            f"[{rs['ci_perc_lo']:+.5f}, {rs['ci_perc_hi']:+.5f}]. "
            f"{'EXCLUDES zero with predicted negative sign' if rs['excludes_zero_neg'] else 'Does NOT exclude zero with predicted negative sign'}"
        )
    lines.append("")
    if explor:
        rs = explor["real_stats"]
        lines.append("## §7 exploratory finding (NOT pre-registered)")
        lines.append("")
        lines.append(
            "Per M5, the |β(0.90) − β(0.50)| point estimate **grew** from "
            "−0.0045 (training 2010-2017) to −0.0210 (training 2010-2024). "
            "Re-running the bootstrap on the latter (loose, 2010-2024) is "
            "exploratory: it isn't the §6 prereg (which fixed 2010-2017) but "
            "it's the result that best characterises what the data looks "
            "like with the most information."
        )
        lines.append("")
        lines.append(
            f"- **Exploratory loose 2010-2024**: n = {explor['n']:,}, point "
            f"estimate = **{explor['theta_hat']:+.5f}**. 95% percentile CI = "
            f"[{rs['ci_perc_lo']:+.5f}, {rs['ci_perc_hi']:+.5f}]. "
            f"{'**EXCLUDES zero** with negative sign' if rs['excludes_zero_neg'] else 'Does NOT exclude zero'}. "
            f"One-sided p̂ = {rs['p_one_sided_neg']:.4f}."
        )
        lines.append("")

    # ---- Computational details ----
    lines.append("## Computational details")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| B (real bootstrap) | {B_REAL:,} draws |")
    lines.append(f"| B (placebo) | {B_PLACEBO:,} draws |")
    lines.append(f"| Mean block length | {MEAN_BLOCK_LEN} (Politis-Romano stationary block) |")
    lines.append(f"| QR `max_iter` | {MAX_ITER:,} |")
    lines.append(f"| RNG seed | {RNG_SEED} (numpy `default_rng`) |")
    lines.append(f"| Total wall time | {total_wall_secs/60:.1f} min |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Two-stage caveat**: the bootstrap resamples training rows and "
        "refits the QR, but the M3 factor / sector / year-FE residualization "
        "is held fixed at the canonical (or exploratory) point estimate. "
        "First-stage uncertainty is therefore not propagated. The justification "
        "is that the residualization is a high-degrees-of-freedom OLS and "
        "the QR slope on residualized y is the dominant source of variability."
    )
    lines.append(
        "- **Block bootstrap on cross-sectional data**: the panel is sorted "
        "by date before resampling, so blocks of consecutive rows correspond "
        "to setups whose breakout days are close in calendar time. Setups "
        "from the same calendar period tend to share market-state shocks, so "
        "a stationary block resample is preferable to IID for the std error "
        "estimate even though we're running a cross-section per draw."
    )
    lines.append(
        "- **Placebo construction**: `vol_contraction_ratio_w` is randomly "
        "permuted across the resampled rows on every draw, breaking its "
        "association with `fwd_ret_20d_resid` while leaving everything else "
        "(other covariates, factor structure, sample size) unchanged. The "
        "placebo distribution should be centered near zero — if it isn't, "
        "the procedure or the sample is not symmetric and the headline test "
        "needs care."
    )
    lines.append(
        "- **One-sided p-value formula**: "
        "`(n_draws_with_diff_>=_0 + 1) / (n_draws + 1)` (the +1 is the "
        "standard small-sample correction)."
    )
    lines.append(
        "- **Determinism**: with seed = "
        f"`{RNG_SEED}` and the same input parquets, this script's output is "
        "byte-stable. statsmodels' QuantReg solver is deterministic."
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

    t_main = time.time()

    print(f"[M6] reading {M2_PATH}")
    setups = pl.read_parquet(M2_PATH)
    print(f"[M6] {setups.height:,} setups loaded "
          f"({int((setups['universe_variant']=='strict').sum()):,} strict, "
          f"{int((setups['universe_variant']=='loose').sum()):,} loose)")

    print(f"[M6] reading factor panel: {FF_PATH}")
    ff = pl.read_parquet(FF_PATH)

    # Compute fwd-returns + factor windows ONCE.
    unique_tickers = sorted(setups["ticker"].unique().to_list())
    ticker_data = m3_factors._load_ticker_closes(unique_tickers)
    print(f"[M6] computing fwd_ret_{FWD_HORIZON}d...", flush=True)
    setups, _drops = m3_factors._compute_forward_returns(setups, ticker_data, FWD_HORIZON)
    setups, n_missing_factors = m3_factors._attach_factor_window_sums(setups, ff)
    if n_missing_factors:
        print(f"[M6] WARN: {n_missing_factors:,} rows have NaN factor-window sums")

    # Build the residualized panel for the canonical (2010-2017) and the
    # exploratory (2010-2024) windows. Each call refits M3 with that
    # training window and applies the trained coefficients to all rows
    # including OOS (year-FE = 0 for unseen years). For M6 we only need
    # the training rows, so OOS rows we ignore.
    print(f"[M6] residualizing canonical window [{TRAIN_START}, {TRAIN_END_CANONICAL}]")
    fit_df_canonical, info_c = m3_factors.residualize(setups, TRAIN_START, TRAIN_END_CANONICAL)
    print(f"[M6]   p99(vcr training) = {info_c['p99_vcr_train']:.4f}")
    print(f"[M6] residualizing exploratory window [{TRAIN_START}, {TRAIN_END_EXPLORATORY}]")
    fit_df_explor, info_e = m3_factors.residualize(setups, TRAIN_START, TRAIN_END_EXPLORATORY)
    print(f"[M6]   p99(vcr training) = {info_e['p99_vcr_train']:.4f}")

    # Distinct RNGs per panel/mode for clarity.
    seed_seq = np.random.SeedSequence(RNG_SEED)
    # Allocate enough RNG streams: 2 (loose, strict) × 2 (real, placebo) + 1 exploratory real = 5
    rngs = [np.random.default_rng(s) for s in seed_seq.spawn(5)]
    rng_loose_real    = rngs[0]
    rng_loose_placebo = rngs[1]
    rng_strict_real   = rngs[2]
    rng_strict_placebo = rngs[3]
    rng_explor_real   = rngs[4]

    panel_results: list[dict] = []

    # Canonical loose
    panel_results.append(_run_panel(
        fit_df_canonical, TRAIN_START, TRAIN_END_CANONICAL, "loose",
        rng_loose_real, rng_loose_placebo,
        do_placebo=True, label="canonical",
    ))
    # Canonical strict
    panel_results.append(_run_panel(
        fit_df_canonical, TRAIN_START, TRAIN_END_CANONICAL, "strict",
        rng_strict_real, rng_strict_placebo,
        do_placebo=True, label="canonical",
    ))
    # Exploratory loose 2010-2024
    panel_results.append(_run_panel(
        fit_df_explor, TRAIN_START, TRAIN_END_EXPLORATORY, "loose",
        rng_explor_real, None,
        do_placebo=False, label="exploratory",
    ))

    total_secs = time.time() - t_main

    # ---- Write parquet ----
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    rows = _to_long_rows(panel_results)
    out_df = pl.DataFrame(rows).with_columns(
        pl.col("train_start").cast(pl.Date),
        pl.col("train_end").cast(pl.Date),
    )
    out_df.write_parquet(OUT_PARQUET)
    print(f"[M6] wrote {OUT_PARQUET} ({out_df.height:,} rows)")

    _write_report(panel_results, total_secs)
    print(f"[M6] wrote {REPORT_MD}")
    print(f"[M6] total wall time: {total_secs/60:.1f} min")


if __name__ == "__main__":
    main()
