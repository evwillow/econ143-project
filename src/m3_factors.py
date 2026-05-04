"""Download and parse Ken French FF3 + UMD daily factors AND residualize M2
setups.

Two responsibilities live in this module:
  1. Build the daily factor panel (date, mkt_rf, smb, hml, umd, rf in decimal
     form) -> data/factors/ff3_umd_daily.parquet.  (Original purpose; idempotent.)
  2. Compute fwd_ret_20d per setup, residualize against the FF3+UMD factors
     (cumulated over [t+1, t+20]) plus sector + year fixed effects on the
     2010-2017 training window, and apply the same fitted coefficients to
     2018-2025 OOS rows. Winsorize vol_contraction_ratio at training-window
     p99 -> data/interim/setups_with_residuals.parquet.

Run with `python src/m3_factors.py` -- step 1 only fires if the factor panel
parquet is missing; step 2 always runs. Validation goes to
reports/m3_validation.md.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

# stats are used in step 2 (residualization). Imported eagerly because
# they're cheap and the residualization step always runs.
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DAILY_BARS_GLOB  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTORS_DIR = REPO_ROOT / "data" / "factors"
RAW_DIR = FACTORS_DIR / "raw"
OUTPUT_PATH = FACTORS_DIR / "ff3_umd_daily.parquet"

# Step 2 paths
M2_PATH = REPO_ROOT / "data" / "interim" / "setups_with_features.parquet"
OUT_PATH = REPO_ROOT / "data" / "interim" / "setups_with_residuals.parquet"
VALIDATION_MD = REPO_ROOT / "reports" / "m3_validation.md"

# Training / OOS split per writeup §6
TRAIN_START = _date(2010, 1, 1)
TRAIN_END   = _date(2017, 12, 31)
OOS_START   = _date(2018, 1, 1)
OOS_END     = _date(2025, 12, 31)

# 20 trading days forward
FWD_HORIZON = 20

# Winsorization quantile (per task brief, M2 flag for vol_contraction_ratio
# max=15.39 vs median=0.94)
WINSOR_Q = 0.99

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
UMD_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

START_DATE = pl.date(2008, 1, 1)


def download_factors() -> tuple[Path, Path]:
    """Download both zips into RAW_DIR and return their paths."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ff3_zip = RAW_DIR / "F-F_Research_Data_Factors_daily_CSV.zip"
    umd_zip = RAW_DIR / "F-F_Momentum_Factor_daily_CSV.zip"
    for url, dest in [(FF3_URL, ff3_zip), (UMD_URL, umd_zip)]:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
    return ff3_zip, umd_zip


def _read_csv_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV inside {zip_path}")
        with zf.open(names[0]) as f:
            return f.read().decode("latin-1")


def _extract_data_block(text: str) -> str:
    """Return only the daily-data block: lines whose first token is 8 digits.

    Ken French CSVs have a multi-line title header, then a data section, then
    sometimes annual/footer sections. The daily block is the run of lines
    starting with an 8-digit YYYYMMDD date.
    """
    out_lines: list[str] = []
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if started:
                break
            continue
        first = line.split(",", 1)[0].strip()
        if len(first) == 8 and first.isdigit():
            out_lines.append(line)
            started = True
        elif started:
            break
    if not out_lines:
        raise RuntimeError("Could not find daily data block in CSV")
    return "\n".join(out_lines)


def parse_ff3(zip_path: Path) -> pl.DataFrame:
    """Parse the FF3 daily CSV into a polars frame with date, mkt_rf, smb, hml, rf (decimals)."""
    text = _read_csv_from_zip(zip_path)
    block = _extract_data_block(text)
    df = pl.read_csv(
        io.StringIO(block),
        has_header=False,
        new_columns=["date_int", "mkt_rf", "smb", "hml", "rf"],
        schema_overrides={
            "date_int": pl.Int64,
            "mkt_rf": pl.Float64,
            "smb": pl.Float64,
            "hml": pl.Float64,
            "rf": pl.Float64,
        },
    )
    return df.with_columns(
        pl.col("date_int").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
        (pl.col("mkt_rf") / 100.0).alias("mkt_rf"),
        (pl.col("smb") / 100.0).alias("smb"),
        (pl.col("hml") / 100.0).alias("hml"),
        (pl.col("rf") / 100.0).alias("rf"),
    ).select("date", "mkt_rf", "smb", "hml", "rf")


def parse_umd(zip_path: Path) -> pl.DataFrame:
    """Parse the UMD/Mom daily CSV into a polars frame with date, umd (decimals)."""
    text = _read_csv_from_zip(zip_path)
    block = _extract_data_block(text)
    df = pl.read_csv(
        io.StringIO(block),
        has_header=False,
        new_columns=["date_int", "umd"],
        schema_overrides={"date_int": pl.Int64, "umd": pl.Float64},
    )
    return df.with_columns(
        pl.col("date_int").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d").alias("date"),
        (pl.col("umd") / 100.0).alias("umd"),
    ).select("date", "umd")


def build_factor_panel(start: pl.Expr | None = None) -> pl.DataFrame:
    ff3_zip, umd_zip = download_factors()
    ff3 = parse_ff3(ff3_zip)
    umd = parse_umd(umd_zip)
    panel = (
        ff3.join(umd, on="date", how="inner")
        .drop_nulls()
        .filter(pl.col("date") >= START_DATE)
        .sort("date")
        .select("date", "mkt_rf", "smb", "hml", "umd", "rf")
    )
    return panel


# ---------------------------------------------------------------------------
# Step 2: residualize M2 setups
# ---------------------------------------------------------------------------

FACTOR_COLS = ("mkt_rf", "smb", "hml", "umd")
FACTOR_CUM_COLS = tuple(f"{c}_cum" for c in FACTOR_COLS)
FACTOR_WINDOW_COLS = tuple(f"{c}_window" for c in FACTOR_COLS)


def _load_ticker_closes(tickers: list[str]) -> dict[str, dict]:
    """Per-ticker {dates, closes} numpy arrays, sorted by date."""
    print(f"[M3] loading daily bars for {len(tickers):,} setup tickers...", flush=True)
    bars = (
        pl.scan_parquet(DAILY_BARS_GLOB)
          .filter(pl.col("ticker").is_in(tickers))
          .select("ticker", "date", "close")
          .sort(["ticker", "date"])
          .collect()
    )
    out: dict[str, dict] = {}
    for tk_df in bars.partition_by("ticker"):
        tk = tk_df["ticker"][0]
        out[tk] = {
            "dates":  tk_df["date"].to_numpy().astype("datetime64[D]"),
            "closes": tk_df["close"].to_numpy().astype(np.float64),
        }
    return out


def _compute_forward_returns(
    setups: pl.DataFrame,
    ticker_data: dict[str, dict],
    horizon: int,
) -> tuple[pl.DataFrame, list[dict]]:
    """Add fwd_ret_20d + fwd_end_date columns. Return (setups, drops)."""
    n = setups.height
    fwd = np.full(n, np.nan, dtype=np.float64)
    fwd_end = np.full(n, np.datetime64("NaT", "D"))
    drops: list[dict] = []
    for i, r in enumerate(setups.iter_rows(named=True)):
        tk = r["ticker"]
        td = ticker_data.get(tk)
        if td is None:
            drops.append({"ticker": tk, "date": r["date"],
                          "universe_variant": r["universe_variant"],
                          "reason": "no daily bars for ticker"})
            continue
        d_t = np.datetime64(r["date"], "D")
        idx_t = int(np.searchsorted(td["dates"], d_t, side="left"))
        if idx_t >= td["dates"].size or td["dates"][idx_t] != d_t:
            drops.append({"ticker": tk, "date": r["date"],
                          "universe_variant": r["universe_variant"],
                          "reason": "setup date not present in bar series"})
            continue
        idx_h = idx_t + horizon
        if idx_h >= td["dates"].size:
            drops.append({"ticker": tk, "date": r["date"],
                          "universe_variant": r["universe_variant"],
                          "reason": f"t+{horizon} beyond end of bar series"})
            continue
        c_t = float(td["closes"][idx_t])
        c_h = float(td["closes"][idx_h])
        if not (np.isfinite(c_t) and np.isfinite(c_h) and c_t > 0 and c_h > 0):
            drops.append({"ticker": tk, "date": r["date"],
                          "universe_variant": r["universe_variant"],
                          "reason": "non-positive/NaN close at t or t+horizon"})
            continue
        fwd[i] = c_h / c_t - 1.0
        fwd_end[i] = td["dates"][idx_h]

    return setups.with_columns(
        pl.Series("fwd_ret_20d", fwd),
        pl.Series("fwd_end_date", fwd_end).cast(pl.Date),
    ), drops


def _attach_factor_window_sums(
    setups: pl.DataFrame, ff_panel: pl.DataFrame
) -> tuple[pl.DataFrame, int]:
    """Add cumulated factor sums over (t, t+horizon] (i.e. the daily factor
    realizations from t+1 through t+horizon, summed). Uses cum_sum + two
    date-joins so the math is O(N) rather than O(N * horizon).

    Returns (df_with_factor_windows, n_rows_with_missing_factor_dates).
    """
    ff_cum = (
        ff_panel.sort("date")
        .with_columns(
            pl.col("mkt_rf").cum_sum().alias("mkt_rf_cum"),
            pl.col("smb").cum_sum().alias("smb_cum"),
            pl.col("hml").cum_sum().alias("hml_cum"),
            pl.col("umd").cum_sum().alias("umd_cum"),
        )
        .select("date", *FACTOR_CUM_COLS)
    )

    # Join cum-at-date(t)
    cum_at_t = ff_cum.rename({c: f"{c}_at_t" for c in FACTOR_CUM_COLS})
    setups = setups.join(cum_at_t, left_on="date", right_on="date", how="left")
    # Join cum-at-date(t+h)
    cum_at_h = ff_cum.rename({c: f"{c}_at_h" for c in FACTOR_CUM_COLS})
    setups = setups.join(cum_at_h, left_on="fwd_end_date", right_on="date", how="left")

    # Window sum = cum_at_h - cum_at_t  (which is sum of daily factors over
    # the dates strictly after t up through t+h, since cum_sum is inclusive).
    setups = setups.with_columns(
        (pl.col("mkt_rf_cum_at_h") - pl.col("mkt_rf_cum_at_t")).alias("mkt_rf_window"),
        (pl.col("smb_cum_at_h")    - pl.col("smb_cum_at_t")).alias("smb_window"),
        (pl.col("hml_cum_at_h")    - pl.col("hml_cum_at_t")).alias("hml_window"),
        (pl.col("umd_cum_at_h")    - pl.col("umd_cum_at_t")).alias("umd_window"),
    ).drop([f"{c}_at_t" for c in FACTOR_CUM_COLS]
           + [f"{c}_at_h" for c in FACTOR_CUM_COLS])

    # Diagnostic: how many rows have NaN in any window col (because their
    # date or fwd_end_date wasn't in the factor panel)?
    n_missing = int(
        setups.select(
            (pl.any_horizontal([pl.col(c).is_null() for c in FACTOR_WINDOW_COLS])).cast(pl.Int64).sum()
        ).item()
    )
    return setups, n_missing


def _build_design_matrix(
    df: pl.DataFrame,
    sectors_train: list[str],
    years_train: list[int],
    sector_ref: str,
    year_ref: int,
) -> tuple[np.ndarray, list[str]]:
    """Return (X, col_names) for OLS. Includes:
        const, mkt_rf_window, smb_window, hml_window, umd_window,
        sector dummies (sectors_train minus sector_ref),
        year dummies (years_train minus year_ref).

    For OOS rows where sector or year isn't in the training set, the
    corresponding dummies are 0 -> row is mapped to the reference category."""
    n = df.height
    factor_part = np.column_stack([df[c].to_numpy() for c in FACTOR_WINDOW_COLS])
    sector_arr = df["sector"].to_numpy()
    year_arr   = df["year"].to_numpy()

    sector_cols = [s for s in sectors_train if s != sector_ref]
    year_cols   = [y for y in years_train   if y != year_ref]

    sector_dummies = np.column_stack(
        [(sector_arr == s).astype(np.float64) for s in sector_cols]
    ) if sector_cols else np.zeros((n, 0))
    year_dummies = np.column_stack(
        [(year_arr == y).astype(np.float64) for y in year_cols]
    ) if year_cols else np.zeros((n, 0))

    const = np.ones((n, 1))
    X = np.hstack([const, factor_part, sector_dummies, year_dummies])

    col_names = (
        ["const"]
        + list(FACTOR_WINDOW_COLS)
        + [f"sector::{s}" for s in sector_cols]
        + [f"year::{y}"   for y in year_cols]
    )
    return X, col_names


def _residualize_main() -> None:
    print(f"[M3] reading M2 setups: {M2_PATH}")
    setups = pl.read_parquet(M2_PATH)
    print(f"[M3] {setups.height:,} setups loaded "
          f"({int((setups['universe_variant']=='strict').sum()):,} strict, "
          f"{int((setups['universe_variant']=='loose').sum()):,} loose)")

    print(f"[M3] reading factor panel: {OUTPUT_PATH}")
    ff = pl.read_parquet(OUTPUT_PATH)

    # ---- Step 1: forward returns ----
    unique_tickers = sorted(setups["ticker"].unique().to_list())
    ticker_data = _load_ticker_closes(unique_tickers)
    print(f"[M3] computing fwd_ret_{FWD_HORIZON}d...", flush=True)
    setups, drops = _compute_forward_returns(setups, ticker_data, FWD_HORIZON)
    n_after_fwd = setups.filter(pl.col("fwd_ret_20d").is_not_null()).height
    print(f"[M3] fwd return computed for {n_after_fwd:,} / {setups.height:,} rows "
          f"({len(drops):,} drops)")

    # ---- Step 2: cumulated factor windows ----
    setups, n_missing_factors = _attach_factor_window_sums(setups, ff)
    if n_missing_factors:
        print(f"[M3] WARN: {n_missing_factors:,} rows have NaN factor-window sums "
              "(setup date or fwd_end_date not in factor panel)")

    # Drop rows missing fwd_ret_20d OR factor windows -- can't residualize them
    fit_setups = setups.filter(
        pl.col("fwd_ret_20d").is_not_null()
        & pl.all_horizontal([pl.col(c).is_not_null() for c in FACTOR_WINDOW_COLS])
    )
    n_dropped_for_residual = setups.height - fit_setups.height
    print(f"[M3] {fit_setups.height:,} rows usable for residualization "
          f"({n_dropped_for_residual:,} cannot be scored)")

    fit_setups = fit_setups.with_columns(
        pl.col("date").dt.year().alias("year")
    )

    # ---- Step 3: train/OOS split ----
    train_mask = (fit_setups["date"] >= TRAIN_START) & (fit_setups["date"] <= TRAIN_END)
    train_df = fit_setups.filter(train_mask)
    oos_df   = fit_setups.filter(~train_mask)
    print(f"[M3] train (2010-2017): {train_df.height:,}  |  OOS (2018-2025): {oos_df.height:,}")

    # ---- Step 4: design matrix encoding ----
    sectors_train = sorted(train_df["sector"].unique().to_list())
    years_train   = sorted(train_df["year"].unique().to_list())
    if not sectors_train:
        raise SystemExit("[M3] no sectors in training set — aborting")
    if not years_train:
        raise SystemExit("[M3] no years in training set — aborting")
    # Reference categories: pick the most-populated sector for stability,
    # and the earliest year (2010 by inspection).
    sector_ref = max(sectors_train, key=lambda s: int((train_df["sector"] == s).sum()))
    year_ref = min(years_train)
    print(f"[M3] sector reference: '{sector_ref}'  |  year reference: {year_ref}")

    X_full, col_names = _build_design_matrix(
        fit_setups, sectors_train, years_train, sector_ref, year_ref
    )
    y_full = fit_setups["fwd_ret_20d"].to_numpy()

    train_idx = np.flatnonzero(train_mask.to_numpy())
    X_train = X_full[train_idx]
    y_train = y_full[train_idx]

    print(f"[M3] OLS fit: n={X_train.shape[0]:,}, k={X_train.shape[1]:,}")
    ols = sm.OLS(y_train, X_train).fit()
    print(f"[M3] R^2={ols.rsquared:.4f}  Adj R^2={ols.rsquared_adj:.4f}")

    # ---- Step 5: residuals via training betas, applied to all rows ----
    fitted_full = X_full @ ols.params
    resid_full  = y_full - fitted_full
    fit_setups = fit_setups.with_columns(
        pl.Series("fwd_ret_20d_resid", resid_full),
    )

    # Sanity: training residuals should be ~mean-0
    train_resid_mean = float(resid_full[train_idx].mean())
    print(f"[M3] training residual mean: {train_resid_mean:+.6f} "
          f"({'PASS' if abs(train_resid_mean) <= 1e-3 else 'FAIL'} |x|<=1e-3)")

    # ---- Step 6: winsorize vol_contraction_ratio at training p99 ----
    p99 = float(train_df["vol_contraction_ratio"].quantile(WINSOR_Q))
    print(f"[M3] vol_contraction_ratio training p{int(WINSOR_Q*100)} = {p99:.4f}")
    fit_setups = fit_setups.with_columns(
        pl.min_horizontal([pl.col("vol_contraction_ratio"), pl.lit(p99)])
        .alias("vol_contraction_ratio_w")
    )

    # ---- Output ----
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fit_setups.write_parquet(OUT_PATH)
    print(f"[M3] wrote {OUT_PATH}")

    _write_validation(
        setups_in=setups,
        out_df=fit_setups,
        train_df=train_df,
        oos_df=oos_df,
        ols_result=ols,
        col_names=col_names,
        sector_ref=sector_ref,
        year_ref=year_ref,
        sectors_train=sectors_train,
        years_train=years_train,
        winsor_p99=p99,
        drops=drops,
        n_missing_factors=n_missing_factors,
        n_dropped_for_residual=n_dropped_for_residual,
        resid_full=resid_full,
        train_idx=train_idx,
    )
    print(f"[M3] wrote {VALIDATION_MD}")


def _quantiles(arr: np.ndarray, qs=(0.10, 0.50, 0.90)) -> dict:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs} | {
            "mean": float("nan"), "std": float("nan"), "n": 0,
        }
    out = {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}
    out["mean"] = float(arr.mean())
    out["std"]  = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    out["n"]    = int(arr.size)
    return out


def _write_validation(
    setups_in: pl.DataFrame,
    out_df: pl.DataFrame,
    train_df: pl.DataFrame,
    oos_df: pl.DataFrame,
    ols_result,
    col_names: list[str],
    sector_ref: str,
    year_ref: int,
    sectors_train: list[str],
    years_train: list[int],
    winsor_p99: float,
    drops: list[dict],
    n_missing_factors: int,
    n_dropped_for_residual: int,
    resid_full: np.ndarray,
    train_idx: np.ndarray,
) -> None:
    n_in = setups_in.height
    n_out = out_df.height

    # Setup counts by year (before/after dropping for missing fwd_ret_20d)
    year_table = (
        setups_in.with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg([
            pl.len().alias("input_rows"),
            pl.col("fwd_ret_20d").is_not_null().sum().alias("kept_rows"),
        ])
        .sort("year")
    )

    # Distribution of fwd_ret_20d and fwd_ret_20d_resid, train vs OOS
    train_y = train_df["fwd_ret_20d"].to_numpy()
    oos_y   = oos_df["fwd_ret_20d"].to_numpy()
    train_resid_arr = resid_full[train_idx]
    oos_idx = np.array([i for i in range(out_df.height) if i not in set(train_idx.tolist())])
    oos_resid_arr = resid_full[oos_idx] if oos_idx.size else np.array([])

    train_y_q = _quantiles(train_y)
    oos_y_q   = _quantiles(oos_y)
    train_r_q = _quantiles(train_resid_arr)
    oos_r_q   = _quantiles(oos_resid_arr)

    # Sanity check
    train_resid_mean = float(train_resid_arr.mean()) if train_resid_arr.size else float("nan")
    sanity_pass = (np.isfinite(train_resid_mean) and abs(train_resid_mean) <= 1e-3)

    # Coefficient table (factor terms only, plus intercept)
    coef_rows = []
    for i, name in enumerate(col_names):
        coef_rows.append({
            "term": name,
            "coef": float(ols_result.params[i]),
            "std_err": float(ols_result.bse[i]),
            "t": float(ols_result.tvalues[i]),
            "p": float(ols_result.pvalues[i]),
        })

    # Correlation of residuals with M2 features (complete-case)
    feature_cols = ["vol_contraction_ratio", "vol_contraction_ratio_w",
                    "adr_pct", "base_duration_days", "rs_slope_vs_spy"]
    corrs = {}
    for c in feature_cols:
        sub = out_df.select([c, "fwd_ret_20d_resid"]).drop_nulls()
        if sub.is_empty():
            corrs[c] = float("nan")
            continue
        a = sub[c].to_numpy().astype(np.float64)
        b = sub["fwd_ret_20d_resid"].to_numpy().astype(np.float64)
        if a.size < 2 or a.std() == 0:
            corrs[c] = float("nan")
        else:
            corrs[c] = float(np.corrcoef(a, b)[0, 1])

    lines: list[str] = []
    lines.append("# M3 — Forward Returns + Factor Residualization Validation")
    lines.append("")
    lines.append(
        "Inputs: M2 setups (`data/interim/setups_with_features.parquet`, "
        f"{n_in:,} rows). Outputs: `data/interim/setups_with_residuals.parquet` "
        f"({n_out:,} rows after dropping setups missing fwd_ret_20d or factor "
        "window data) plus per-row residuals + winsorized "
        "`vol_contraction_ratio_w`."
    )
    lines.append("")
    lines.append(
        "Training window: **2010-01-01 to 2017-12-31**. OOS: 2018-01-01 to "
        "2025-12-31. OOS residuals are computed by applying the trained "
        "coefficients (factor betas + sector dummies + intercept). Year "
        "fixed-effect dummies are encoded only for training years; OOS rows "
        "get all year dummies = 0 (i.e. mapped to the reference year for "
        "scoring). This means OOS residuals carry any year-level drift as "
        "an additive offset, but cross-sectional variation -- which is what "
        "the M4 quantile regression cares about -- is preserved."
    )
    lines.append("")

    # ---- Setup counts by year ----
    lines.append("## Setup counts by year (before / after dropping for missing fwd_ret_20d)")
    lines.append("")
    lines.append("| Year | Input rows | Kept rows | Dropped |")
    lines.append("|---:|---:|---:|---:|")
    for r in year_table.iter_rows(named=True):
        dropped = r["input_rows"] - r["kept_rows"]
        lines.append(f"| {r['year']} | {r['input_rows']:,} | {r['kept_rows']:,} | {dropped:,} |")
    lines.append(f"| **Total** | **{n_in:,}** | **{n_out:,}** | **{n_in - n_out:,}** |")
    lines.append("")

    # ---- Drop reasons ----
    lines.append("## Dropped setups (rolled up by reason)")
    lines.append("")
    if not drops and n_missing_factors == 0:
        lines.append("- None.")
    else:
        from collections import Counter
        reasons = Counter(d["reason"] for d in drops)
        if n_missing_factors:
            reasons[f"factor panel missing date(s) for setup window"] = n_missing_factors
        for reason, n in reasons.most_common():
            lines.append(f"- {reason}: {n:,}")
    lines.append("")

    # ---- fwd_ret_20d distributions ----
    lines.append("## fwd_ret_20d distribution (raw)")
    lines.append("")
    lines.append("| Slice | n | mean | std | p10 | p50 | p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, q in [("training (2010-2017)", train_y_q), ("OOS (2018-2025)", oos_y_q)]:
        lines.append(
            f"| {label} | {q['n']:,} | {q['mean']:+.4f} | {q['std']:.4f} | "
            f"{q['p10']:+.4f} | {q['p50']:+.4f} | {q['p90']:+.4f} |"
        )
    lines.append("")

    lines.append("## fwd_ret_20d_resid distribution (residualized)")
    lines.append("")
    lines.append("| Slice | n | mean | std | p10 | p50 | p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, q in [("training (2010-2017)", train_r_q), ("OOS (2018-2025)", oos_r_q)]:
        lines.append(
            f"| {label} | {q['n']:,} | {q['mean']:+.4f} | {q['std']:.4f} | "
            f"{q['p10']:+.4f} | {q['p50']:+.4f} | {q['p90']:+.4f} |"
        )
    lines.append("")

    sanity_emoji = "PASS" if sanity_pass else "FAIL"
    lines.append(
        f"**Sanity check**: training residual mean = {train_resid_mean:+.6f}. "
        f"{sanity_emoji} (threshold |x| <= 1e-3 -- mechanical given OLS fit "
        "with intercept on the training window)."
    )
    lines.append("")

    # ---- Regression diagnostics ----
    lines.append("## Factor regression on training window")
    lines.append("")
    lines.append(
        f"- Model: `fwd_ret_20d ~ const + Mkt-RF + SMB + HML + UMD + "
        f"sector FE + year FE`"
    )
    lines.append(
        f"- Sector FE: {len(sectors_train)} categories, reference = "
        f"`{sector_ref}` (most-populated). Year FE: {len(years_train)} "
        f"categories, reference = `{year_ref}` (earliest). All factor variables "
        "are 20-trading-day cumulated decimal returns over (t, t+20]."
    )
    lines.append(
        f"- n = {ols_result.nobs:,.0f}, k = {len(col_names):,}, "
        f"R² = {ols_result.rsquared:.4f}, Adj R² = {ols_result.rsquared_adj:.4f}"
    )
    lines.append("")

    # Table: factors + intercept
    lines.append("### Factor coefficients (+ intercept)")
    lines.append("")
    lines.append("| Term | Coef | Std err | t | p |")
    lines.append("|---|---:|---:|---:|---:|")
    factor_terms = ["const"] + list(FACTOR_WINDOW_COLS)
    for row in coef_rows:
        if row["term"] in factor_terms:
            lines.append(
                f"| `{row['term']}` | {row['coef']:+.4f} | "
                f"{row['std_err']:.4f} | {row['t']:+.2f} | {row['p']:.3g} |"
            )
    lines.append("")

    # Sector FE
    lines.append("### Sector fixed effects (deviations from reference)")
    lines.append("")
    lines.append("| Sector | Coef | Std err | t | p |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in coef_rows:
        if row["term"].startswith("sector::"):
            lines.append(
                f"| {row['term'].split('::', 1)[1]} | {row['coef']:+.4f} | "
                f"{row['std_err']:.4f} | {row['t']:+.2f} | {row['p']:.3g} |"
            )
    lines.append("")

    # Year FE
    lines.append("### Year fixed effects (deviations from reference)")
    lines.append("")
    lines.append("| Year | Coef | Std err | t | p |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in coef_rows:
        if row["term"].startswith("year::"):
            lines.append(
                f"| {row['term'].split('::', 1)[1]} | {row['coef']:+.4f} | "
                f"{row['std_err']:.4f} | {row['t']:+.2f} | {row['p']:.3g} |"
            )
    lines.append("")

    # ---- Winsorization ----
    lines.append("## Winsorization of vol_contraction_ratio")
    lines.append("")
    train_vcr = train_df["vol_contraction_ratio"].to_numpy()
    train_vcr_max = float(train_vcr.max()) if train_vcr.size else float("nan")
    n_capped_train = int((train_vcr > winsor_p99).sum())
    full_vcr = out_df["vol_contraction_ratio"].to_numpy()
    n_capped_full = int((full_vcr > winsor_p99).sum())
    lines.append(
        f"- Training-window p{int(WINSOR_Q*100)} = **{winsor_p99:.4f}**. "
        f"Training raw max = {train_vcr_max:.4f}."
    )
    lines.append(
        f"- Capped: {n_capped_train:,} training rows (of {train_df.height:,}, "
        f"{n_capped_train/max(1, train_df.height)*100:.2f}%) and "
        f"{n_capped_full:,} total rows of {out_df.height:,} "
        f"({n_capped_full/max(1, out_df.height)*100:.2f}%)."
    )
    lines.append(
        f"- Output column `vol_contraction_ratio_w` = "
        f"`min(vol_contraction_ratio, {winsor_p99:.4f})`. The raw column is "
        "kept alongside for reference."
    )
    lines.append("")

    # ---- Correlation with M2 features ----
    lines.append("## Correlation of fwd_ret_20d_resid with M2 features (complete-case Pearson)")
    lines.append("")
    lines.append("| Feature | Pearson ρ |")
    lines.append("|---|---:|")
    for c in feature_cols:
        v = corrs[c]
        cell = f"{v:+.4f}" if np.isfinite(v) else "—"
        lines.append(f"| `{c}` | {cell} |")
    lines.append("")
    lines.append(
        "Interpretation note: ρ for `vol_contraction_ratio_w` is the linear "
        "association *at the conditional mean*. The pre-registered hypothesis "
        "is about the **τ=0.90 quantile** vs the τ=0.50 quantile, so a near-"
        "zero linear ρ does not imply the hypothesis fails -- it implies the "
        "mean-effect channel is small."
    )
    lines.append("")

    # ---- Notes ----
    lines.append("## Notes / caveats")
    lines.append("")
    lines.append(
        "- **Residualization is two-pass**. We fit one OLS on the training "
        "subset (2010-2017) with sector + year FE included; we then apply the "
        "fitted coefficient vector to **every** row (training and OOS) to get "
        "fwd_ret_20d_resid. Year-FE dummies are not encoded for OOS years, so "
        "OOS rows are scored as if they were the reference year (2010). The "
        "induced level shift on OOS residuals is constant per OOS year and "
        "doesn't affect cross-sectional inferences."
    )
    lines.append(
        "- **Factor windows use cum-sum subtraction** (`mkt_rf_cum[t+20] - "
        "mkt_rf_cum[t]`). This is exactly the sum of the daily decimal factor "
        "returns over (t, t+20] when the setup date `t` and `fwd_end_date` "
        "are both in the factor panel. If either date isn't, the window sum "
        "is null and the row is dropped from the residualization frame "
        "(reported under \"Dropped setups\" above)."
    )
    lines.append(
        "- **Setups appearing in both strict and loose** (the same "
        "ticker+date) get the **same** fwd_ret_20d, factor windows, and "
        "residual. They differ only in the universe_variant column. This is "
        "intentional -- the variants share the underlying chart event."
    )
    lines.append(
        "- **Determinism**: same input parquets -> same output parquet. The "
        "factor panel is not refetched if it already exists at "
        "`data/factors/ff3_umd_daily.parquet`."
    )
    lines.append("")

    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not OUTPUT_PATH.exists():
        print(f"[M3] factor panel missing at {OUTPUT_PATH}; downloading + parsing...")
        panel = build_factor_panel()
        FACTORS_DIR.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(OUTPUT_PATH)
        print(f"[M3] wrote {OUTPUT_PATH} ({panel.height:,} rows)")
    else:
        print(f"[M3] factor panel found at {OUTPUT_PATH}; skipping rebuild")

    _residualize_main()


if __name__ == "__main__":
    main()
