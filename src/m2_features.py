"""
M2 — per-setup feature computation.

Inputs
------
data/interim/setups.parquet                 — M1 output (Round 2 detector)
breakoutStudyTool/data/pipeline/stocks/.../*.parquet — daily OHLCV bars
data/interim/reference/yfinance_types.parquet       — for sector lookup
data/interim/spy_daily.parquet              — cached SPY (fetched once if absent)

Outputs
-------
data/interim/setups_with_features.parquet   — M1 setups + 5 new feature columns
reports/m2_validation.md                    — NaN audit, distributions, sanity-check

Features (all computed over the consolidation window
[cons_start_date, cons_end_date] inclusive, which equals [pivot_idx, t-1]
in M1's indexing — i.e. excludes the breakout day t):

  vol_contraction_ratio   = mean(volume, second half) / mean(volume, first half)
                            Half-split: floor(N/2) | ceil(N/2). Always split in
                            half exactly, regardless of base length (per the
                            task brief's locked design decision).
  adr_pct                 = mean( (high - low) / close ) over base bars.
                            Decimal form (0.04 == 4%).
  base_duration_days      = copy of cons_duration_days from M1.
  rs_slope_vs_spy         = OLS slope of log(stock_close / spy_close) on day
                            index 0..N-1 over the base. Days with missing SPY
                            data are skipped but original day indices are
                            preserved (so the regression sees gapped x's).
  sector                  = yfinance .info['sector'] from the M1-built cache.
                            Missing or empty -> 'Unknown'.

If >5% of setups have NaN in any of the four numeric features, the parquet is
NOT written and the validation report flags the failure so the user can
investigate. (Per task brief.)

Deterministic. The only network call is the one-time SPY fetch when
data/interim/spy_daily.parquet is absent.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DAILY_BARS_GLOB,
    REPO_ROOT,
    TICKER_TYPES_PARQUET,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SETUPS_PARQUET = REPO_ROOT / "data" / "interim" / "setups.parquet"
OUT_PARQUET = REPO_ROOT / "data" / "interim" / "setups_with_features.parquet"
VALIDATION_MD = REPO_ROOT / "reports" / "m2_validation.md"
SPY_PARQUET = REPO_ROOT / "data" / "interim" / "spy_daily.parquet"

SPY_FETCH_START = date(2008, 1, 1)
SPY_FETCH_END   = date(2026, 3, 1)

NUMERIC_FEATURES = [
    "vol_contraction_ratio",
    "adr_pct",
    "base_duration_days",
    "rs_slope_vs_spy",
]

NAN_FLAG_THRESHOLD = 0.05  # >5% NaN in any feature -> don't write parquet

# Qullamaggie expected ranges, for the sanity-check section of the report.
EXPECTED_RANGES = {
    "vol_contraction_ratio": (0.4, 1.5),
    "adr_pct":               (0.02, 0.08),
}


# ---------------------------------------------------------------------------
# SPY loader (cached fetch)
# ---------------------------------------------------------------------------

def _load_or_fetch_spy() -> pl.DataFrame:
    """Return SPY daily DataFrame with columns (date: pl.Date, close: pl.Float64),
    sorted by date. Caches to data/interim/spy_daily.parquet on first run."""
    if SPY_PARQUET.exists():
        df = pl.read_parquet(SPY_PARQUET)
        print(f"[M2] loaded SPY from cache: {df.height:,} bars "
              f"({df['date'].min()} to {df['date'].max()})", flush=True)
        return df

    print(f"[M2] SPY cache absent; fetching via yfinance "
          f"({SPY_FETCH_START} to {SPY_FETCH_END})...", flush=True)
    import yfinance as yf  # local import: only needed when fetching
    raw = yf.download(
        "SPY",
        start=SPY_FETCH_START.isoformat(),
        end=SPY_FETCH_END.isoformat(),
        auto_adjust=True,           # split/dividend-adjusted to match daily bars
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise SystemExit("[M2] SPY fetch returned empty data")
    # yfinance returns multi-index columns when only one ticker; flatten.
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = raw.columns.get_level_values(0)
    spy = pl.from_pandas(raw.reset_index()).select(
        pl.col("Date").cast(pl.Date).alias("date"),
        pl.col("Close").cast(pl.Float64).alias("close"),
    ).sort("date")
    SPY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    spy.write_parquet(SPY_PARQUET)
    print(f"[M2] SPY fetched + cached: {spy.height:,} bars "
          f"({spy['date'].min()} to {spy['date'].max()})", flush=True)
    return spy


# ---------------------------------------------------------------------------
# Per-ticker bar arrays
# ---------------------------------------------------------------------------

def _load_ticker_bars(tickers: list[str]) -> dict[str, dict]:
    """Return ticker -> {dates, opens, highs, lows, closes, volumes} dict
    of numpy arrays. Sorted by date within each ticker."""
    print(f"[M2] loading daily bars for {len(tickers):,} setup tickers...", flush=True)
    bars = (
        pl.scan_parquet(DAILY_BARS_GLOB)
          .filter(pl.col("ticker").is_in(tickers))
          .sort(["ticker", "date"])
          .collect()
    )
    print(f"[M2] loaded {bars.height:,} bar rows for {bars['ticker'].n_unique():,} tickers", flush=True)

    out: dict[str, dict] = {}
    for tk_df in bars.partition_by("ticker"):
        tk = tk_df["ticker"][0]
        out[tk] = {
            "dates":  tk_df["date"].to_numpy().astype("datetime64[D]"),
            "opens":  tk_df["open"].to_numpy().astype(np.float64),
            "highs":  tk_df["high"].to_numpy().astype(np.float64),
            "lows":   tk_df["low"].to_numpy().astype(np.float64),
            "closes": tk_df["close"].to_numpy().astype(np.float64),
            "volumes": tk_df["volume"].to_numpy().astype(np.float64),
        }
    return out


# ---------------------------------------------------------------------------
# Sector lookup
# ---------------------------------------------------------------------------

def _load_sector_map() -> dict[str, str]:
    if not TICKER_TYPES_PARQUET.exists():
        print("[M2] WARNING: yfinance_types.parquet missing; all sectors -> 'Unknown'", flush=True)
        return {}
    df = pl.read_parquet(TICKER_TYPES_PARQUET)
    if "sector" not in df.columns:
        print("[M2] WARNING: yfinance_types.parquet has no 'sector' column", flush=True)
        return {}
    out: dict[str, str] = {}
    for r in df.select(["ticker", "sector"]).iter_rows():
        tk, sec = r
        if sec and isinstance(sec, str) and sec.strip():
            out[tk] = sec
    return out


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope (closed form). Returns NaN when x has zero variance."""
    if x.size < 2:
        return float("nan")
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / denom)


def _compute_one(
    cons_start: date,
    cons_end: date,
    td: dict,
    spy_map: dict,
) -> tuple[float, float, float, str | None]:
    """Compute (vol_contraction_ratio, adr_pct, rs_slope_vs_spy, drop_reason)
    for one setup. drop_reason is None on success."""
    dates = td["dates"]
    cs = np.datetime64(cons_start, "D")
    ce = np.datetime64(cons_end, "D")
    idx_lo = int(np.searchsorted(dates, cs, side="left"))
    idx_hi = int(np.searchsorted(dates, ce, side="right"))  # exclusive
    if idx_hi <= idx_lo:
        return float("nan"), float("nan"), float("nan"), "no bars in cons window"

    cw_dates  = dates[idx_lo:idx_hi]
    cw_highs  = td["highs"][idx_lo:idx_hi]
    cw_lows   = td["lows"][idx_lo:idx_hi]
    cw_closes = td["closes"][idx_lo:idx_hi]
    cw_vols   = td["volumes"][idx_lo:idx_hi]
    n = idx_hi - idx_lo

    # vol_contraction_ratio
    half = n // 2
    if half >= 1 and (n - half) >= 1:
        v_first = cw_vols[:half].mean()
        v_second = cw_vols[half:].mean()
        vcr = float(v_second / v_first) if v_first > 0 else float("nan")
    else:
        vcr = float("nan")

    # adr_pct
    if (cw_closes > 0).all():
        adr = float(((cw_highs - cw_lows) / cw_closes).mean())
    else:
        adr = float("nan")

    # rs_slope_vs_spy
    spy_closes = np.array(
        [spy_map.get(d, np.nan) for d in cw_dates], dtype=np.float64
    )
    valid = np.isfinite(spy_closes) & (spy_closes > 0) & (cw_closes > 0)
    if valid.sum() >= 3:
        log_rs = np.log(cw_closes[valid] / spy_closes[valid])
        x_full = np.arange(n, dtype=np.float64)
        slope = _ols_slope(x_full[valid], log_rs)
    else:
        slope = float("nan")

    return vcr, adr, slope, None


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _quantiles(s: pl.Series) -> dict[str, float]:
    """Return {min, p10, p25, p50, p75, p90, max} for non-null values."""
    s = s.drop_nulls()
    if s.is_empty():
        return {k: float("nan") for k in ["min", "p10", "p25", "p50", "p75", "p90", "max"]}
    return {
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def _correlation_matrix(df: pl.DataFrame, cols: list[str]) -> dict:
    """Pairwise Pearson correlation on rows where ALL listed columns are non-null."""
    sub = df.select(cols).drop_nulls()
    if sub.is_empty():
        return {(a, b): float("nan") for a in cols for b in cols}
    arr = np.column_stack([sub[c].to_numpy().astype(np.float64) for c in cols])
    cm = np.corrcoef(arr, rowvar=False)
    out = {}
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            out[(a, b)] = float(cm[i, j])
    return out


def _write_validation(
    setups_in: pl.DataFrame,
    out_df: pl.DataFrame,
    drops: list,
    nan_flagged: list,
    spy_info: dict,
    splits_note: str,
) -> None:
    n_in = setups_in.height
    n_out = out_df.height

    lines: list[str] = []
    lines.append("# M2 — Per-Setup Feature Validation")
    lines.append("")
    lines.append(
        "Inputs: M1 setups (`data/interim/setups.parquet`, "
        f"{n_in:,} rows: "
        f"{int((setups_in['universe_variant']=='strict').sum()):,} strict + "
        f"{int((setups_in['universe_variant']=='loose').sum()):,} loose)."
    )
    lines.append("")
    lines.append(
        "Outputs: 5 features added per row "
        "(`vol_contraction_ratio`, `adr_pct`, `base_duration_days`, "
        "`rs_slope_vs_spy`, `sector`). All M1 columns preserved."
    )
    lines.append("")
    lines.append(f"- Output rows: **{n_out:,}** ({n_in - n_out:,} dropped — see table at bottom).")
    lines.append(
        f"- SPY source: `{spy_info['path']}` ({spy_info['rows']:,} bars, "
        f"{spy_info['min_date']} to {spy_info['max_date']}, "
        f"`auto_adjust=True` to match split-adjusted daily bars)."
    )
    lines.append("")
    if splits_note:
        lines.append(f"> {splits_note}")
        lines.append("")

    # ---- NaN audit ----
    lines.append("## NaN counts per feature, by universe variant")
    lines.append("")
    lines.append("| Feature | strict NaN | strict total | strict NaN % | loose NaN | loose total | loose NaN % | overall NaN % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for col in NUMERIC_FEATURES + ["sector"]:
        s_strict = out_df.filter(pl.col("universe_variant") == "strict")[col]
        s_loose  = out_df.filter(pl.col("universe_variant") == "loose")[col]
        if col == "sector":
            n_s_nan = int((s_strict == "Unknown").sum())
            n_l_nan = int((s_loose  == "Unknown").sum())
            n_total_nan = int((out_df[col] == "Unknown").sum())
        else:
            n_s_nan = int(s_strict.is_null().sum() + s_strict.is_nan().sum())
            n_l_nan = int(s_loose.is_null().sum()  + s_loose.is_nan().sum())
            n_total_nan = int(out_df[col].is_null().sum() + out_df[col].is_nan().sum())
        n_s = s_strict.len()
        n_l = s_loose.len()
        s_pct = n_s_nan / n_s * 100 if n_s else 0.0
        l_pct = n_l_nan / n_l * 100 if n_l else 0.0
        o_pct = n_total_nan / n_out * 100 if n_out else 0.0
        suffix = " (sector counts 'Unknown' as missing)" if col == "sector" else ""
        lines.append(
            f"| {col}{suffix} | {n_s_nan:,} | {n_s:,} | {s_pct:.2f}% | "
            f"{n_l_nan:,} | {n_l:,} | {l_pct:.2f}% | {o_pct:.2f}% |"
        )
    lines.append("")

    if nan_flagged:
        lines.append("### NaN threshold breach")
        lines.append("")
        lines.append(
            f"Per task brief, parquet output is **withheld** when any feature "
            f"exceeds {NAN_FLAG_THRESHOLD*100:.0f}% NaN. Breached features:"
        )
        for col, pct in nan_flagged:
            lines.append(f"- `{col}`: {pct*100:.2f}% NaN")
        lines.append("")

    # ---- Distribution stats ----
    lines.append("## Distribution stats (numeric features)")
    lines.append("")
    lines.append("| Feature | min | p10 | p25 | p50 | p75 | p90 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for col in NUMERIC_FEATURES:
        q = _quantiles(out_df[col])
        if col == "base_duration_days":
            fmt = lambda v: f"{int(v)}" if np.isfinite(v) else "—"
        else:
            fmt = lambda v: f"{v:.4f}" if np.isfinite(v) else "—"
        lines.append(
            f"| {col} | {fmt(q['min'])} | {fmt(q['p10'])} | {fmt(q['p25'])} | "
            f"{fmt(q['p50'])} | {fmt(q['p75'])} | {fmt(q['p90'])} | {fmt(q['max'])} |"
        )
    lines.append("")

    # ---- Sanity check vs Qullamaggie expected ranges ----
    lines.append("## Sanity check vs Qullamaggie expected ranges")
    lines.append("")
    lines.append("| Feature | expected | observed (p25–p75) | observed (p10–p90) | verdict |")
    lines.append("|---|---|---|---|---|")
    for col, (lo, hi) in EXPECTED_RANGES.items():
        q = _quantiles(out_df[col])
        in_iqr = (q["p25"] >= lo and q["p75"] <= hi)
        in_p1090 = (q["p10"] >= lo and q["p90"] <= hi)
        if in_iqr:
            verdict = "IQR inside expected"
        elif in_p1090:
            verdict = "p10-p90 inside; IQR drifts"
        else:
            verdict = "distribution shifted vs expected"
        lines.append(
            f"| {col} | [{lo}, {hi}] | "
            f"[{q['p25']:.3f}, {q['p75']:.3f}] | "
            f"[{q['p10']:.3f}, {q['p90']:.3f}] | {verdict} |"
        )
    bdd = _quantiles(out_df["base_duration_days"])
    bdd_verdict = "matches" if (bdd['min'] >= 10 and bdd['max'] <= 42) else "outside [10,42]"
    lines.append(
        f"| base_duration_days | matches M1 cons range [10, 42] | "
        f"[{int(bdd['p25'])}, {int(bdd['p75'])}] | "
        f"[{int(bdd['p10'])}, {int(bdd['p90'])}] | {bdd_verdict} |"
    )
    lines.append("")

    # ---- Correlation matrix ----
    lines.append("## Correlation matrix (Pearson, complete-case)")
    lines.append("")
    cm = _correlation_matrix(out_df, NUMERIC_FEATURES)
    lines.append("|  | " + " | ".join(NUMERIC_FEATURES) + " |")
    lines.append("|---|" + "---:|" * len(NUMERIC_FEATURES))
    for a in NUMERIC_FEATURES:
        cells = [f"{cm[(a, b)]:+.3f}" for b in NUMERIC_FEATURES]
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Sector counts ----
    lines.append("## Sector counts")
    lines.append("")
    lines.append("| Sector | strict | loose | total |")
    lines.append("|---|---:|---:|---:|")
    sector_counts = (
        out_df.group_by(["sector", "universe_variant"]).agg(pl.len().alias("n"))
        .pivot(values="n", index="sector", on="universe_variant")
        .with_columns(
            pl.col("strict").fill_null(0) if "strict" in out_df["universe_variant"].unique().to_list() else pl.lit(0).alias("strict"),
            pl.col("loose").fill_null(0)  if "loose"  in out_df["universe_variant"].unique().to_list() else pl.lit(0).alias("loose"),
        )
        .with_columns((pl.col("strict") + pl.col("loose")).alias("total"))
        .sort("total", descending=True)
    )
    for r in sector_counts.iter_rows(named=True):
        lines.append(
            f"| {r['sector']} | {int(r['strict']):,} | "
            f"{int(r['loose']):,} | {int(r['total']):,} |"
        )
    lines.append("")

    # ---- Drops ----
    lines.append("## Dropped setups")
    lines.append("")
    if not drops:
        lines.append("- None.")
    else:
        from collections import Counter
        reason_counts = Counter(d["reason"] for d in drops)
        lines.append(f"- **Total dropped: {len(drops):,}** (vs {n_in:,} input rows)")
        for reason, n in reason_counts.most_common():
            lines.append(f"  - {reason}: {n:,}")
        lines.append("")
        lines.append("| Ticker | Setup date | Variant | Reason |")
        lines.append("|---|---|---|---|")
        for d in drops[:50]:
            lines.append(
                f"| {d['ticker']} | {d['date']} | "
                f"{d['universe_variant']} | {d['reason']} |"
            )
        if len(drops) > 50:
            lines.append(f"| _... {len(drops) - 50:,} more rows ..._ |  |  |  |")
    lines.append("")

    # ---- Notes ----
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- **Half-split rule** (vol_contraction_ratio): always split base in "
        "half exactly (`floor(N/2)` / `ceil(N/2)`), regardless of "
        "`cons_duration_days`. This interprets the task brief's "
        "\"`If cons_duration_days < 20: split base in half exactly`\" as "
        "applying universally — the `<20` qualifier emphasizes that no "
        "minimum is required, and no separate rule is specified for `>=20`. "
        "If a different rule was intended (e.g. last-week-vs-first-week, or "
        "quintile splits) for longer bases, this needs revisiting before M3."
    )
    lines.append(
        "- **rs_slope_vs_spy**: x is the original day index in [0, N-1] over "
        "the base window. Days where SPY data is missing (e.g. exchange "
        "holidays the stock trades but SPY doesn't) are skipped from y but "
        "their indices are preserved in x — the regression sees gapped x's. "
        "When fewer than 3 valid bars remain, the slope is NaN."
    )
    lines.append(
        "- **SPY adjustment**: fetched with `auto_adjust=True` so split/dividend "
        "adjustments match the breakoutStudyTool daily bars (which are "
        "split-adjusted). If the daily-bar pipeline ever changes adjustment "
        "convention, refetch SPY."
    )
    lines.append(
        "- **Sector**: from yfinance `.info['sector']` cached at "
        "`data/interim/reference/yfinance_types.parquet`. Tickers whose "
        "yfinance fetch failed (~119 of 1,773 in the M1 cache) get sector = "
        "'Unknown'. M2 doesn't refetch."
    )
    lines.append("")

    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[M2] reading {SETUPS_PARQUET}")
    setups = pl.read_parquet(SETUPS_PARQUET)
    print(f"[M2] {setups.height:,} setups loaded "
          f"({int((setups['universe_variant']=='strict').sum()):,} strict, "
          f"{int((setups['universe_variant']=='loose').sum()):,} loose)")

    spy_df = _load_or_fetch_spy()
    spy_info = {
        "path": str(SPY_PARQUET.relative_to(REPO_ROOT)),
        "rows": spy_df.height,
        "min_date": str(spy_df["date"].min()),
        "max_date": str(spy_df["date"].max()),
    }
    spy_map = {
        np.datetime64(d, "D"): float(c)
        for d, c in zip(spy_df["date"].to_list(), spy_df["close"].to_list())
    }

    sector_map = _load_sector_map()

    unique_tickers = sorted(setups["ticker"].unique().to_list())
    ticker_data = _load_ticker_bars(unique_tickers)

    # ---- Per-setup compute ----
    print(f"[M2] computing features for {setups.height:,} setups...", flush=True)
    vcrs   = np.full(setups.height, np.nan)
    adrs   = np.full(setups.height, np.nan)
    bdds   = np.full(setups.height, np.nan)
    slopes = np.full(setups.height, np.nan)
    sectors: list[str] = []
    drops: list = []

    for i, r in enumerate(setups.iter_rows(named=True)):
        tk = r["ticker"]
        cs = r["cons_start_date"]
        ce = r["cons_end_date"]
        bdd = r["cons_duration_days"]
        bdds[i] = float(bdd)

        sectors.append(sector_map.get(tk) or "Unknown")

        if tk not in ticker_data:
            drops.append({"ticker": tk, "date": r["date"],
                          "universe_variant": r["universe_variant"],
                          "reason": "no daily bars for ticker"})
            continue

        td = ticker_data[tk]
        vcr, adr, slope, drop_reason = _compute_one(cs, ce, td, spy_map)
        vcrs[i] = vcr
        adrs[i] = adr
        slopes[i] = slope
        if drop_reason is not None:
            drops.append({"ticker": tk, "date": r["date"],
                          "universe_variant": r["universe_variant"],
                          "reason": drop_reason})

    out_df = setups.with_columns(
        pl.Series("vol_contraction_ratio", vcrs),
        pl.Series("adr_pct", adrs),
        pl.Series("base_duration_days", bdds.astype(np.int64), dtype=pl.Int64),
        pl.Series("rs_slope_vs_spy", slopes),
        pl.Series("sector", sectors),
    )

    nan_flagged: list = []
    for col in NUMERIC_FEATURES:
        s = out_df[col]
        n_nan = int(s.is_null().sum() + s.is_nan().sum())
        pct = n_nan / out_df.height if out_df.height else 0.0
        if pct > NAN_FLAG_THRESHOLD:
            nan_flagged.append((col, pct))
            print(f"[M2] FLAG: {col} {pct*100:.2f}% NaN > "
                  f"{NAN_FLAG_THRESHOLD*100:.0f}% threshold")

    if nan_flagged:
        print("[M2] NaN threshold breached -- writing validation report only, "
              "NOT writing parquet (per task brief).")
    else:
        OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        out_df.write_parquet(OUT_PARQUET)
        print(f"[M2] wrote {OUT_PARQUET}")

    splits_note = ""
    _write_validation(
        setups_in=setups,
        out_df=out_df,
        drops=drops,
        nan_flagged=nan_flagged,
        spy_info=spy_info,
        splits_note=splits_note,
    )
    print(f"[M2] wrote {VALIDATION_MD}")
    print(f"[M2] strict total: {int((out_df['universe_variant']=='strict').sum()):,}")
    print(f"[M2] loose total:  {int((out_df['universe_variant']=='loose').sum()):,}")
    print(f"[M2] dropped: {len(drops):,}")


if __name__ == "__main__":
    main()
