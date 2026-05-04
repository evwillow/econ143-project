"""
M1 — Universe Filter & Breakout Candidate Detection
====================================================

Design question answers (§M1 + Blueprint §4.1), written before any code.

--- Q1: Universe filter ---

Applied point-in-time at every date t in [2010-01-01, 2025-12-31]:
  - close[t] > $5
  - 20-day rolling avg dollar volume (close*volume) over t-20..t-1 > $5M
    (lagged by one day so today's bar isn't used to qualify itself)
  - >= 252 trading days of history strictly before t (one full year of bars)
  - is_common_stock == True  -- DROPPED: not present in this dataset; logged
    as a caveat in m1_validation.md and deferred to M2/M3 to handle (e.g. by
    joining Polygon ticker metadata).

Ticker-recycling protection (BBBY finding from M0): a ticker that re-appears
after a >= 60 calendar-day gap is treated as a fresh identity for the
days-of-history check. We assign each ticker a `segment_id` that increments at
every >=60d gap, and the history count is reset within each segment. This
prevents the post-bankruptcy SPAC reissue from inheriting the dead company's
2003-2023 history.

--- Q2: Breakout candidate rule ---

Strict variant (the spec-locked headline universe):
  - cross-sectional 12-1 momentum percentile within universe[t] >= 0.90
  - base detected: high pivot in last 60 trading days, at least 5 bars before t;
    swing low in the 60 bars prior to (and not including) the pivot;
    base duration (= pivot_idx - swing_low_idx) in [10, 42] trading days
  - pullback depth = (pivot_close - close[t]) / pivot_close in [0.08, 0.30]
  - close[t] within 15% of trailing 252-day max close
  - 10-day SMA(close) > 20-day SMA > 50-day SMA at date t

Loose variant: same, but momentum percentile >= 0.80 and the 52w-high filter
is dropped. Used for threshold-sensitivity in the writeup.

Pivot rule (kept simple, documented here for the reader):
  - pivot candidate set = closes[max(0, t-60) : t-4]  (so pivot_idx in
    [t-60, t-5] inclusive, "at least 5 days before t")
  - high pivot = argmax of that set
  - swing-low candidate set = closes[max(0, p-60) : p]  (60 bars strictly
    before pivot; pivot itself excluded so base_duration > 0)
  - swing low = argmin of that set
This is a local-extremum rule; it ignores intraday high/low to avoid wick
noise. We use close throughout so the pivot is unambiguous.

--- Q3: Performance ---

Daily files are partitioned by year, so the lazy scan reads each yearly
parquet exactly once. Rolling features (MAs, ADV, mom_12_1, 252-day high) are
computed in polars via .over("ticker", "segment_id") on a sorted frame.
Pivot/swing-low needs the *index* of the windowed extremum, which polars
doesn't expose for windowed argmin/argmax — that step drops to a per-segment
numpy pass using sliding_window_view, which keeps the whole pivot computation
under a minute on the full ~60M-row frame.

Variants share all features. main() computes features once and applies the
two variant filters separately; the spec-mandated detect_setups() does both
end-to-end for callers that want a one-shot interface.

--- Interface ---

Inputs : data/raw/stocks/daily/*/*.parquet  (via DAILY_BARS_GLOB)
Outputs: data/interim/setups.parquet         (one row per qualifying setup)
         reports/m1_validation.md
         reports/m1_sample_setups.csv        (10 random rows for chart QC)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DAILY_BARS_GLOB, REPO_ROOT  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE = date(2010, 1, 1)
END_DATE = date(2025, 12, 31)
# 2y of pre-START_DATE bars is enough to satisfy the 252-day history filter
# for any setup on or after 2010-01-01 (whose ticker existed by 2008).
LOOKBACK_START = date(2008, 1, 1)

OUT_PARQUET = REPO_ROOT / "data" / "interim" / "setups.parquet"
VALIDATION_MD = REPO_ROOT / "reports" / "m1_validation.md"
SAMPLE_CSV = REPO_ROOT / "reports" / "m1_sample_setups.csv"

UNIVERSE_FILTER: dict = {
    "min_close": 5.0,
    "min_adv_20": 5_000_000.0,
    "min_days_history": 252,
}

BASE_PARAMS: dict = {
    "min_base_duration": 10,
    "max_base_duration": 42,
    "min_pullback_pct": 0.08,
    "max_pullback_pct": 0.30,
    "max_dist_52w_high_pct": 0.15,
    "pivot_lookback": 60,
    "pivot_min_lag": 5,
    "swing_lookback": 60,
    "ticker_recycle_gap_days": 60,
}

VARIANTS: dict = {
    "strict": {"momentum_percentile": 0.90, "require_52w_high": True},
    "loose":  {"momentum_percentile": 0.80, "require_52w_high": False},
}


# ---------------------------------------------------------------------------
# Bars + segments
# ---------------------------------------------------------------------------

def _scan_bars(price_df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        price_df.select(["ticker", "date", "open", "high", "low", "close", "volume"])
        .filter((pl.col("date") >= LOOKBACK_START) & (pl.col("date") <= END_DATE))
        .sort(["ticker", "date"])
    )


def _add_segment_id(df: pl.LazyFrame, gap_days: int) -> pl.LazyFrame:
    """Increment segment_id within each ticker at every calendar gap >= gap_days."""
    return df.with_columns(
        (
            (pl.col("date") - pl.col("date").shift(1).over("ticker"))
            .dt.total_days()
            .fill_null(0)
            >= gap_days
        )
        .cast(pl.Int64)
        .cum_sum()
        .over("ticker")
        .alias("segment_id")
    )


# ---------------------------------------------------------------------------
# Rolling features (polars)
# ---------------------------------------------------------------------------

def _add_rolling_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add MAs, ADV, momentum, 252d high, days_in_segment.

    Caller must pre-sort by (ticker, segment_id, date); rolling and shift
    operations rely on that order within each (ticker, segment_id) partition.
    """
    grp = ["ticker", "segment_id"]
    return df.with_columns(
        pl.col("close").rolling_mean(window_size=10).over(grp).alias("ma_10"),
        pl.col("close").rolling_mean(window_size=20).over(grp).alias("ma_20"),
        pl.col("close").rolling_mean(window_size=50).over(grp).alias("ma_50"),
        # ADV uses bars t-20..t-1 only; lag by 1 so today's volume can't
        # qualify today's bar.
        ((pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=20)
            .shift(1)
            .over(grp)
            .alias("adv_20")),
        # 12-1 momentum: log return from t-252 to t-21.
        ((pl.col("close").shift(21) / pl.col("close").shift(252))
            .log()
            .over(grp)
            .alias("mom_12_1")),
        pl.col("close").rolling_max(window_size=252).over(grp).alias("high_252"),
        pl.col("ticker").cum_count().over(grp).alias("days_in_segment"),
    )


# ---------------------------------------------------------------------------
# Pivot features (numpy, vectorized per segment)
# ---------------------------------------------------------------------------

def _pivot_arrays_for_segment(
    closes: np.ndarray,
    pivot_lb: int,
    pivot_min_lag: int,
    swing_lb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For one segment's close series, return (hp_close, hp_idx, bl_idx, base_dur).

    All arrays length n; sentinel -1 / NaN means the row had no valid pivot.
    Indices are LOCAL to the segment.
    """
    n = closes.size
    hp_close = np.full(n, np.nan, dtype=np.float64)
    hp_idx = np.full(n, -1, dtype=np.int64)
    bl_idx = np.full(n, -1, dtype=np.int64)
    base_dur = np.full(n, -1, dtype=np.int64)

    # The smallest index where the pivot window is non-empty: i - pivot_min_lag > 0
    first_pivot_row = pivot_min_lag + 1
    if n <= first_pivot_row:
        return hp_close, hp_idx, bl_idx, base_dur

    pw = pivot_lb - pivot_min_lag + 1  # pivot window size when fully populated

    # ---- Pivot ----
    # Full-window rows: i in [pivot_lb, n).  Window = closes[i-pivot_lb : i-pivot_min_lag+1].
    if n > pivot_lb:
        windows = sliding_window_view(closes, pw)             # (n - pw + 1, pw)
        win_argmax = np.argmax(windows, axis=1)               # (n - pw + 1,)
        rows_full = np.arange(pivot_lb, n)
        win_k = rows_full - pivot_lb
        pivot_idxs = win_k + win_argmax[win_k]
        hp_close[rows_full] = closes[pivot_idxs]
        hp_idx[rows_full] = pivot_idxs

    # Edge rows (small partial windows): i in [first_pivot_row, min(pivot_lb, n)).
    for i in range(first_pivot_row, min(pivot_lb, n)):
        hi = i - pivot_min_lag + 1
        if hi <= 0:
            continue
        offset = int(np.argmax(closes[:hi]))
        hp_close[i] = closes[offset]
        hp_idx[i] = offset

    # ---- Swing low ----
    # Window for pivot p: closes[max(0, p-swing_lb) : p]  (pivot itself excluded).
    valid_rows = np.flatnonzero(hp_idx >= 0)
    if valid_rows.size == 0:
        return hp_close, hp_idx, bl_idx, base_dur

    pivots = hp_idx[valid_rows]
    full_mask = pivots >= swing_lb
    full_rows = valid_rows[full_mask]
    full_pivots = pivots[full_mask]
    if full_rows.size > 0 and n >= swing_lb:
        sw_windows = sliding_window_view(closes, swing_lb)    # (n - swing_lb + 1, swing_lb)
        sw_argmin = np.argmin(sw_windows, axis=1)
        sw_k = full_pivots - swing_lb
        low_idxs = sw_k + sw_argmin[sw_k]
        bl_idx[full_rows] = low_idxs
        base_dur[full_rows] = full_pivots - low_idxs

    partial_rows = valid_rows[~full_mask]
    partial_pivots = pivots[~full_mask]
    for r, p in zip(partial_rows.tolist(), partial_pivots.tolist()):
        if p == 0:
            continue  # no swing-low candidates strictly before pivot
        offset = int(np.argmin(closes[:p]))
        bl_idx[r] = offset
        base_dur[r] = p - offset

    return hp_close, hp_idx, bl_idx, base_dur


def _add_pivot_features(df: pl.DataFrame, base_params: dict) -> pl.DataFrame:
    """Materialize per-segment pivot/base columns into df.

    Adds: high_pivot_close (Float64), base_end_date (Date), base_start_date
    (Date), base_duration_days (Int64).
    """
    df = df.sort(["ticker", "segment_id", "date"])
    n = df.height
    closes = df["close"].to_numpy()
    tickers = df["ticker"].to_numpy()
    seg_ids = df["segment_id"].to_numpy()

    # Run-length boundaries on (ticker, segment_id).
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (tickers[1:] != tickers[:-1]) | (seg_ids[1:] != seg_ids[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)

    hp_close_full = np.full(n, np.nan, dtype=np.float64)
    hp_idx_full = np.full(n, -1, dtype=np.int64)
    bl_idx_full = np.full(n, -1, dtype=np.int64)
    base_dur_full = np.full(n, -1, dtype=np.int64)

    pivot_lb = base_params["pivot_lookback"]
    pivot_min_lag = base_params["pivot_min_lag"]
    swing_lb = base_params["swing_lookback"]

    for s, e in zip(starts.tolist(), ends.tolist()):
        hp_c, hp_i, bl_i, bd = _pivot_arrays_for_segment(
            closes[s:e], pivot_lb, pivot_min_lag, swing_lb
        )
        hp_close_full[s:e] = hp_c
        hp_idx_full[s:e] = np.where(hp_i >= 0, hp_i + s, -1)
        bl_idx_full[s:e] = np.where(bl_i >= 0, bl_i + s, -1)
        base_dur_full[s:e] = bd

    # Look up dates without materializing 60M python date objects.
    dates_np = df["date"].to_numpy().astype("datetime64[D]")
    nat = np.datetime64("NaT", "D")
    hp_dates = np.where(hp_idx_full >= 0, dates_np[hp_idx_full.clip(min=0)], nat)
    bl_dates = np.where(bl_idx_full >= 0, dates_np[bl_idx_full.clip(min=0)], nat)

    return df.with_columns(
        pl.Series("high_pivot_close", hp_close_full, dtype=pl.Float64),
        pl.Series("base_end_date", hp_dates).cast(pl.Date),
        pl.Series("base_start_date", bl_dates).cast(pl.Date),
        pl.Series("base_duration_days", base_dur_full, dtype=pl.Int64),
    )


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def _compute_features(price_df: pl.LazyFrame, base_params: dict) -> pl.DataFrame:
    """Load -> segment -> rolling -> pivot -> derive. Returns eager DataFrame."""
    bars = _scan_bars(price_df)
    bars = _add_segment_id(bars, base_params["ticker_recycle_gap_days"])
    bars = _add_rolling_features(bars)
    materialized = bars.collect()
    materialized = _add_pivot_features(materialized, base_params)
    return materialized.with_columns(
        ((pl.col("high_pivot_close") - pl.col("close")) / pl.col("high_pivot_close"))
        .alias("pullback_pct"),
        ((pl.col("high_252") - pl.col("close")) / pl.col("high_252"))
        .alias("dist_52w_high_pct"),
    )


# ---------------------------------------------------------------------------
# Variant filters
# ---------------------------------------------------------------------------

def _apply_variant(
    features: pl.DataFrame,
    universe_filter: dict,
    base_params: dict,
    variant: str,
) -> pl.DataFrame:
    """Apply universe filter + variant-specific breakout rules; emit setups."""
    cfg = VARIANTS[variant]

    # Step 1: in-window universe (note: days_in_segment is 1-indexed including
    # today, so subtract 1 to get strictly-prior-bar count).
    in_universe = (
        features.lazy()
        .filter(pl.col("date") >= START_DATE)
        .filter(pl.col("date") <= END_DATE)
        .filter(pl.col("close") > universe_filter["min_close"])
        .filter(pl.col("adv_20") > universe_filter["min_adv_20"])
        .filter(
            (pl.col("days_in_segment") - 1) >= universe_filter["min_days_history"]
        )
        .filter(pl.col("mom_12_1").is_not_null())
        .collect()
    )

    # Step 2: cross-sectional momentum percentile within in_universe per date.
    in_universe = in_universe.with_columns(
        pl.col("mom_12_1").rank(method="average").over("date").alias("_rank"),
        pl.len().over("date").alias("_n_per_date"),
    ).with_columns(
        (pl.col("_rank") / pl.col("_n_per_date")).alias("mom_pct")
    ).drop(["_rank", "_n_per_date"])

    # Step 3: variant rules.
    rules = (
        (pl.col("mom_pct") >= cfg["momentum_percentile"])
        & (pl.col("base_duration_days").is_between(
            base_params["min_base_duration"], base_params["max_base_duration"]
        ))
        & (pl.col("pullback_pct").is_between(
            base_params["min_pullback_pct"], base_params["max_pullback_pct"]
        ))
        & (pl.col("ma_10") > pl.col("ma_20"))
        & (pl.col("ma_20") > pl.col("ma_50"))
    )
    if cfg["require_52w_high"]:
        rules = rules & (
            pl.col("dist_52w_high_pct") <= base_params["max_dist_52w_high_pct"]
        )

    setups = in_universe.filter(rules).with_columns(
        pl.lit(variant).alias("universe_variant")
    )

    return setups.select([
        "ticker", "date", "universe_variant",
        "mom_12_1", "mom_pct",
        "base_start_date", "base_end_date", "base_duration_days",
        "pullback_pct", "dist_52w_high_pct",
        "ma_10", "ma_20", "ma_50",
        "close", "adv_20",
    ])


def detect_setups(
    price_df: pl.LazyFrame,
    universe_filter: dict,
    base_params: dict,
    variant: str,
) -> pl.DataFrame:
    """Spec-mandated public interface. End-to-end per variant.

    For both variants, prefer to compute features once and call _apply_variant
    twice — that's what main() does. detect_setups exists for callers that
    want a single self-contained call.
    """
    features = _compute_features(price_df, base_params)
    return _apply_variant(features, universe_filter, base_params, variant)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _write_validation(features: pl.DataFrame, setups: pl.DataFrame) -> None:
    setups = setups.with_columns(pl.col("date").dt.year().alias("year"))

    by_year_variant = (
        setups.group_by(["year", "universe_variant"])
        .agg(pl.len().alias("n"))
        .sort(["year", "universe_variant"])
    )
    by_year: dict[int, dict[str, int]] = {}
    for row in by_year_variant.to_dicts():
        by_year.setdefault(row["year"], {})[row["universe_variant"]] = row["n"]

    top_lines: dict[str, list[tuple[str, int]]] = {}
    for variant in ("strict", "loose"):
        top = (
            setups.filter(pl.col("universe_variant") == variant)
            .group_by("ticker").agg(pl.len().alias("n"))
            .sort("n", descending=True).head(10)
        )
        top_lines[variant] = [(r["ticker"], r["n"]) for r in top.to_dicts()]

    sample_n = min(10, setups.height)
    sample = setups.sample(n=sample_n, seed=42).sort("date") if sample_n else setups
    sample_out = sample.select([
        "ticker", "date", "universe_variant",
        "mom_12_1", "mom_pct", "base_start_date", "base_end_date",
        "base_duration_days", "pullback_pct", "dist_52w_high_pct",
        "ma_10", "ma_20", "ma_50", "close", "adv_20",
    ])
    sample_out.write_csv(SAMPLE_CSV)

    # Recycled-ticker count (segments per ticker > 1)
    seg_counts = features.group_by("ticker").agg(pl.col("segment_id").max().alias("max_seg"))
    recycled_count = int(seg_counts.filter(pl.col("max_seg") > 0).height)
    total_tickers = int(seg_counts.height)

    # Regime sanity check (strict universe)
    strict_yearly = {y: vals.get("strict", 0) for y, vals in by_year.items()}
    s2018 = strict_yearly.get(2018, 0)
    s2020 = strict_yearly.get(2020, 0)
    s2021 = strict_yearly.get(2021, 0)
    s2022 = strict_yearly.get(2022, 0)
    flags: list[str] = []
    if (s2020 + s2021) <= 2 * s2018:
        flags.append(
            f"WARN: strict 2020+2021 ({s2020 + s2021:,}) not heavier than 2x 2018 ({2*s2018:,})."
        )
    if s2022 > s2018:
        flags.append(
            f"WARN: strict 2022 ({s2022:,}) > strict 2018 ({s2018:,})."
        )

    strict_total = int(setups.filter(pl.col("universe_variant") == "strict").height)
    loose_total = int(setups.filter(pl.col("universe_variant") == "loose").height)

    lines: list[str] = []
    lines.append("# M1 — Universe & Breakout Detection Validation")
    lines.append("")
    lines.append(f"- Feature table: {features.height:,} rows ({total_tickers:,} tickers)")
    lines.append(
        f"- Tickers split into >1 segment by the {BASE_PARAMS['ticker_recycle_gap_days']}d "
        f"recycling gap: **{recycled_count:,}** ({recycled_count/total_tickers*100:.2f}%)"
    )
    lines.append(f"- Strict total setups: **{strict_total:,}**")
    lines.append(f"- Loose total setups:  **{loose_total:,}**")
    lines.append("")

    lines.append("## Setups by year x variant")
    lines.append("")
    lines.append("| Year | strict | loose |")
    lines.append("|---:|---:|---:|")
    for y in sorted(by_year):
        s = by_year[y].get("strict", 0)
        l = by_year[y].get("loose", 0)
        lines.append(f"| {y} | {s:,} | {l:,} |")
    lines.append("")

    lines.append("## Top 10 most-frequent setup tickers per variant")
    lines.append("")
    for variant in ("strict", "loose"):
        lines.append(f"### {variant}")
        lines.append("")
        lines.append("| Ticker | Setups |")
        lines.append("|---|---:|")
        for tk, n in top_lines[variant]:
            lines.append(f"| {tk} | {n:,} |")
        lines.append("")

    lines.append("## Regime sanity check (strict)")
    lines.append("")
    lines.append(f"- 2018: {s2018:,}")
    lines.append(f"- 2020: {s2020:,}")
    lines.append(f"- 2021: {s2021:,}")
    lines.append(f"- 2022: {s2022:,}")
    lines.append("")
    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("- ok: 2020+2021 heavier than 2018 and 2022 lighter than 2018.")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **`is_common_stock` filter not applied**: the daily-bars schema "
        "(ticker, date, OHLCV, transactions) has no security-type field. The "
        "universe therefore includes ETFs, ADRs, units, warrants, and other "
        "non-common-stock issuers. M2/M3 should either join Polygon ticker "
        "metadata or accept the contamination as a documented limitation."
    )
    lines.append(
        "- Pivot rule uses close-based extrema (no intraday H/L) for "
        "unambiguous argmax/argmin. Will diverge from chart-based pivot "
        "detection that uses high/low wicks."
    )
    lines.append("")

    lines.append("## 10 random sample setups (in `m1_sample_setups.csv`)")
    lines.append("")
    lines.append(
        "| Ticker | Date | Variant | mom_12_1 | mom_pct | base_dur | "
        "pullback_pct | dist_52w_high_pct | close |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in sample.to_dicts():
        lines.append(
            f"| {r['ticker']} | {r['date']} | {r['universe_variant']} | "
            f"{r['mom_12_1']:.4f} | {r['mom_pct']:.3f} | "
            f"{r['base_duration_days']} | {r['pullback_pct']:.4f} | "
            f"{r['dist_52w_high_pct']:.4f} | {r['close']:.2f} |"
        )
    lines.append("")

    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[M1] strict total: {strict_total:,}")
    print(f"[M1] loose total:  {loose_total:,}")
    print(f"[M1] recycled tickers: {recycled_count:,} / {total_tickers:,}")
    for f in flags:
        print(f"[M1] {f}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("[M1] scanning daily bars and computing features...")
    price_df = pl.scan_parquet(DAILY_BARS_GLOB)
    features = _compute_features(price_df, BASE_PARAMS)
    print(f"[M1] feature table: {features.height:,} rows")

    setup_frames = []
    for variant in ("strict", "loose"):
        print(f"[M1] applying variant: {variant}")
        s = _apply_variant(features, UNIVERSE_FILTER, BASE_PARAMS, variant)
        print(f"[M1]   {variant}: {s.height:,} setups")
        setup_frames.append(s)

    all_setups = pl.concat(setup_frames)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    all_setups.write_parquet(OUT_PARQUET)
    print(f"[M1] wrote {OUT_PARQUET}")

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    _write_validation(features, all_setups)
    print(f"[M1] wrote {VALIDATION_MD}")
    print(f"[M1] wrote {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
