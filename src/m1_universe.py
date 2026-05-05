"""
M1 — Universe Filter & Qullamaggie Breakout Detection (rewrite per
reports/m1_rule_redesign.md, approved 2026-05-03).

Six-stage pipeline lifted from breakouts.trade's quality_breakouts.py and
adapted to polars/numpy on the EC143 daily parquet:

  Stage 0  Universe filter (close>$5, ADV20>$5M, >=252d history within
           segment, equity-only via yfinance type cache).
  Stage 1  Vectorized "is t a breakout day?" mask:
             close[t]   >  high[t-1]
             high[t]    >  high[t-1]
             volume[t]  >  vol_surge_x * volume_avg_20[t]
             daily_range_pct[t] >= 0.025
             adr_20[t]          >= 0.025
             close[t] > sma_20[t]  AND  close[t] > sma_50[t]
  Stage 2  Per-candidate prior big move (anti-spike pivot):
             pivot = argmax(high[t-90 .. t-15])
             reject pivot if <2 neighboring bars in [pivot+/-5] (excl. pivot)
               have high >= 0.95 * pivot_high  (option (b), §m1_rule_redesign Q6)
             find earliest low_idx in [pivot-60, pivot-15] with
               leg gain in [leg_min_gain, 3.0], duration >=15 trading days,
               up-close ratio >=0.40; snap to local low within +20 bars.
  Stage 3  Consolidation tightness over [pivot, t-1]:
             cons_duration in [10, 42] trading days
             max drop from pivot.high <= 0.30 (allow up to 2 days in (.30,.35]
               and zero days >.35)
             deepest pullback >= cons_min_pullback_pct
             10sma & 20sma rising end-to-start (>=0.98 ratio)
             >=40% of cons bars touch within 2% of any of {10,20,50}-SMA from above
  Stage 4  Pre-breakout extension reject:
             5d gain into t (close[t-1]/open[t-5]) <= 0.08
             close[t-1] / sma_10[t-1] - 1 <= 0.06
             max 2 consecutive "meaningful up-closes" in [t-5, t-1]
               where meaningful = (close>open) AND (close > 1.003 * prev close)
  Stage 5  Variant filter:
             strict: mom_pct>=0.90 AND dist_52w_high_pct<=0.15
             loose:  mom_pct>=0.80
  Stage 6  Per-ticker spacing: 30 trading days minimum between kept setups,
           rank by composite score (cons tightness/pullback/duration/post-bo gain)
           when collisions occur.

Schema notes (per task brief, decision 2):
  KEEP    ticker, date, universe_variant, mom_12_1, mom_pct,
          dist_52w_high_pct, close, adv_20
  RENAME  base_start_date     -> legup_low_date
          base_end_date       -> legup_high_date
          base_duration_days  -> legup_duration_days
          pullback_pct        -> legup_gain_pct  (note: real gain, not bug-named pullback)
  ADD     cons_start_date, cons_end_date, cons_duration_days,
          cons_max_drop_pct, cons_exception_days, ma_touches_pct_in_cons,
          breakout_volume_ratio, breakout_range_pct
  DEMOTED higher_low_count, range_contraction_ratio,
          pct_closes_above_20ma_in_cons -- still in parquet as supplementary
          stats, computed over the CONSOLIDATION window not the leg-up; not
          used in the pass/fail rule.

Limitations:
  - Daily OHLCV only. breakouts.trade additionally gates on first-30-min
    intraday (volume + range scaled to a partial-day bar). We can't replicate
    -- Polygon access cancelled. Documented in reports/m1_validation.md.
  - Relative strength is the within-universe momentum percentile; we did NOT
    add SPY-based 6m RS (decision 3).
  - Episodic-pivot family is out of scope (decision 7).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DAILY_BARS_GLOB,
    NON_COMMON_STOCK_EXCLUSIONS,
    REPO_ROOT,
    TICKER_TYPES_PARQUET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE = date(2010, 1, 1)
END_DATE = date(2025, 12, 31)
LOOKBACK_START = date(2008, 1, 1)

OUT_PARQUET = REPO_ROOT / "data" / "m1_setups.parquet"
VALIDATION_MD = REPO_ROOT / "reports" / "m1_validation.md"
SAMPLE_CSV = REPO_ROOT / "reports" / "m1_sample_setups.csv"

UNIVERSE_FILTER: dict = {
    "min_close": 5.0,
    "min_adv_20": 5_000_000.0,
    "min_days_history": 252,
}

# Default "strict-rule" parameters (post-redesign). The relaxed-rerun pass
# overrides leg_min_gain_pct, vol_surge_x, and cons_min_pullback_pct.
DEFAULT_PARAMS: dict = {
    "ticker_recycle_gap_days": 60,
    # Stage 1
    "vol_surge_x": 1.5,
    "min_daily_range_pct": 0.025,
    "min_adr_20_pct": 0.025,
    # Stage 2
    "pivot_lookback": 90,
    "pivot_min_lag": 15,
    "leg_min_trading_days": 15,
    "leg_max_trading_days": 60,
    "leg_min_gain_pct": 0.35,
    "leg_max_gain_pct": 3.00,
    "leg_min_up_close_ratio": 0.40,
    "anti_spike_window": 5,
    "anti_spike_neighbor_pct": 0.95,
    "anti_spike_min_neighbors": 2,
    # Bug 2 fix (manual QC 2026-05-04): SE/OXY had earnings gap-up bars as
    # pivots that satisfied the neighbor-bar test (price stayed elevated
    # post-gap). Independently reject any pivot bar that opened >15% above
    # the prior close.
    "pivot_max_gap_open_pct": 0.15,
    "snap_low_window": 20,
    # Bug 3 fix (belt-and-suspenders): cons_duration in [10, 42] already
    # enforces this, but add an explicit 60-trading-day cap from
    # legup_high_date to breakout day t.
    "max_legup_high_to_t_trading_days": 60,
    # Stage 3
    # Fix 3 attempted in 2026-05-04 round 3 (15 -> tighten "too short" cons),
    # but Round 3 QC dropped to 2.5/10 so reverted back to 10. See "QC
    # iteration history" in m1_validation.md.
    "cons_min_trading_days": 10,
    "cons_max_trading_days": 42,
    "cons_max_drop_pct": 0.30,
    "cons_exception_drop_pct": 0.35,
    "cons_max_exception_days": 2,
    "cons_min_pullback_pct": 0.04,
    "ma_rising_tol": 0.98,
    "ma_touch_min_pct": 0.40,
    "ma_touch_low_pct": 1.02,
    "ma_touch_close_pct": 0.98,
    # Round 3 (2026-05-04) attempted to add a cons_low_trend_slope reject
    # (PPTA-style declining lows) and a pre_legup_return reject (QTWO/ATYR
    # V-bottom). Round 3 QC dropped to 2.5/10 so both rejects were rolled
    # back. The METRICS are still computed and emitted as informational
    # columns (useful for M2 / writeup) but no longer gate pass/fail.
    "pre_legup_lookback_days": 60,
    # Stage 4
    "pre_extend_5d_max": 0.08,
    "pre_extend_close_vs_sma10_max": 0.06,
    "pre_extend_consec_up_threshold": 1.003,
    "pre_extend_max_consec_up": 2,
    # Stage 5
    "max_dist_52w_high_pct": 0.15,
    # Stage 6
    "spacing_trading_days": 30,
}

RELAXED_OVERRIDES: dict = {
    "vol_surge_x": 1.2,
    "leg_min_gain_pct": 0.20,
    "cons_min_pullback_pct": 0.02,
}

VARIANTS: dict = {
    "strict": {"momentum_percentile": 0.90, "require_52w_high": True},
    "loose":  {"momentum_percentile": 0.80, "require_52w_high": False},
}

# Hardcoded ADR / non-US fallback for tickers that yfinance returns
# `Not Found` for (so country/long_name are unfetchable). Discovered during
# the 2026-05-04 manual QC re-run — yfinance now 404s on ERJ even though it
# trades. Only include tickers we are CONFIDENT are non-common-US-equity.
KNOWN_ADRS_FALLBACK: frozenset[str] = frozenset({
    "ERJ",      # Embraer S.A. (Brazil) -- the QC-flagged case
    "AZUL",     # Azul S.A. (Brazil)
    "BRFS",     # BRF S.A. (Brazil)
    "EBR",      # Centrais Eletricas Brasileiras (Brazil)
    "ELP",      # Companhia Paranaense de Energia (Brazil)
    "VALE.P",   # Vale S.A. preferred (Brazil)
    "IMAB",     # I-Mab Biopharma (China/Cayman)
    "SFUN",     # Fang Holdings (China)
    "YY",       # JOYY Inc. (China)
    "TRQ",      # Turquoise Hill Resources (Canada -- since delisted 2022)
})

PREV_COUNTS = {"strict": 1_115, "loose": 2_741}  # from prior (broken) rule


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
# Feature pipeline (vectorized polars)
# ---------------------------------------------------------------------------

def _compute_features(price_df: pl.LazyFrame, params: dict) -> pl.DataFrame:
    """Add MAs, ADV ($), volume_avg_20 (shares), mom_12_1, high_252,
    daily_range_pct, adr_20, days_in_segment, dist_52w_high_pct.
    Returns sorted eager DataFrame with `_row_idx` column."""
    bars = _scan_bars(price_df)
    bars = _add_segment_id(bars, params["ticker_recycle_gap_days"])
    grp = ["ticker", "segment_id"]
    bars = bars.with_columns(
        pl.col("close").rolling_mean(window_size=10).over(grp).alias("ma_10"),
        pl.col("close").rolling_mean(window_size=20).over(grp).alias("ma_20"),
        pl.col("close").rolling_mean(window_size=50).over(grp).alias("ma_50"),
        ((pl.col("close") * pl.col("volume"))
            .rolling_mean(window_size=20)
            .shift(1)
            .over(grp)
            .alias("adv_20")),
        (pl.col("volume").rolling_mean(window_size=20)
            .shift(1)
            .over(grp)
            .alias("volume_avg_20")),
        ((pl.col("close").shift(21) / pl.col("close").shift(252))
            .log()
            .over(grp)
            .alias("mom_12_1")),
        pl.col("close").rolling_max(window_size=252).over(grp).alias("high_252"),
        pl.col("ticker").cum_count().over(grp).alias("days_in_segment"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("daily_range_pct"),
    )
    bars = bars.with_columns(
        pl.col("daily_range_pct")
            .rolling_mean(window_size=20)
            .shift(1)
            .over(grp)
            .alias("adr_20"),
    )
    bars = bars.with_columns(
        ((pl.col("high_252") - pl.col("close")) / pl.col("high_252"))
            .alias("dist_52w_high_pct"),
        # Shift on the FULL segment, not the universe-filtered frame, so that
        # mid-cons low-volume days don't shift the "prior bar" to 2 days ago.
        pl.col("high").shift(1).over(grp).alias("prev_high"),
    )
    df = bars.collect().sort(["ticker", "segment_id", "date"])
    return df.with_columns(pl.int_range(0, df.height).alias("_row_idx"))


# ---------------------------------------------------------------------------
# Security-type filter (yfinance cache; unchanged from prior version)
# ---------------------------------------------------------------------------

def _load_non_equity_set() -> tuple[set[str], str, dict[str, int]]:
    """Return (non_equity_set, source_label, breakdown).

    Bug 1 fix (manual QC 2026-05-04): yfinance's quoteType=EQUITY includes
    ADRs (e.g. ERJ = Embraer, an SA Brazilian issuer). Layer two ADR
    detection rules on top of the existing quote_type filter:
      - country is non-null and != "United States"  -> foreign issuer
      - long_name (or short_name) matches case-insensitive regex
        "ADR|American Depositary|Sponsored ADR"
    A ticker hits the non-equity set if ANY of the three rules fire.
    """
    if TICKER_TYPES_PARQUET.exists():
        try:
            df = pl.read_parquet(TICKER_TYPES_PARQUET)
            non_eq_quote = set(
                df.filter(pl.col("quote_type") != "EQUITY")["ticker"].to_list()
            )
            adr_country: set[str] = set()
            adr_name: set[str] = set()
            # Word boundaries on \bADR\b avoid false positives like
            # "MaDRigal" matching "ADR".
            adr_re = r"(?i)\bADR\b|American Depositary|Sponsored ADR"
            if "country" in df.columns:
                adr_country = set(
                    df.filter(
                        pl.col("country").is_not_null()
                        & (pl.col("country") != "United States")
                    )["ticker"].to_list()
                )
            name_cols = [c for c in ("long_name", "short_name") if c in df.columns]
            for col in name_cols:
                hits = set(
                    df.filter(
                        pl.col(col).is_not_null()
                        & pl.col(col).str.contains(adr_re)
                    )["ticker"].to_list()
                )
                adr_name |= hits
            non_eq = non_eq_quote | adr_country | adr_name | set(KNOWN_ADRS_FALLBACK)
            breakdown = {
                "cache_size": int(df.height),
                "via_quote_type": len(non_eq_quote),
                "via_country_non_us": len(adr_country),
                "via_name_match_adr": len(adr_name),
                "via_hardcoded_adr_fallback": len(KNOWN_ADRS_FALLBACK),
                "total_non_equity": len(non_eq),
                "added_by_adr_layer": len(non_eq) - len(non_eq_quote),
            }
            label = (
                f"yfinance ({df.height:,} tickers, {len(non_eq):,} non-equity: "
                f"{len(non_eq_quote):,} non-EQUITY + {len(adr_country):,} non-US "
                f"+ {len(adr_name):,} name-match ADR "
                f"+ {len(KNOWN_ADRS_FALLBACK):,} hardcoded-ADR fallback)"
            )
            return non_eq, label, breakdown
        except Exception as e:
            return (
                set(NON_COMMON_STOCK_EXCLUSIONS),
                f"hardcoded fallback (yfinance cache unreadable: {e})",
                {},
            )
    return (
        set(NON_COMMON_STOCK_EXCLUSIONS),
        f"hardcoded fallback ({len(NON_COMMON_STOCK_EXCLUSIONS):,} known ETFs/ETNs)",
        {},
    )


# ---------------------------------------------------------------------------
# Stage 0 + 1: universe filter + breakout-day mask + cross-sectional mom_pct
# ---------------------------------------------------------------------------

def _apply_universe_and_stage1(
    features: pl.DataFrame,
    universe_filter: dict,
    params: dict,
    non_equity_set: set[str],
) -> tuple[pl.DataFrame, dict]:
    """Apply Stage 0 universe + security-type filter, compute cross-sectional
    mom_pct per date, then apply the Stage 1 vectorized breakout-day mask.

    Returns (candidates_df, stats) where candidates_df has all surviving rows
    plus a `mom_pct` column. stats reports stage-by-stage drop counts.
    """
    stats: dict[str, int] = {}

    # Stage 0a: in-window universe (pre security filter)
    pre_sec = (
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
    stats["after_universe_pre_security"] = pre_sec.height
    stats["unique_tickers_pre_security"] = pre_sec["ticker"].n_unique()

    # Stage 0b: security-type filter
    if non_equity_set:
        in_universe = pre_sec.filter(~pl.col("ticker").is_in(list(non_equity_set)))
    else:
        in_universe = pre_sec
    stats["after_security_filter"] = in_universe.height
    stats["unique_tickers_post_security"] = in_universe["ticker"].n_unique()
    stats["bars_dropped_by_security_filter"] = (
        stats["after_universe_pre_security"] - stats["after_security_filter"]
    )
    stats["tickers_dropped_by_security_filter"] = (
        stats["unique_tickers_pre_security"] - stats["unique_tickers_post_security"]
    )

    # Cross-sectional momentum percentile per date (within universe)
    in_universe = in_universe.with_columns(
        pl.col("mom_12_1").rank(method="average").over("date").alias("_rank"),
        pl.len().over("date").alias("_n_per_date"),
    ).with_columns(
        (pl.col("_rank") / pl.col("_n_per_date")).alias("mom_pct")
    ).drop(["_rank", "_n_per_date"])

    # Stage 1: vectorized "is t a breakout day?" mask. `prev_high` was added
    # in _compute_features over the FULL segment so it reflects the actual
    # prior trading day, not the prior bar in the universe-filtered frame.
    in_universe = in_universe.sort(["ticker", "segment_id", "date"])
    candidates = in_universe.filter(
        pl.col("prev_high").is_not_null()
        & (pl.col("close")  > pl.col("prev_high"))
        & (pl.col("high") > pl.col("prev_high"))
        & (pl.col("volume") > params["vol_surge_x"] * pl.col("volume_avg_20"))
        & (pl.col("daily_range_pct") >= params["min_daily_range_pct"])
        & (pl.col("adr_20")          >= params["min_adr_20_pct"])
        & (pl.col("close") > pl.col("ma_20"))
        & (pl.col("close") > pl.col("ma_50"))
    )
    stats["after_stage1_breakout_mask"] = candidates.height

    return candidates, stats


# ---------------------------------------------------------------------------
# Stages 2-4: per-candidate evaluation against the global feature arrays
# ---------------------------------------------------------------------------

@dataclass
class _Arrays:
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    sma_10: np.ndarray
    sma_20: np.ndarray
    sma_50: np.ndarray
    vol_avg_20: np.ndarray
    daily_range_pct: np.ndarray
    dates: np.ndarray  # datetime64[D]
    cum_up: np.ndarray  # per-segment cumulative count of "close>prev_close"
    seg_start_for_row: np.ndarray  # global idx of segment start per row


def _build_arrays(features: pl.DataFrame) -> _Arrays:
    n = features.height
    opens   = features["open"].to_numpy()
    highs   = features["high"].to_numpy()
    lows    = features["low"].to_numpy()
    closes  = features["close"].to_numpy()
    volumes = features["volume"].to_numpy()
    sma_10  = features["ma_10"].to_numpy()
    sma_20  = features["ma_20"].to_numpy()
    sma_50  = features["ma_50"].to_numpy()
    vol_avg = features["volume_avg_20"].to_numpy()
    drange  = features["daily_range_pct"].to_numpy()
    dates_np = features["date"].to_numpy().astype("datetime64[D]")
    tickers = features["ticker"].to_numpy()
    seg_ids = features["segment_id"].to_numpy()

    # Segment boundaries
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (tickers[1:] != tickers[:-1]) | (seg_ids[1:] != seg_ids[:-1])
    starts = np.flatnonzero(change)
    seg_start_for_row = np.empty(n, dtype=np.int64)
    for i in range(starts.size):
        s = starts[i]
        e = starts[i + 1] if i + 1 < starts.size else n
        seg_start_for_row[s:e] = s

    # Per-segment cumulative up-close count.  diff[k] = 1 if closes[k] >
    # closes[k-1] AND k is not a segment start.
    diff = np.zeros(n, dtype=np.int64)
    if n > 1:
        diff[1:] = (closes[1:] > closes[:-1]).astype(np.int64)
    diff[starts] = 0
    cum_up = np.cumsum(diff)

    return _Arrays(
        opens=opens, highs=highs, lows=lows, closes=closes, volumes=volumes,
        sma_10=sma_10, sma_20=sma_20, sma_50=sma_50, vol_avg_20=vol_avg,
        daily_range_pct=drange, dates=dates_np, cum_up=cum_up,
        seg_start_for_row=seg_start_for_row,
    )


def _evaluate_one(t: int, arr: _Arrays, p: dict) -> dict | None:
    seg_start = int(arr.seg_start_for_row[t])

    # ---- Stage 2A: pivot = argmax(high[t-90 .. t-15]) ----
    pivot_lo = max(seg_start, t - p["pivot_lookback"])
    pivot_hi = t - p["pivot_min_lag"] + 1  # exclusive
    if pivot_hi <= pivot_lo:
        return None
    pivot_idx = pivot_lo + int(np.argmax(arr.highs[pivot_lo:pivot_hi]))
    pivot_high = float(arr.highs[pivot_idx])
    if not np.isfinite(pivot_high) or pivot_high <= 0:
        return None

    # ---- Stage 2B: anti-spike (option (b)) ----
    spike_lo = max(seg_start, pivot_idx - p["anti_spike_window"])
    spike_hi = min(t, pivot_idx + p["anti_spike_window"] + 1)  # exclude t (breakout day)
    threshold = pivot_high * p["anti_spike_neighbor_pct"]
    nbr_window = arr.highs[spike_lo:spike_hi]
    nbr_count = int((nbr_window >= threshold).sum()) - 1  # subtract pivot itself
    if nbr_count < p["anti_spike_min_neighbors"]:
        return None

    # ---- Stage 2B': earnings-gap pivot reject (Bug 2 fix). Catches pivots
    # like SE 2019 / OXY 2022 where price gapped >15% on the pivot bar and
    # then drifted enough to satisfy the neighbor-bar test. A real
    # consolidation pivot should rarely gap >15% on its setup-defining bar.
    if pivot_idx > seg_start:
        prev_close = float(arr.closes[pivot_idx - 1])
        if prev_close > 0:
            gap_open_pct = float(arr.opens[pivot_idx]) / prev_close - 1.0
            if gap_open_pct > p["pivot_max_gap_open_pct"]:
                return None

    # ---- Stage 2C: leg-up (earliest valid low_idx, vectorized) ----
    leg_lo = max(seg_start, pivot_idx - p["leg_max_trading_days"])
    leg_hi = pivot_idx - p["leg_min_trading_days"] + 1  # exclusive
    if leg_hi <= leg_lo:
        return None
    rng = np.arange(leg_lo, leg_hi)
    cl = arr.closes[leg_lo:leg_hi]
    safe_cl = np.where(cl > 0, cl, 1.0)
    gains = (pivot_high - cl) / safe_cl
    durations = pivot_idx - rng
    up_counts = arr.cum_up[pivot_idx] - arr.cum_up[rng]
    up_ratios = up_counts / np.where(durations > 0, durations, 1)
    valid = (
        (cl > 0)
        & (gains >= p["leg_min_gain_pct"])
        & (gains <= p["leg_max_gain_pct"])
        & (up_ratios >= p["leg_min_up_close_ratio"])
    )
    if not valid.any():
        return None
    legup_low_idx = leg_lo + int(np.argmax(valid))  # earliest True

    # Snap to local low within +20 bars (still leaving >= leg_min_trading_days
    # before the pivot).
    snap_lo = legup_low_idx
    snap_hi = min(legup_low_idx + p["snap_low_window"] + 1,
                  pivot_idx - p["leg_min_trading_days"] + 1)
    if snap_hi > snap_lo:
        local_off = int(np.argmin(arr.closes[snap_lo:snap_hi]))
        legup_low_idx = snap_lo + local_off
    legup_close_at_low = float(arr.closes[legup_low_idx])
    if legup_close_at_low <= 0:
        return None
    legup_gain = (pivot_high - legup_close_at_low) / legup_close_at_low
    legup_duration = pivot_idx - legup_low_idx

    # ---- Stage 2D: pre-leg-up return (informational only after Round 3
    # rollback). Computed when there's enough segment history; NaN
    # otherwise. NOT used to reject candidates.
    pre_lookback = p["pre_legup_lookback_days"]
    pre_anchor = legup_low_idx - pre_lookback
    pre_legup_return = float("nan")
    if pre_anchor >= seg_start and legup_low_idx > seg_start:
        pre_anchor_close = float(arr.closes[pre_anchor])
        pre_end_close = float(arr.closes[legup_low_idx - 1])
        if pre_anchor_close > 0:
            pre_legup_return = (pre_end_close - pre_anchor_close) / pre_anchor_close

    # ---- Stage 3: consolidation tightness over [pivot, t-1] ----
    cons_lo = pivot_idx
    cons_hi = t  # exclusive
    cons_dur = cons_hi - cons_lo
    if cons_dur < p["cons_min_trading_days"] or cons_dur > p["cons_max_trading_days"]:
        return None
    # Bug 3 belt-and-suspenders: even if cons_max_trading_days were ever
    # widened, never let legup_high precede t by more than this hard cap.
    if cons_dur > p["max_legup_high_to_t_trading_days"]:
        return None

    cons_lows  = arr.lows[cons_lo:cons_hi]
    cons_highs = arr.highs[cons_lo:cons_hi]
    cons_closes = arr.closes[cons_lo:cons_hi]
    cons_opens = arr.opens[cons_lo:cons_hi]
    drops = (pivot_high - cons_lows) / pivot_high
    cons_max_drop = float(drops.max())

    if cons_max_drop > p["cons_max_drop_pct"]:
        exception_days = int(((drops > p["cons_max_drop_pct"]) &
                              (drops <= p["cons_exception_drop_pct"])).sum())
        severe_days = int((drops > p["cons_exception_drop_pct"]).sum())
        if exception_days > p["cons_max_exception_days"] or severe_days > 0:
            return None
    else:
        exception_days = 0

    if cons_max_drop < p["cons_min_pullback_pct"]:
        return None

    # MA-rising check
    sma10_s, sma10_e = arr.sma_10[cons_lo], arr.sma_10[cons_hi - 1]
    sma20_s, sma20_e = arr.sma_20[cons_lo], arr.sma_20[cons_hi - 1]
    rt = p["ma_rising_tol"]
    if (np.isfinite(sma10_s) and np.isfinite(sma10_e) and sma10_s > 0
            and sma10_e < sma10_s * rt):
        return None
    if (np.isfinite(sma20_s) and np.isfinite(sma20_e) and sma20_s > 0
            and sma20_e < sma20_s * rt):
        return None

    # MA touch ratio
    cs10 = arr.sma_10[cons_lo:cons_hi]
    cs20 = arr.sma_20[cons_lo:cons_hi]
    cs50 = arr.sma_50[cons_lo:cons_hi]
    low_pct = p["ma_touch_low_pct"]
    cls_pct = p["ma_touch_close_pct"]

    def _touch(sma):
        return (np.isfinite(sma) & (sma > 0)
                & (cons_lows <= sma * low_pct)
                & (cons_closes >= sma * cls_pct))

    touches = _touch(cs10) | _touch(cs20) | _touch(cs50)
    ma_touches_pct = float(touches.sum() / max(1, touches.size))
    if ma_touches_pct < p["ma_touch_min_pct"]:
        return None

    # ---- Stage 3': cons low-trend slope (informational only after Round 3
    # rollback). OLS slope of lows/pivot_high vs day index over the
    # post-pivot bars. NOT used to reject candidates.
    cons_low_trend_slope = float("nan")
    post_pivot_lo = pivot_idx + 1
    post_pivot_hi = cons_hi  # = t (exclusive)
    if post_pivot_hi - post_pivot_lo >= 3:
        y = arr.lows[post_pivot_lo:post_pivot_hi] / pivot_high
        x = np.arange(y.size, dtype=np.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom > 0:
            cons_low_trend_slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)

    # ---- Stage 4: pre-breakout extension reject ----
    if t - 5 < seg_start:
        return None
    open_t5 = float(arr.opens[t - 5])
    close_tm1 = float(arr.closes[t - 1])
    if open_t5 > 0:
        if (close_tm1 - open_t5) / open_t5 > p["pre_extend_5d_max"]:
            return None
    sma10_tm1 = float(arr.sma_10[t - 1])
    if np.isfinite(sma10_tm1) and sma10_tm1 > 0:
        if (close_tm1 / sma10_tm1 - 1.0) > p["pre_extend_close_vs_sma10_max"]:
            return None
    consec = 0
    max_consec = 0
    thr = p["pre_extend_consec_up_threshold"]
    for k in range(t - 4, t):  # k = t-4 .. t-1
        if k <= seg_start:
            continue
        if arr.closes[k] > arr.opens[k] and arr.closes[k] > thr * arr.closes[k - 1]:
            consec += 1
            if consec > max_consec:
                max_consec = consec
        else:
            consec = 0
    if max_consec > p["pre_extend_max_consec_up"]:
        return None

    # ---- Supplementary stats over the consolidation window ----
    wlen = cons_closes.size
    hl_count = 0
    prev_low = None
    for j in range(2, wlen - 2):
        c = cons_closes[j]
        if (c < cons_closes[j - 2] and c < cons_closes[j - 1]
                and c < cons_closes[j + 1] and c < cons_closes[j + 2]):
            if prev_low is not None and c > prev_low:
                hl_count += 1
            prev_low = c

    rc_ratio = float("nan")
    if wlen >= 2:
        denom_close = np.where(cons_closes > 0, cons_closes, 1.0)
        ranges = (cons_highs - cons_lows) / denom_close
        half = wlen // 2
        if half >= 1 and (wlen - half) >= 1:
            early = ranges[:half].mean()
            late = ranges[half:].mean()
            if early > 0:
                rc_ratio = float(late / early)

    pct_above = float("nan")
    valid_mask = np.isfinite(cs20)
    if valid_mask.any():
        n_valid = int(valid_mask.sum())
        above = int((cons_closes[valid_mask] > cs20[valid_mask]).sum())
        pct_above = above / n_valid

    # ---- Breakout-day stats ----
    vol_avg = float(arr.vol_avg_20[t])
    breakout_volume_ratio = (
        float(arr.volumes[t]) / vol_avg if (np.isfinite(vol_avg) and vol_avg > 0)
        else float("nan")
    )
    breakout_range_pct = float(arr.daily_range_pct[t])

    return {
        "legup_low_date":  arr.dates[legup_low_idx].item(),
        "legup_high_date": arr.dates[pivot_idx].item(),
        "legup_duration_days": int(legup_duration),
        "legup_gain_pct": float(legup_gain),
        "pre_legup_return": float(pre_legup_return),
        "cons_start_date": arr.dates[pivot_idx].item(),
        "cons_end_date":   arr.dates[t - 1].item(),
        "cons_duration_days": int(cons_dur),
        "cons_max_drop_pct": float(cons_max_drop),
        "cons_exception_days": int(exception_days),
        "cons_low_trend_slope": float(cons_low_trend_slope),
        "ma_touches_pct_in_cons": float(ma_touches_pct),
        "breakout_volume_ratio": float(breakout_volume_ratio),
        "breakout_range_pct": float(breakout_range_pct),
        "higher_low_count": int(hl_count),
        "range_contraction_ratio": float(rc_ratio),
        "pct_closes_above_20ma_in_cons": float(pct_above),
    }


def _evaluate_candidates(
    features: pl.DataFrame,
    candidates: pl.DataFrame,
    params: dict,
    arr: _Arrays | None = None,
) -> pl.DataFrame:
    """Run stages 2-4 per candidate. Returns a DataFrame of surviving setups
    enriched with the new schema columns. Drops candidates that fail any
    stage. Pass `arr` to skip rebuilding (used by run_pipeline for
    post-eval diagnostics)."""
    if arr is None:
        arr = _build_arrays(features)

    cand_idxs = candidates["_row_idx"].to_numpy()
    keep_mask = np.zeros(cand_idxs.size, dtype=bool)
    out_records: list[dict] = []

    drop_stage2 = 0
    drop_stage3 = 0
    drop_stage4 = 0

    for k, t in enumerate(cand_idxs.tolist()):
        rec = _evaluate_one(int(t), arr, params)
        if rec is not None:
            keep_mask[k] = True
            out_records.append(rec)

    # Coarse stage attribution: re-run the per-record check is costly. We just
    # report a single "after_stages_2_4" count.
    if not out_records:
        return pl.DataFrame()

    enrichment_schema = {
        "legup_low_date": pl.Date,
        "legup_high_date": pl.Date,
        "legup_duration_days": pl.Int64,
        "legup_gain_pct": pl.Float64,
        "pre_legup_return": pl.Float64,
        "cons_start_date": pl.Date,
        "cons_end_date": pl.Date,
        "cons_duration_days": pl.Int64,
        "cons_max_drop_pct": pl.Float64,
        "cons_exception_days": pl.Int64,
        "cons_low_trend_slope": pl.Float64,
        "ma_touches_pct_in_cons": pl.Float64,
        "breakout_volume_ratio": pl.Float64,
        "breakout_range_pct": pl.Float64,
        "higher_low_count": pl.Int64,
        "range_contraction_ratio": pl.Float64,
        "pct_closes_above_20ma_in_cons": pl.Float64,
    }
    enrichment = pl.DataFrame(out_records, schema=enrichment_schema)
    kept = candidates.filter(pl.Series("_mask", keep_mask)).with_row_index("_keep_idx")
    enrichment = enrichment.with_row_index("_keep_idx")
    return kept.join(enrichment, on="_keep_idx", how="inner").drop("_keep_idx")


# ---------------------------------------------------------------------------
# Stage 5 + 6: variant filter + per-ticker spacing
# ---------------------------------------------------------------------------

def _apply_variant_filter(setups: pl.DataFrame, variant: str, params: dict) -> pl.DataFrame:
    cfg = VARIANTS[variant]
    rule = pl.col("mom_pct") >= cfg["momentum_percentile"]
    if cfg["require_52w_high"]:
        rule = rule & (pl.col("dist_52w_high_pct") <= params["max_dist_52w_high_pct"])
    return setups.filter(rule)


def _quality_score(setups: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Composite score for spacing tiebreaks. Higher = better."""
    max_drop = params["cons_max_drop_pct"]
    return setups.with_columns(
        # tightness: 1 - max_drop / 0.30, clipped
        (1.0 - (pl.col("cons_max_drop_pct") / max_drop)).clip(0.0, 1.0).alias("_tight"),
        # cons length normalized to 60d
        (pl.col("cons_duration_days") / 60.0).clip(0.0, 1.0).alias("_dur"),
        # MA touches
        pl.col("ma_touches_pct_in_cons").clip(0.0, 1.0).alias("_ma"),
        # Volume surge (cap at 5x for scoring)
        (pl.col("breakout_volume_ratio") / 5.0).clip(0.0, 1.0).alias("_vol"),
    ).with_columns(
        (pl.col("_tight") * 0.50
         + pl.col("_dur") * 0.10
         + pl.col("_ma") * 0.20
         + pl.col("_vol") * 0.20).alias("_score")
    ).drop(["_tight", "_dur", "_ma", "_vol"])


def _apply_spacing(setups: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Per-ticker, drop setups within `spacing_trading_days` of an already-kept
    setup. When a collision happens, keep the higher-scoring one. Uses the
    `days_in_segment` column as a trading-day proxy, then for cross-segment
    setups (rare) falls back to calendar days / 1.4 ≈ trading days. Implemented
    with a Python loop per ticker -- fast at this scale."""
    spacing = params["spacing_trading_days"]
    if setups.is_empty():
        return setups
    scored = _quality_score(setups, params)
    rows = scored.sort(["ticker", "date"]).to_dicts()
    keep_idxs: list[int] = []
    by_ticker: dict[str, list[dict]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(r)
    out: list[dict] = []
    for tk, lst in by_ticker.items():
        # Greedy spacing with score-based tiebreak: walk chronologically, but
        # if a higher-scoring later candidate collides with a kept one, swap.
        kept: list[dict] = []
        for r in lst:
            collision_idx = None
            for i, k in enumerate(kept):
                if r["segment_id"] == k["segment_id"]:
                    gap = abs(r["days_in_segment"] - k["days_in_segment"])
                else:
                    gap = abs((r["date"] - k["date"]).days) / 1.4
                if gap < spacing:
                    collision_idx = i
                    break
            if collision_idx is None:
                kept.append(r)
            else:
                if r["_score"] > kept[collision_idx]["_score"]:
                    kept[collision_idx] = r
        out.extend(kept)
    if not out:
        return scored.head(0).drop("_score")
    return pl.DataFrame(out).drop("_score")


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "ticker", "date", "universe_variant",
    "mom_12_1", "mom_pct",
    "legup_low_date", "legup_high_date",
    "legup_duration_days", "legup_gain_pct",
    "pre_legup_return",
    "cons_start_date", "cons_end_date",
    "cons_duration_days", "cons_max_drop_pct", "cons_exception_days",
    "cons_low_trend_slope",
    "ma_touches_pct_in_cons",
    "breakout_volume_ratio", "breakout_range_pct",
    "higher_low_count", "range_contraction_ratio", "pct_closes_above_20ma_in_cons",
    "dist_52w_high_pct", "close", "adv_20",
]


def _project_output(setups: pl.DataFrame, variant: str) -> pl.DataFrame:
    return setups.with_columns(pl.lit(variant).alias("universe_variant")) \
                 .select(OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    features: pl.DataFrame,
    universe_filter: dict,
    params: dict,
    non_equity_set: set[str],
    diagnostic_targets: list[tuple[str, str]] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """One end-to-end pass with the given param dict. Returns (setups, stats).
    setups carries both variants stacked, with universe_variant column.

    `diagnostic_targets` is a list of (ticker, ISO date) pairs to print full
    cons stats for after stages 2-4 -- used to debug specific QC failures.
    """
    print("[M1]   stage 0+1: universe filter + breakout-day mask")
    candidates, stats = _apply_universe_and_stage1(
        features, universe_filter, params, non_equity_set
    )
    print(f"[M1]     -> {candidates.height:,} stage-1 breakout-day candidates")

    arr = _build_arrays(features)

    print(f"[M1]   stages 2-4: per-candidate evaluation ({candidates.height:,} candidates)")
    survivors = _evaluate_candidates(features, candidates, params, arr=arr)
    stats["after_stages_2_4"] = survivors.height
    print(f"[M1]     -> {survivors.height:,} after structural filters")

    # Diagnostic: print stats for target (ticker, date) pairs.
    if diagnostic_targets:
        print("[M1]   diagnostic: cons stats for QC targets")
        from datetime import date as _date
        for tk, dt in diagnostic_targets:
            target_date = _date.fromisoformat(dt)
            cand_row = candidates.filter(
                (pl.col("ticker") == tk) & (pl.col("date") == target_date)
            )
            if cand_row.is_empty():
                print(f"[M1]     {tk} {dt}: not a Stage-1 candidate")
                continue
            global_idx = int(cand_row["_row_idx"][0])
            rec = _evaluate_one(global_idx, arr, params)
            if rec is None:
                print(f"[M1]     {tk} {dt}: rejected by stages 2-4")
                continue
            in_survivors = not survivors.filter(
                (pl.col("ticker") == tk) & (pl.col("date") == target_date)
            ).is_empty()
            tag = "ACCEPTED" if in_survivors else "rejected-but-recoverable"
            print(
                f"[M1]     {tk} {dt} [{tag}]: "
                f"cons_max_drop_pct={rec['cons_max_drop_pct']:.4f} "
                f"cons_low_trend_slope={rec['cons_low_trend_slope']:+.5f} "
                f"pre_legup_return={rec['pre_legup_return']:+.4f} "
                f"cons_dur={rec['cons_duration_days']} "
                f"legup_gain={rec['legup_gain_pct']:.3f}"
            )

    if survivors.is_empty():
        empty = pl.DataFrame(schema={c: pl.Utf8 for c in OUTPUT_COLUMNS})
        return empty, stats

    out_frames = []
    for variant in ("strict", "loose"):
        v_setups = _apply_variant_filter(survivors, variant, params)
        stats[f"after_variant_{variant}"] = v_setups.height
        v_setups = _apply_spacing(v_setups, params)
        stats[f"after_spacing_{variant}"] = v_setups.height
        out_frames.append(_project_output(v_setups, variant))
        print(f"[M1]     {variant}: {stats[f'after_variant_{variant}']:,} -> "
              f"{stats[f'after_spacing_{variant}']:,} after spacing")

    return pl.concat(out_frames), stats


def detect_setups(
    price_df: pl.LazyFrame,
    universe_filter: dict,
    base_params: dict,
    variant: str,
) -> pl.DataFrame:
    """Public spec interface (single variant). Computes features once, runs
    full pipeline, then filters to the requested variant."""
    features = _compute_features(price_df, base_params)
    non_eq, _, _ = _load_non_equity_set()
    setups, _ = run_pipeline(features, universe_filter, base_params, non_eq)
    return setups.filter(pl.col("universe_variant") == variant)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _write_validation(
    features: pl.DataFrame,
    setups_strict_pass: pl.DataFrame,
    stats_strict: dict,
    setups_relaxed_pass: pl.DataFrame | None,
    stats_relaxed: dict | None,
    canonical_label: str,
    canonical_setups: pl.DataFrame,
    security_source: str,
    params_used: dict,
    sec_breakdown: dict | None = None,
) -> None:
    setups = canonical_setups
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

    # Seed restored to the Round 2 value (2026504) after the Round 3
    # rollback so the sample exactly matches the Round-2 QC reference.
    sample_n = min(10, setups.height)
    sample = setups.sample(n=sample_n, seed=2026504).sort("date") if sample_n else setups
    sample.select(OUTPUT_COLUMNS).write_csv(SAMPLE_CSV)

    seg_counts = features.group_by("ticker").agg(pl.col("segment_id").max().alias("max_seg"))
    recycled_count = int(seg_counts.filter(pl.col("max_seg") > 0).height)
    total_tickers = int(seg_counts.height)

    strict_total = int(setups.filter(pl.col("universe_variant") == "strict").height)
    loose_total = int(setups.filter(pl.col("universe_variant") == "loose").height)

    lines: list[str] = []
    lines.append("# M1 — Universe & Breakout Detection Validation (rewrite)")
    lines.append("")
    lines.append(
        "Detector rewritten 2026-05-03 per `reports/m1_rule_redesign.md`. "
        "Old rule fired on pullback days inside a not-really-a-base; new rule "
        "fires on the breakout day itself after a real big-move + tight "
        "consolidation. **Schema changed** (see column list at bottom). "
        "Headline test runs on the **loose** universe; strict is a §7.1 "
        "robustness check. **Detector ships at Round 2** — see iteration "
        "history below."
    )
    lines.append("")
    lines.append("## QC iteration history")
    lines.append("")
    lines.append(
        "Manual QC of 10 random sample setups was performed three times "
        "during M1 development. Each round adjusted the detector based on "
        "specific chart-level failures. Documented honestly here so the "
        "writeup §8 limitations section can reference the empirical search:"
    )
    lines.append("")
    lines.append("| Round | Date | Detector | QC score | strict / loose totals |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| 1 (broken) | 2026-05-03 | original `m1_universe.py`: "
        "`base = leg-up`, `t = day in pullback` | **0/10** "
        "(every sample missed the breakout day) | 1,115 / 2,741 |"
    )
    lines.append(
        "| 2 (ships) | 2026-05-04 | full rewrite: 6-stage pipeline "
        "(see redesign doc); ADR layer; 15%-gap-open pivot reject; "
        "60-day belt-and-suspenders cap; hardcoded ADR fallback for "
        "yfinance 404s | **6.5/10** (6 yes, 1 sort-of, 3 no) | "
        "**2,214 / 3,075** ← canonical |"
    )
    lines.append(
        "| 3 (rolled back) | 2026-05-04 | added cons_low_trend_slope reject "
        "(<−0.002), pre_legup_return reject (<−0.20), and tightened "
        "cons_min 10→15 trading days | **2.5/10** | 1,735 / 2,371 (rejected) |"
    )
    lines.append("")
    lines.append(
        "Round 3's rejects were over-fitted to the three failed Round 2 "
        "samples (PPTA, QTWO, ATYR) and hurt non-flagged setups — the "
        "regression to 2.5/10 came from cutting valid setups, not from "
        "passing more bad ones. Both Round 3 metrics survive in this "
        "parquet as **informational columns** (`cons_low_trend_slope`, "
        "`pre_legup_return`) — useful for M2 / M3 feature exploration "
        "without affecting M1's setup count."
    )
    lines.append("")
    lines.append("")
    lines.append(f"- Feature table: {features.height:,} rows ({total_tickers:,} tickers)")
    lines.append(
        f"- Tickers split into >1 segment by the {DEFAULT_PARAMS['ticker_recycle_gap_days']}d "
        f"recycling gap: **{recycled_count:,}** ({recycled_count/total_tickers*100:.2f}%)"
    )
    lines.append(
        f"- **Canonical pass: `{canonical_label}`** (used for `setups.parquet` and the QC sample)."
    )
    lines.append(f"- Canonical strict total: **{strict_total:,}** (was {PREV_COUNTS['strict']:,} under old rule)")
    lines.append(f"- Canonical loose total:  **{loose_total:,}** (was {PREV_COUNTS['loose']:,} under old rule)")
    lines.append("")

    lines.append("## Security-type filter (Bug 1 fix, 2026-05-04)")
    lines.append("")
    lines.append(f"- Source: **{security_source}**")
    if sec_breakdown:
        lines.append("")
        lines.append("| Detection rule | Tickers flagged |")
        lines.append("|---|---:|")
        lines.append(f"| `quote_type != EQUITY` (existing rule) | {sec_breakdown.get('via_quote_type', 0):,} |")
        lines.append(f"| `country` non-null and != 'United States' (NEW) | {sec_breakdown.get('via_country_non_us', 0):,} |")
        lines.append(f"| `long_name`/`short_name` matches `(?i)\\bADR\\b\\|American Depositary\\|Sponsored ADR` (NEW) | {sec_breakdown.get('via_name_match_adr', 0):,} |")
        lines.append(f"| Hardcoded ADR fallback (yfinance 404 rescue, e.g. ERJ) | {sec_breakdown.get('via_hardcoded_adr_fallback', 0):,} |")
        lines.append(f"| **Union (final non-equity set)** | **{sec_breakdown.get('total_non_equity', 0):,}** |")
        lines.append(f"| _added by ADR layer vs old EQUITY-only filter_ | _{sec_breakdown.get('added_by_adr_layer', 0):,}_ |")
    lines.append("")
    lines.append(
        "Manual QC of the 2026-05-03 sample flagged ERJ (Embraer ADR) and a "
        "suspected EMES ETF as having slipped through the old "
        "`quote_type != EQUITY`-only filter. yfinance returns "
        "`quoteType=EQUITY` for ADRs, so two additional layers were added: "
        "country origin and long-name regex match."
    )
    lines.append("")

    lines.append("## Old rule vs new rule (totals)")
    lines.append("")
    lines.append("| Variant | Old (broken) rule | New rule (default params) | New rule (relaxed params) |")
    lines.append("|---|---:|---:|---:|")
    new_default_strict = stats_strict.get("after_spacing_strict", 0)
    new_default_loose  = stats_strict.get("after_spacing_loose", 0)
    if stats_relaxed is not None:
        new_relaxed_strict = stats_relaxed.get("after_spacing_strict", 0)
        new_relaxed_loose  = stats_relaxed.get("after_spacing_loose", 0)
        lines.append(f"| strict | {PREV_COUNTS['strict']:,} | {new_default_strict:,} | {new_relaxed_strict:,} |")
        lines.append(f"| loose  | {PREV_COUNTS['loose']:,}  | {new_default_loose:,}  | {new_relaxed_loose:,} |")
    else:
        lines.append(f"| strict | {PREV_COUNTS['strict']:,} | {new_default_strict:,} | (not triggered: loose>=200) |")
        lines.append(f"| loose  | {PREV_COUNTS['loose']:,}  | {new_default_loose:,}  | (not triggered: loose>=200) |")
    lines.append("")

    lines.append("## Pipeline dropouts (canonical pass)")
    lines.append("")
    s = stats_relaxed if stats_relaxed is not None else stats_strict
    lines.append("| Stage | Rows surviving |")
    lines.append("|---|---:|")
    lines.append(f"| Stage 0a — universe (close/ADV/history) pre-security | {s.get('after_universe_pre_security', 0):,} |")
    lines.append(f"| Stage 0b — security-type filter (drop ETFs/ADRs/...) | {s.get('after_security_filter', 0):,} |")
    lines.append(f"| Stage 1  — vectorized breakout-day mask              | {s.get('after_stage1_breakout_mask', 0):,} |")
    lines.append(f"| Stages 2-4 — pivot+leg, consolidation, pre-extension | {s.get('after_stages_2_4', 0):,} |")
    lines.append(f"| Stage 5 (strict)  — variant filter                   | {s.get('after_variant_strict', 0):,} |")
    lines.append(f"| Stage 6 (strict)  — per-ticker 30-day spacing        | {s.get('after_spacing_strict', 0):,} |")
    lines.append(f"| Stage 5 (loose)   — variant filter                   | {s.get('after_variant_loose', 0):,} |")
    lines.append(f"| Stage 6 (loose)   — per-ticker 30-day spacing        | {s.get('after_spacing_loose', 0):,} |")
    lines.append("")

    lines.append("## Setups by year x variant (canonical)")
    lines.append("")
    lines.append("| Year | strict | loose |")
    lines.append("|---:|---:|---:|")
    for y in sorted(by_year):
        s_n = by_year[y].get("strict", 0)
        l_n = by_year[y].get("loose", 0)
        lines.append(f"| {y} | {s_n:,} | {l_n:,} |")
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

    lines.append("## Parameters used (canonical pass)")
    lines.append("")
    lines.append("```")
    for k in sorted(params_used):
        lines.append(f"{k} = {params_used[k]}")
    lines.append("```")
    lines.append("")

    lines.append("## Output schema")
    lines.append("")
    lines.append(
        "Schema changed in this rewrite. Previous columns `base_start_date`, "
        "`base_end_date`, `base_duration_days`, `pullback_pct` were misnamed: "
        "they described the **leg up** before the pivot, not the consolidation. "
        "Rename map (decision 2 of the redesign):"
    )
    lines.append("")
    lines.append("| Old | New | Meaning |")
    lines.append("|---|---|---|")
    lines.append("| `base_start_date` | `legup_low_date` | low of the prior 35–300% advance |")
    lines.append("| `base_end_date`   | `legup_high_date` | pivot high terminating the advance |")
    lines.append("| `base_duration_days` | `legup_duration_days` | trading days (low→pivot) |")
    lines.append("| `pullback_pct` (misnamed) | `legup_gain_pct` | actual leg-up gain |")
    lines.append("")
    lines.append("New columns describing the **consolidation** between pivot and breakout:")
    lines.append("")
    lines.append("- `cons_start_date` (= `legup_high_date`)")
    lines.append("- `cons_end_date` (= the day before `date`)")
    lines.append("- `cons_duration_days` (trading days, in [10, 42])")
    lines.append("- `cons_max_drop_pct` (deepest dip from pivot during consolidation)")
    lines.append("- `cons_exception_days` (count of bars in (.30, .35] drop band)")
    lines.append("- `ma_touches_pct_in_cons` (fraction of cons bars touching 10/20/50-SMA from above)")
    lines.append("- `breakout_volume_ratio` (volume[t] / 20d avg share volume)")
    lines.append("- `breakout_range_pct` (daily range on breakout day)")
    lines.append("")
    lines.append(
        "Demoted (kept as supplementary stats over the **consolidation** "
        "window, NOT the leg-up — and not in the pass/fail rule): "
        "`higher_low_count`, `range_contraction_ratio`, "
        "`pct_closes_above_20ma_in_cons` (renamed from `_in_base`)."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **Daily OHLCV only.** breakouts.trade additionally gates on the "
        "first 30 minutes of intraday (volume + range scaled to a partial-day "
        "bar). We can't replicate this — the Polygon API key was cancelled "
        "for this session and only daily files are available. Some "
        "marginal-volume breakouts that would fail intraday confirmation will "
        "slip through; some valid breakouts whose intraday behavior is clean "
        "may be over-rejected by the daily-only filters. Net direction "
        "unknown but expected to be small."
    )
    lines.append(
        "- **Relative strength is the within-universe momentum percentile**, "
        "not stock_6m_return / SPY_6m_return (the working tool's gate). "
        "Decision 3 of the redesign was to keep `mom_pct` only; SPY data is "
        "not loaded."
    )
    lines.append(
        "- **No Episodic Pivot detection.** Gap-up/news-catalyst setups "
        "(Qullamaggie's other primary family) are out of scope for M1 "
        "(decision 7). Future work."
    )
    lines.append(
        "- **Reference-table location deviates from spec.** "
        "`data/raw/reference/` is a Windows junction we can't write through "
        "(see M0 audit); `yfinance_types.parquet` lives in "
        "`data/` instead."
    )
    lines.append(
        "- **yfinance type coverage is partial.** Source: "
        f"**{security_source}**. Tickers absent from the cache are assumed "
        "EQUITY (safe direction; alternative is silently dropping common "
        "stocks the cache hasn't seen)."
    )
    lines.append("")

    lines.append("## 10 random sample setups (in `m1_sample_setups.csv`)")
    lines.append("")
    lines.append(
        "| Ticker | Date | Variant | mom_pct | legup_gain | pre_leg_ret | "
        "cons_dur | cons_drop | cons_slope | ma_touch | bo_vol_x | bo_range |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sample.to_dicts():
        lines.append(
            f"| {r['ticker']} | {r['date']} | {r['universe_variant']} | "
            f"{r['mom_pct']:.3f} | {r['legup_gain_pct']:.3f} | "
            f"{r['pre_legup_return']:+.3f} | "
            f"{r['cons_duration_days']} | {r['cons_max_drop_pct']:.3f} | "
            f"{r['cons_low_trend_slope']:+.5f} | "
            f"{r['ma_touches_pct_in_cons']:.3f} | "
            f"{r['breakout_volume_ratio']:.2f} | {r['breakout_range_pct']:.3f} |"
        )
    lines.append("")

    VALIDATION_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[M1] strict total (canonical): {strict_total:,}")
    print(f"[M1] loose total (canonical):  {loose_total:,}")
    print(f"[M1] recycled tickers: {recycled_count:,} / {total_tickers:,}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    print("[M1] scanning daily bars and computing features...")
    price_df = pl.scan_parquet(DAILY_BARS_GLOB)
    features = _compute_features(price_df, DEFAULT_PARAMS)
    print(f"[M1] feature table: {features.height:,} rows")

    non_eq, security_source, sec_breakdown = _load_non_equity_set()
    print(f"[M1] security-type filter source: {security_source}")
    print(f"[M1] non-equity tickers in drop set: {len(non_eq):,}")
    if sec_breakdown:
        print(
            f"[M1]   breakdown: {sec_breakdown.get('via_quote_type', 0):,} via quote_type!=EQUITY, "
            f"{sec_breakdown.get('via_country_non_us', 0):,} via country!=US, "
            f"{sec_breakdown.get('via_name_match_adr', 0):,} via name match"
        )
        print(
            f"[M1]   ADR layer added {sec_breakdown.get('added_by_adr_layer', 0):,} "
            f"tickers vs the old EQUITY-only filter"
        )

    print("[M1] === pass 1: default params ===")
    setups_default, stats_default = run_pipeline(
        features, UNIVERSE_FILTER, DEFAULT_PARAMS, non_eq,
        diagnostic_targets=[
            ("WMS", "2021-05-21"),    # Round 2 QC anchor (still passes)
            ("PLPC", "2025-10-08"),   # Round 2 QC anchor (still passes)
        ],
    )
    loose_default = stats_default.get("after_spacing_loose", 0)
    strict_default = stats_default.get("after_spacing_strict", 0)
    print(f"[M1] default pass: strict={strict_default:,}, loose={loose_default:,}")

    relaxed_used = False
    setups_relaxed = None
    stats_relaxed = None
    params_canonical = DEFAULT_PARAMS
    canonical_label = "default"
    canonical_setups = setups_default

    if loose_default < 200:
        print(f"[M1] loose<200; running relaxed pass (vol_surge_x={RELAXED_OVERRIDES['vol_surge_x']}, "
              f"leg_min_gain_pct={RELAXED_OVERRIDES['leg_min_gain_pct']}, "
              f"cons_min_pullback_pct={RELAXED_OVERRIDES['cons_min_pullback_pct']})")
        relaxed_params = {**DEFAULT_PARAMS, **RELAXED_OVERRIDES}
        setups_relaxed, stats_relaxed = run_pipeline(
            features, UNIVERSE_FILTER, relaxed_params, non_eq,
        )
        loose_relaxed = stats_relaxed.get("after_spacing_loose", 0)
        strict_relaxed = stats_relaxed.get("after_spacing_strict", 0)
        print(f"[M1] relaxed pass: strict={strict_relaxed:,}, loose={loose_relaxed:,}")
        relaxed_used = True
        params_canonical = relaxed_params
        canonical_label = "relaxed (vol/leg/pullback thresholds eased — needed for QR sample size)"
        canonical_setups = setups_relaxed

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    canonical_setups.write_parquet(OUT_PARQUET)
    print(f"[M1] wrote {OUT_PARQUET}")

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    _write_validation(
        features,
        setups_strict_pass=setups_default,
        stats_strict=stats_default,
        setups_relaxed_pass=setups_relaxed,
        stats_relaxed=stats_relaxed,
        canonical_label=canonical_label,
        canonical_setups=canonical_setups,
        security_source=security_source,
        params_used=params_canonical,
        sec_breakdown=sec_breakdown,
    )
    print(f"[M1] wrote {VALIDATION_MD}")
    print(f"[M1] wrote {SAMPLE_CSV}")


if __name__ == "__main__":
    main()
