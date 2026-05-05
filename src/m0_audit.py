"""
M0 — Survivorship & Bad-Bar Audit
==================================

Design question answers (§M0), written before any code.

--- Q1: Does Polygon data contain delisted tickers? ---

Check tickers: LEH (Lehman Brothers, bankrupt Sep 2008), BBBY (Bed Bath &
Beyond, bankrupt Apr 2023), WCG (WellCare Group, acquired Jan 2020), SIVB
(Silicon Valley Bank, closed Mar 2023), FRC (First Republic, seized May 2023).

Expected behavior: Polygon's flat-file exports include delisted names because
they were part of the universe during their trading life. If a ticker's last
bar matches its known delisting date (±5 trading days), the survivorship bias
for that name is bounded. If the ticker is entirely absent, the data is
survivors-only for that name.

Decision gate (from spec):
  - >=50% of the 5 probes present -> proceed; caveat bias in writeup.
  - <20% present -> pivot to CRSP via library access before continuing.

Bias direction: absent delisters inflate backtest returns (dead stocks tend
to be the worst performers). Bound the upward bias as:
  bias <= delisting_rate_per_year * avg_delisting_excess_loss * horizon_years
We'll estimate this and state it explicitly in reports/audit.md.

--- Q2: Are there bad bars? ---

Categories counted independently across all bars in [2010, 2025]:
  a. OHLC inconsistent  : high < low, or low > min(open, close),
                          or high < max(open, close)
  b. Null OHLCV         : any null in open/high/low/close/volume
  c. Non-positive price : close <= 0 or open <= 0 or high <= 0 or low <= 0
  d. Stale feed         : open == high == low == close AND volume > 0
  e. Suspicious move    : |close_t / close_{t-1} - 1| > 0.5; counted but not
                          treated as bad (often an unadjusted split). Cross-
                          referenced against splits.parquet to report the
                          fraction unexplained.

Hard-bad fraction = (a + b + c + d) / total_bars. Threshold: any year above
0.1% is flagged in audit_summary.json. Bad rows are not dropped here — that
decision belongs in M1 — but they are recorded.

--- Q3: Date coverage per year ---

Expected trading days per calendar year: ~252 (US market schedule).
Threshold: any year in [2010, 2025] with < 200 distinct trading dates is
flagged as suspect coverage.

--- Interface ---

Inputs : data/raw/stocks/daily  (parquets via utils.py paths)
         data/raw/corporate_actions/splits.parquet (optional)
Outputs: reports/audit.md
         data/m0_audit_summary.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Allow running this file directly as `python src/m0_audit.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DAILY_BARS, DAILY_BARS_GLOB, REPO_ROOT, SPLITS  # noqa: E402

DELISTED_PROBES: list[str] = ["LEH", "BBBY", "WCG", "SIVB", "FRC"]
START_YEAR = 2010
END_YEAR = 2025
BAD_BAR_FRACTION_THRESHOLD = 0.001  # 0.1%
MIN_TRADING_DAYS_PER_YEAR = 200
SUSPICIOUS_MOVE_THRESHOLD = 0.5

AUDIT_MD = REPO_ROOT / "reports" / "audit.md"
AUDIT_JSON = REPO_ROOT / "data" / "m0_audit_summary.json"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _scan_daily_bars() -> pl.LazyFrame:
    """Lazily scan every daily-bar parquet matching DAILY_BARS_GLOB
    (one file per year, e.g. .../daily/2010/2010.parquet), returning a
    LazyFrame with a normalized schema:
        ticker (Utf8), date (Date),
        open/high/low/close (Float64), volume (Int64).

    Tolerates a few common Polygon column-name variants (T/symbol for
    ticker; window_start/timestamp/t for date) so the loader does not
    silently break if the upstream dump format changes.
    """
    lf = pl.scan_parquet(DAILY_BARS_GLOB)
    schema = lf.collect_schema()
    cols = schema.names()

    # ---- ticker ----
    if "ticker" in cols:
        ticker_expr = pl.col("ticker").cast(pl.Utf8)
    elif "symbol" in cols:
        ticker_expr = pl.col("symbol").cast(pl.Utf8).alias("ticker")
    elif "T" in cols:
        ticker_expr = pl.col("T").cast(pl.Utf8).alias("ticker")
    else:
        raise RuntimeError(
            f"daily bars: no ticker column found. schema={dict(schema)}"
        )

    # ---- date ----
    if "date" in cols:
        date_expr = pl.col("date").cast(pl.Date)
    elif "window_start" in cols:
        date_expr = pl.col("window_start").cast(pl.Datetime).dt.date().alias("date")
    elif "timestamp" in cols:
        date_expr = pl.col("timestamp").cast(pl.Datetime).dt.date().alias("date")
    elif "t" in cols:
        # Polygon REST timestamp; nanosecond if it's huge, else millisecond.
        t_dtype = schema["t"]
        unit = "ns" if str(t_dtype).startswith("Int64") else "ms"
        date_expr = pl.from_epoch(pl.col("t"), time_unit=unit).dt.date().alias("date")
    else:
        raise RuntimeError(
            f"daily bars: no date column found. schema={dict(schema)}"
        )

    # ---- OHLCV: tolerate Polygon REST shorthand (o/h/l/c/v) ----
    rename_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    rename_now = {k: v for k, v in rename_map.items() if k in cols and v not in cols}
    if rename_now:
        lf = lf.rename(rename_now)
        cols = lf.collect_schema().names()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(
            f"daily bars: missing required OHLCV columns {missing}. schema={dict(schema)}"
        )

    return lf.with_columns(ticker_expr, date_expr).select(
        ["ticker", "date", "open", "high", "low", "close", "volume"]
    )


def _scan_splits() -> pl.LazyFrame | None:
    """Return a LazyFrame with columns (ticker, date) marking split events,
    or None if splits.parquet is absent / unreadable."""
    try:
        if not SPLITS.exists():
            return None
        lf = pl.scan_parquet(str(SPLITS))
    except Exception:
        # Junction-traversal errors on Windows fall through here too.
        return None
    cols = lf.collect_schema().names()

    if "ticker" in cols:
        t = pl.col("ticker").cast(pl.Utf8)
    elif "symbol" in cols:
        t = pl.col("symbol").cast(pl.Utf8).alias("ticker")
    else:
        return None

    for cand in ("execution_date", "ex_date", "date", "effective_date"):
        if cand in cols:
            d = pl.col(cand).cast(pl.Date).alias("date")
            break
    else:
        return None

    return lf.select([t, d]).unique()


# ---------------------------------------------------------------------------
# Check 1: survivorship
# ---------------------------------------------------------------------------

def survivorship_check(bars: pl.LazyFrame) -> dict:
    summary = (
        bars.filter(pl.col("ticker").is_in(DELISTED_PROBES))
        .group_by("ticker")
        .agg(
            pl.len().alias("bar_count"),
            pl.col("date").max().alias("last_date"),
            pl.col("date").min().alias("first_date"),
        )
        .collect()
    )

    found = {row["ticker"]: row for row in summary.to_dicts()}
    rows = []
    for tk in DELISTED_PROBES:
        r = found.get(tk)
        if r is None:
            rows.append(
                {
                    "ticker": tk,
                    "present": False,
                    "bar_count": 0,
                    "first_date": None,
                    "last_date": None,
                }
            )
        else:
            rows.append(
                {
                    "ticker": tk,
                    "present": True,
                    "bar_count": int(r["bar_count"]),
                    "first_date": r["first_date"].isoformat() if r["first_date"] else None,
                    "last_date": r["last_date"].isoformat() if r["last_date"] else None,
                }
            )

    present_count = sum(1 for r in rows if r["present"])
    pct_present = present_count / len(DELISTED_PROBES)
    if pct_present >= 0.5:
        gate = "proceed"
    elif pct_present < 0.2:
        gate = "pivot"
    else:
        gate = "caveat"

    return {
        "tickers": rows,
        "present_count": present_count,
        "total_probes": len(DELISTED_PROBES),
        "fraction_present": pct_present,
        "decision_gate": gate,
    }


# ---------------------------------------------------------------------------
# Check 2: bad bars
# ---------------------------------------------------------------------------

def bad_bar_check(
    bars: pl.LazyFrame, splits: pl.LazyFrame | None
) -> dict:
    in_window = bars.filter(
        (pl.col("date") >= pl.date(START_YEAR, 1, 1))
        & (pl.col("date") <= pl.date(END_YEAR, 12, 31))
    ).with_columns(pl.col("date").dt.year().alias("year"))

    flag_ohlc = (
        (pl.col("high") < pl.col("low"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("high") < pl.max_horizontal("open", "close"))
    )
    flag_null = pl.any_horizontal(
        pl.col("open").is_null(),
        pl.col("high").is_null(),
        pl.col("low").is_null(),
        pl.col("close").is_null(),
        pl.col("volume").is_null(),
    )
    flag_nonpos = (
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
    )
    flag_stale = (
        (pl.col("open") == pl.col("high"))
        & (pl.col("high") == pl.col("low"))
        & (pl.col("low") == pl.col("close"))
        & (pl.col("volume") > 0)
    )

    prev_close = (
        pl.col("close").shift(1).over("ticker", order_by="date")
    )
    flag_susp = (
        prev_close.is_not_null()
        & (prev_close > 0)
        & (pl.col("close") > 0)
        & ((pl.col("close") / prev_close - 1).abs() > SUSPICIOUS_MOVE_THRESHOLD)
    )

    flagged = in_window.with_columns(
        flag_ohlc.alias("_ohlc"),
        flag_null.alias("_null"),
        flag_nonpos.alias("_nonpos"),
        flag_stale.alias("_stale"),
        flag_susp.alias("_susp"),
    )

    totals = flagged.select(
        pl.len().alias("total"),
        pl.col("_ohlc").sum().alias("ohlc"),
        pl.col("_null").sum().alias("null"),
        pl.col("_nonpos").sum().alias("nonpos"),
        pl.col("_stale").sum().alias("stale"),
        pl.col("_susp").sum().alias("susp"),
    ).collect().row(0, named=True)

    total = int(totals["total"])
    counts = {
        "ohlc_inconsistent": int(totals["ohlc"]),
        "null_ohlcv": int(totals["null"]),
        "nonpositive_price": int(totals["nonpos"]),
        "stale_feed": int(totals["stale"]),
        "suspicious_move": int(totals["susp"]),
    }

    def _frac(n: int) -> float:
        return n / total if total else 0.0

    categories = {
        name: {"count": cnt, "fraction": _frac(cnt)} for name, cnt in counts.items()
    }

    # Cross-reference suspicious moves against splits.parquet.
    unexplained_susp: int | None = None
    if splits is not None and counts["suspicious_move"] > 0:
        try:
            susp_rows = (
                flagged.filter(pl.col("_susp"))
                .select(["ticker", "date"])
                .join(splits.with_columns(pl.lit(True).alias("_split")),
                       on=["ticker", "date"], how="left")
                .select(pl.col("_split").is_null().sum().alias("n_unexpl"))
                .collect()
            )
            unexplained_susp = int(susp_rows.row(0)[0])
            categories["suspicious_move"]["unexplained_by_splits"] = unexplained_susp
            categories["suspicious_move"]["unexplained_fraction"] = _frac(unexplained_susp)
        except Exception as e:  # don't fail the audit on a side check
            categories["suspicious_move"]["splits_join_error"] = str(e)
    elif splits is None:
        categories["suspicious_move"]["unexplained_by_splits"] = None
        categories["suspicious_move"]["splits_note"] = "splits.parquet not present"

    # Per-year hard-bad fraction (a+b+c+d only).
    by_year_df = (
        flagged.group_by("year")
        .agg(
            pl.len().alias("bars"),
            (pl.col("_ohlc") | pl.col("_null") | pl.col("_nonpos") | pl.col("_stale"))
            .sum()
            .alias("hard_bad"),
        )
        .sort("year")
        .collect()
    )

    by_year = []
    flagged_years: list[int] = []
    for row in by_year_df.to_dicts():
        bars_y = int(row["bars"])
        bad_y = int(row["hard_bad"])
        frac = bad_y / bars_y if bars_y else 0.0
        flag = frac > BAD_BAR_FRACTION_THRESHOLD
        if flag:
            flagged_years.append(int(row["year"]))
        by_year.append(
            {
                "year": int(row["year"]),
                "bars": bars_y,
                "hard_bad": bad_y,
                "hard_bad_fraction": frac,
                "flagged": flag,
            }
        )

    hard_bad_total = (
        counts["ohlc_inconsistent"]
        + counts["null_ohlcv"]
        + counts["nonpositive_price"]
        + counts["stale_feed"]
    )

    return {
        "total_bars": total,
        "hard_bad_total": hard_bad_total,
        "hard_bad_fraction": _frac(hard_bad_total),
        "categories": categories,
        "by_year": by_year,
        "flagged_years": flagged_years,
        "thresholds": {
            "bad_bar_fraction": BAD_BAR_FRACTION_THRESHOLD,
            "suspicious_move": SUSPICIOUS_MOVE_THRESHOLD,
        },
    }


# ---------------------------------------------------------------------------
# Check 3: coverage
# ---------------------------------------------------------------------------

def coverage_check(bars: pl.LazyFrame) -> dict:
    df = (
        bars.filter(
            (pl.col("date") >= pl.date(START_YEAR, 1, 1))
            & (pl.col("date") <= pl.date(END_YEAR, 12, 31))
        )
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.col("date").n_unique().alias("trading_dates"))
        .sort("year")
        .collect()
    )

    seen = {int(r["year"]): int(r["trading_dates"]) for r in df.to_dicts()}
    rows = []
    flagged: list[int] = []
    for y in range(START_YEAR, END_YEAR + 1):
        td = seen.get(y, 0)
        flag = td < MIN_TRADING_DAYS_PER_YEAR
        if flag:
            flagged.append(y)
        rows.append({"year": y, "trading_dates": td, "flagged": flag})

    return {
        "by_year": rows,
        "flagged_years": flagged,
        "min_trading_days_threshold": MIN_TRADING_DAYS_PER_YEAR,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def render_markdown(results: dict) -> str:
    s = results["survivorship"]
    b = results["bad_bars"]
    c = results["coverage"]

    lines: list[str] = []
    lines.append("# M0 — Survivorship & Bad-Bar Audit")
    lines.append("")
    lines.append(f"_Generated {results['generated_at']}_")
    lines.append("")
    lines.append(f"- Daily bars source: `{DAILY_BARS}`")
    lines.append(f"- Window: {START_YEAR}-{END_YEAR}")
    lines.append("")

    # --- 1. Survivorship ---
    lines.append("## 1. Survivorship Check")
    lines.append("")
    lines.append(
        f"{s['present_count']} of {s['total_probes']} known-delisted probe tickers "
        f"present ({_fmt_pct(s['fraction_present'])}). Decision gate: **{s['decision_gate']}**."
    )
    lines.append("")
    lines.append("| Ticker | Present | Bar count | First bar | Last bar |")
    lines.append("|---|---|---|---|---|")
    for r in s["tickers"]:
        lines.append(
            f"| {r['ticker']} | {'yes' if r['present'] else 'NO'} | "
            f"{r['bar_count']:,} | {r['first_date'] or '—'} | {r['last_date'] or '—'} |"
        )
    lines.append("")
    lines.append(
        "Decision gate (per spec): >=50% present -> proceed with caveat; "
        "<20% present -> pivot to CRSP."
    )
    lines.append("")

    # --- 2. Bad bars ---
    lines.append("## 2. Bad-Bar Check")
    lines.append("")
    lines.append(
        f"Total bars in {START_YEAR}-{END_YEAR}: **{b['total_bars']:,}**. "
        f"Hard-bad (a+b+c+d): {b['hard_bad_total']:,} "
        f"({_fmt_pct(b['hard_bad_fraction'])})."
    )
    lines.append("")
    lines.append("| Category | Count | Fraction |")
    lines.append("|---|---:|---:|")
    cat_labels = {
        "ohlc_inconsistent": "(a) OHLC inconsistent",
        "null_ohlcv": "(b) Null OHLCV",
        "nonpositive_price": "(c) Non-positive price",
        "stale_feed": "(d) Stale feed (O==H==L==C, V>0)",
        "suspicious_move": "(e) Suspicious move >50% (informational)",
    }
    for key, label in cat_labels.items():
        cat = b["categories"][key]
        lines.append(f"| {label} | {cat['count']:,} | {_fmt_pct(cat['fraction'])} |")
    susp = b["categories"]["suspicious_move"]
    if susp.get("unexplained_by_splits") is not None:
        lines.append("")
        lines.append(
            f"Of {susp['count']:,} suspicious moves, "
            f"**{susp['unexplained_by_splits']:,}** "
            f"({_fmt_pct(susp.get('unexplained_fraction', 0.0))} of all bars) are "
            "not explained by a split on the same date in `splits.parquet`."
        )
    elif "splits_note" in susp:
        lines.append("")
        lines.append(f"_Note: {susp['splits_note']}; suspicious moves not cross-referenced._")
    lines.append("")
    lines.append(
        f"Per-year hard-bad fraction (flag threshold: "
        f">{_fmt_pct(BAD_BAR_FRACTION_THRESHOLD)}):"
    )
    lines.append("")
    lines.append("| Year | Bars | Hard-bad | Fraction | Flagged |")
    lines.append("|---:|---:|---:|---:|---|")
    for row in b["by_year"]:
        lines.append(
            f"| {row['year']} | {row['bars']:,} | {row['hard_bad']:,} | "
            f"{_fmt_pct(row['hard_bad_fraction'])} | "
            f"{'YES' if row['flagged'] else ''} |"
        )
    if b["flagged_years"]:
        lines.append("")
        lines.append(
            f"**Flagged years:** {', '.join(str(y) for y in b['flagged_years'])}"
        )
    lines.append("")

    # --- 3. Coverage ---
    lines.append("## 3. Coverage Check")
    lines.append("")
    lines.append(
        f"Distinct trading dates per calendar year (flag threshold: "
        f"< {MIN_TRADING_DAYS_PER_YEAR}, expected ~252):"
    )
    lines.append("")
    lines.append("| Year | Trading dates | Flagged |")
    lines.append("|---:|---:|---|")
    for r in c["by_year"]:
        lines.append(
            f"| {r['year']} | {r['trading_dates']:,} | "
            f"{'YES' if r['flagged'] else ''} |"
        )
    if c["flagged_years"]:
        lines.append("")
        lines.append(
            f"**Flagged years:** {', '.join(str(y) for y in c['flagged_years'])}"
        )
    else:
        lines.append("")
        lines.append("No years flagged.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    bars = _scan_daily_bars()
    splits = _scan_splits()

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "daily_bars_path": str(DAILY_BARS),
        "splits_path": str(SPLITS) if splits is not None else None,
        "window": {"start_year": START_YEAR, "end_year": END_YEAR},
        "survivorship": survivorship_check(bars),
        "bad_bars": bad_bar_check(bars, splits),
        "coverage": coverage_check(bars),
    }

    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    AUDIT_MD.write_text(render_markdown(results), encoding="utf-8")

    print(f"wrote {AUDIT_MD}")
    print(f"wrote {AUDIT_JSON}")


if __name__ == "__main__":
    main()
