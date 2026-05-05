"""
One-off helper: fetch quoteType from yfinance for every ticker that appeared
in data/m1_setups.parquet and cache the result so M1 can drop non-equity
issuers (ETFs, ADRs, units, warrants, etc.) without contacting the network on
each rerun.

Resumable: re-reads the cache parquet on startup and only queries tickers
that aren't there yet. Writes incrementally every CHECKPOINT_EVERY tickers,
so a crash loses at most that many fetches.

Run with `python src/_fetch_ticker_types.py`. Expected runtime ~30 min for
the ~3.7K-ticker setup universe at the default 0.5 s sleep.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import REFERENCE_DIR, TICKER_TYPES_FAILURES, TICKER_TYPES_PARQUET  # noqa: E402

SETUPS_PATH = Path(__file__).resolve().parents[1] / "data" / "m1_setups.parquet"

SLEEP_SECONDS = 0.5
CHECKPOINT_EVERY = 50
PROGRESS_EVERY = 25


def _load_existing_cache() -> tuple[set[str], list[dict]]:
    if not TICKER_TYPES_PARQUET.exists():
        return set(), []
    df = pl.read_parquet(TICKER_TYPES_PARQUET)
    rows = df.to_dicts()
    return {r["ticker"] for r in rows}, rows


def _write_cache(rows: list[dict]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(TICKER_TYPES_PARQUET)


def _append_failure(ticker: str, reason: str) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    new = not TICKER_TYPES_FAILURES.exists()
    with TICKER_TYPES_FAILURES.open("a", encoding="utf-8") as f:
        if new:
            f.write("ticker,reason,fetched_at\n")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # Quote reason in case it contains commas.
        f.write(f"{ticker},\"{reason}\",{ts}\n")


def main() -> None:
    if not SETUPS_PATH.exists():
        raise SystemExit(f"setups.parquet not found at {SETUPS_PATH}")

    setups = pl.read_parquet(SETUPS_PATH, columns=["ticker"])
    tickers_all = sorted(t for t in setups["ticker"].unique().to_list() if t)
    print(f"[fetch] {len(tickers_all):,} unique tickers in setups.parquet")

    cached, rows = _load_existing_cache()
    todo = [t for t in tickers_all if t not in cached]
    print(f"[fetch] {len(cached):,} already cached, {len(todo):,} to fetch")

    if not todo:
        print("[fetch] nothing to do")
        return

    start = time.time()
    n_done = 0
    n_ok = 0
    n_fail = 0

    for tk in todo:
        n_done += 1
        try:
            info = yf.Ticker(tk).info or {}
            qt = info.get("quoteType")
            if qt is None:
                n_fail += 1
                _append_failure(tk, "no quoteType")
            else:
                n_ok += 1
                rows.append({
                    "ticker": tk,
                    "quote_type": qt,
                    # Sector / industry are GICS-ish but yfinance-flavored
                    # ("Technology", "Semiconductors", "Healthcare",
                    # "Biotechnology", ...). Empty for ETFs/funds. M2 will
                    # map these to GICS-11 buckets for the sector FE.
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
        except Exception as e:
            n_fail += 1
            # Truncate the message; some yfinance errors are huge HTML dumps.
            msg = str(e).replace("\n", " ").replace("\r", " ")[:200]
            _append_failure(tk, msg)

        if n_done % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(todo) - n_done) / rate if rate > 0 else 0
            print(
                f"[fetch] {n_done:,}/{len(todo):,}  ok={n_ok:,}  fail={n_fail:,}  "
                f"rate={rate:.1f}/s  eta={eta/60:.1f}min"
            )

        if n_done % CHECKPOINT_EVERY == 0:
            _write_cache(rows)

        time.sleep(SLEEP_SECONDS)

    _write_cache(rows)
    print(f"[fetch] done: ok={n_ok:,}  fail={n_fail:,}  cache={TICKER_TYPES_PARQUET}")
    if n_fail:
        print(f"[fetch] failures logged to {TICKER_TYPES_FAILURES}")


if __name__ == "__main__":
    main()
