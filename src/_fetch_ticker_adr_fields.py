"""
One-off helper to AUGMENT data/interim/reference/yfinance_types.parquet with
the fields needed for the ADR exclusion in M1's _load_non_equity_set:

  - country (e.g. "United States", "Brazil")
  - long_name (e.g. "Embraer S.A. American Depositary Shares")

Refetches `.info` for every unique ticker in the current
data/interim/setups.parquet, preserving any existing rows that aren't in the
new fetch list. Resumable: re-reads the cache and skips tickers that already
have BOTH country AND long_name filled (i.e., were fetched after this script
ran). Failures are appended to data/interim/reference/yfinance_failures.csv.

Run with `python src/_fetch_ticker_adr_fields.py`.
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

SETUPS_PATH = Path(__file__).resolve().parents[1] / "data" / "interim" / "setups.parquet"

SLEEP_SECONDS = 0.25
CHECKPOINT_EVERY = 50
PROGRESS_EVERY = 25


def _load_existing_cache() -> dict[str, dict]:
    if not TICKER_TYPES_PARQUET.exists():
        return {}
    df = pl.read_parquet(TICKER_TYPES_PARQUET)
    return {r["ticker"]: r for r in df.to_dicts()}


def _write_cache(rows_by_ticker: dict[str, dict]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(rows_by_ticker.values())
    if not rows:
        return
    # Polars infers schema from records; missing keys become null.
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
    print(f"[fetch-adr] {len(tickers_all):,} unique tickers in setups.parquet", flush=True)

    cache = _load_existing_cache()
    # Skip tickers already fully populated with the new fields.
    todo = [
        t for t in tickers_all
        if not (
            t in cache
            and cache[t].get("country")
            and cache[t].get("long_name")
        )
    ]
    print(f"[fetch-adr] {len(cache):,} cached entries; "
          f"{len(todo):,} need country+long_name fetched", flush=True)

    if not todo:
        print("[fetch-adr] nothing to do", flush=True)
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
            country = info.get("country")
            long_name = info.get("longName")
            short_name = info.get("shortName")
            sector = info.get("sector")
            industry = info.get("industry")
            if qt is None and country is None and long_name is None:
                n_fail += 1
                _append_failure(tk, "empty info")
                continue
            n_ok += 1
            row = {
                "ticker": tk,
                "quote_type": qt,
                "sector": sector,
                "industry": industry,
                "country": country,
                "long_name": long_name,
                "short_name": short_name,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            cache[tk] = row
        except Exception as e:
            n_fail += 1
            _append_failure(tk, f"exception:{type(e).__name__}:{str(e)[:80]}")

        if n_done % PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            rate = n_done / elapsed if elapsed else 0
            eta = (len(todo) - n_done) / rate if rate else 0
            print(f"[fetch-adr] {n_done:,}/{len(todo):,} "
                  f"({n_ok:,} ok, {n_fail:,} fail) "
                  f"~{rate:.1f}/s eta={eta/60:.1f}min", flush=True)
        if n_done % CHECKPOINT_EVERY == 0:
            _write_cache(cache)
        time.sleep(SLEEP_SECONDS)

    _write_cache(cache)
    elapsed = time.time() - start
    print(f"[fetch-adr] done. {n_done:,} fetched in {elapsed/60:.1f}min "
          f"({n_ok:,} ok / {n_fail:,} fail). "
          f"Cache size: {len(cache):,} tickers", flush=True)


if __name__ == "__main__":
    main()
