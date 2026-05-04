"""Download and parse Ken French FF3 + UMD daily factors.

Builds a single tidy panel: date, mkt_rf, smb, hml, umd, rf (all decimals).
Saved to data/factors/ff3_umd_daily.parquet.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTORS_DIR = REPO_ROOT / "data" / "factors"
RAW_DIR = FACTORS_DIR / "raw"
OUTPUT_PATH = FACTORS_DIR / "ff3_umd_daily.parquet"

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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    panel = build_factor_panel()

    print(f"Rows: {panel.height}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print("\nFirst 5 rows:")
    print(panel.head(5))
    print("\nLast 5 rows:")
    print(panel.tail(5))

    covid = panel.filter(pl.col("date") == pl.date(2020, 3, 16))
    if covid.height == 0:
        print("\n[WARN] 2020-03-16 not found in panel")
    else:
        mkt_rf = covid["mkt_rf"][0]
        print(f"\nValidation 2020-03-16 Mkt-RF = {mkt_rf:.4f} (expected ~ -0.13)")
        if abs(mkt_rf - (-0.13)) < 0.02:
            print("  PASS: parsing looks correct")
        else:
            print("  FAIL: value off from expected -0.13")

    FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(OUTPUT_PATH)
    print(f"\nWrote {OUTPUT_PATH} ({panel.height} rows)")


if __name__ == "__main__":
    main()
