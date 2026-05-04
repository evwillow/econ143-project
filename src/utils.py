from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DAILY_BARS = DATA_RAW / "stocks" / "daily"
# Daily bars live in the breakoutStudyTool pipeline directory; access them
# directly to avoid Windows junction traversal issues with polars' scanner.
DAILY_BARS_GLOB = "C:/Users/evanm/Desktop/projects/breakoutStudyTool/data/pipeline/stocks/daily/*/*.parquet"
SPLITS = DATA_RAW / "corporate_actions" / "splits.parquet"
SPY_DAILY = DATA_RAW / "indices" / "daily"

# Spec calls for data/raw/reference/ but data/raw is a Windows junction we
# can't write through (see M0 audit). Live with the deviation: reference
# tables we generate (yfinance ticker types) go in interim instead.
REFERENCE_DIR = REPO_ROOT / "data" / "interim" / "reference"
TICKER_TYPES_PARQUET = REFERENCE_DIR / "yfinance_types.parquet"
TICKER_TYPES_FAILURES = REFERENCE_DIR / "yfinance_failures.csv"

# Hardcoded fallback for known non-common-stock tickers (option (c) in the
# Issue 1 plan): used only if yfinance fetch hasn't produced a types table.
NON_COMMON_STOCK_EXCLUSIONS: frozenset[str] = frozenset({
    # Broad-market index ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV", "VEA", "VWO", "EFA", "EEM",
    # Sector SPDRs
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
    # Other sector / thematic ETFs that have appeared in setups
    "XME", "XOP", "XBI", "XHB", "XRT", "XPH", "XSD", "SMH", "SOXX", "ITA",
    "IBB", "IYR", "IYT", "IGV", "IHI", "KBE", "KRE", "KIE", "JETS",
    # Leveraged / inverse ETFs
    "SOXL", "SOXS", "TQQQ", "SQQQ", "TNA", "TZA", "FAS", "FAZ", "SPXL", "SPXS",
    "SPXU", "UPRO", "TMF", "TMV", "UVXY", "VXX", "SVXY", "VIXY", "BOIL", "KOLD",
    "NUGT", "DUST", "JNUG", "JDST", "GUSH", "DRIP", "LABU", "LABD", "TECL", "TECS",
    "FNGU", "FNGD", "BULZ", "WEBL", "BERZ", "DPST", "WANT", "HIBL", "HIBS",
    # Volatility / commodity / currency ETFs/ETNs
    "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG", "DBA", "DBC", "DBO", "DBE",
    "UUP", "FXE", "FXY", "FXB", "FXA",
    # Bond ETFs
    "TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP", "MUB",
    "EMB", "BIL",
    # Notable thematic / large active ETFs
    "ARKK", "ARKG", "ARKW", "ARKQ", "ARKF", "BITO", "PSQ", "SH",
})
