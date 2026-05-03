from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"
DAILY_BARS = DATA_RAW / "stocks" / "daily"
# Daily bars live in the breakoutStudyTool pipeline directory; access them
# directly to avoid Windows junction traversal issues with polars' scanner.
DAILY_BARS_GLOB = "C:/Users/evanm/Desktop/projects/breakoutStudyTool/data/pipeline/stocks/daily/*/*.parquet"
SPLITS = DATA_RAW / "corporate_actions" / "splits.parquet"
SPY_DAILY = DATA_RAW / "indices" / "daily"
