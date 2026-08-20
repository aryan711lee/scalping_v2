"""
Standalone script to download, clean, resample, store, and validate historical candle data.

Usage:
    python scripts/download_historical.py
    python scripts/download_historical.py --symbol NSE:RELIANCE-EQ
    python scripts/download_historical.py --from 2024-01-01 --to 2024-06-30
"""
import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.logger  # noqa: F401 — must be first so logging is configured
import logging

from app.config import WATCHLIST, TIMEFRAMES, HISTORICAL_DATA_START
from data.ingestion.fyers_client import FyersClient
from data.ingestion.historical import download_symbol
from data.cleaning.cleaner import clean
from data.resampling.resampler import resample_all
from data.validation.validator import run_validation_report, validate_symbol
from database.database import get_engine
from database.repository import upsert_candles, log_download, log_quality_check

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical OHLCV data from Fyers")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol override, e.g. NSE:RELIANCE-EQ")
    parser.add_argument("--from", dest="date_from", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", type=str, default=None, help="End date YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datetime import date
    date_from = args.date_from or HISTORICAL_DATA_START
    date_to = args.date_to or date.today().strftime("%Y-%m-%d")
    symbols = [args.symbol] if args.symbol else WATCHLIST
    timeframes = list(TIMEFRAMES.keys())

    logger.info("=== Starting download pipeline ===")
    logger.info("Symbols: %d | Timeframes: %s | Range: %s – %s", len(symbols), timeframes, date_from, date_to)

    # Step 1: Init Fyers client and DB
    client = FyersClient()
    get_engine()  # creates tables if needed

    # Maps for validation
    all_clean: dict[tuple[str, str], object] = {}
    interval_map = {k: int(v) for k, v in TIMEFRAMES.items()}

    for symbol in symbols:
        for tf_label in timeframes:
            logger.info("--- %s [%s] ---", symbol, tf_label)

            # Step 2: Download raw
            raw_df = download_symbol(client, symbol, tf_label, date_from, date_to)

            if raw_df is None or raw_df.empty:
                log_download(symbol, tf_label, date_from, date_to, 0, "failed", "No data returned")
                continue

            log_download(symbol, tf_label, date_from, date_to, len(raw_df), "success")

            # Step 3: Clean
            interval_min = int(TIMEFRAMES[tf_label])
            try:
                clean_df = clean(raw_df, symbol, tf_label, interval_min)
            except Exception as exc:
                logger.error("Cleaning failed for %s [%s]: %s", symbol, tf_label, exc)
                log_download(symbol, tf_label, date_from, date_to, len(raw_df), "partial", str(exc))
                continue

            all_clean[(symbol, tf_label)] = clean_df

            # Step 4: Resample if this is 1-min data
            if tf_label == "1min" and not clean_df.empty:
                try:
                    resample_all(clean_df, symbol)
                except Exception as exc:
                    logger.error("Resampling failed for %s: %s", symbol, exc)

            # Step 5: Store to DB
            try:
                upsert_candles(clean_df, symbol, tf_label, is_clean=True)
            except Exception as exc:
                logger.error("DB upsert failed for %s [%s]: %s", symbol, tf_label, exc)

            # Step 6: Log quality
            val_result = validate_symbol(clean_df, symbol, tf_label, interval_min)
            log_quality_check(
                symbol, tf_label, "completeness",
                f"{val_result['completeness']:.2%}",
                val_result["completeness"],
                val_result["flagged"],
            )

    # Step 7: Validation report
    logger.info("=== Running validation report ===")
    run_validation_report(all_clean, interval_map)

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
