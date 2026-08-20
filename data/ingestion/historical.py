import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import app.logger  # noqa: F401
from app.config import (
    WATCHLIST,
    TIMEFRAMES,
    HISTORICAL_DATA_START,
    RAW_DIR,
    API_RATE_LIMIT_SLEEP,
)
from data.ingestion.fyers_client import FyersClient

logger = logging.getLogger(__name__)


def _raw_path(symbol: str, timeframe_label: str, date_from: str, date_to: str) -> Path:
    safe_symbol = symbol.replace(":", "_").replace("-", "_")
    return RAW_DIR / f"{safe_symbol}_{timeframe_label}_{date_from}_{date_to}.csv"


def download_symbol(
    client: FyersClient,
    symbol: str,
    timeframe_label: str,
    date_from: str,
    date_to: str,
) -> pd.DataFrame | None:
    out_path = _raw_path(symbol, timeframe_label, date_from, date_to)

    if out_path.exists():
        logger.info("Skipping %s [%s] — file already exists: %s", symbol, timeframe_label, out_path.name)
        return pd.read_csv(out_path, parse_dates=["datetime"])

    resolution = TIMEFRAMES[timeframe_label]
    logger.info("Downloading %s [%s] from %s to %s ...", symbol, timeframe_label, date_from, date_to)

    try:
        df = client.get_historical_candles(symbol, resolution, date_from, date_to)
    except Exception as exc:
        logger.error("Download failed for %s [%s]: %s", symbol, timeframe_label, exc)
        return None

    if df is None or df.empty:
        logger.warning("No candles returned for %s [%s]", symbol, timeframe_label)
        return None

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d candles → %s", len(df), out_path.name)
    return df


def download_all(
    date_from: str | None = None,
    date_to: str | None = None,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Download raw OHLCV data for all (or specified) symbols and timeframes.
    Returns mapping of (symbol, timeframe_label) → DataFrame.
    """
    client = FyersClient()

    _date_from = date_from or HISTORICAL_DATA_START
    _date_to = date_to or date.today().strftime("%Y-%m-%d")
    _symbols = symbols or WATCHLIST
    _timeframes = timeframes or list(TIMEFRAMES.keys())

    results: dict[tuple[str, str], pd.DataFrame] = {}
    total = len(_symbols) * len(_timeframes)
    done = 0

    for symbol in _symbols:
        for tf_label in _timeframes:
            df = download_symbol(client, symbol, tf_label, _date_from, _date_to)
            if df is not None:
                results[(symbol, tf_label)] = df
            done += 1
            logger.info("Progress: %d/%d", done, total)
            time.sleep(API_RATE_LIMIT_SLEEP)

    logger.info(
        "Download complete. %d/%d combinations succeeded.",
        len(results),
        total,
    )
    return results
