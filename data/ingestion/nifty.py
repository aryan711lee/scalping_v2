"""Download and load NIFTY 50 index candles for use as market context features."""
import logging
from datetime import date

import pandas as pd

from app.config import (
    HISTORICAL_DATA_START,
    NIFTY_SAFE,
    NIFTY_SYMBOL,
    PROCESSED_DIR,
    RAW_DIR,
    TIMEFRAMES,
)
from data.cleaning.cleaner import clean
from data.ingestion.fyers_client import FyersClient

logger = logging.getLogger(__name__)

_INTERVAL_MINUTES = {"1min": 1, "3min": 3, "5min": 5}


def fetch_nifty(start_date: str = HISTORICAL_DATA_START, end_date: str | None = None) -> None:
    """Download NIFTY 50 candles for all three timeframes and save cleaned Parquets."""
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    client = FyersClient()

    for tf_label, tf_resolution in TIMEFRAMES.items():
        clean_path = PROCESSED_DIR / f"{NIFTY_SAFE}_{tf_label}_clean.parquet"
        if clean_path.exists():
            logger.info(f"NIFTY {tf_label}: already present at {clean_path}, skipping download.")
            continue

        logger.info(f"NIFTY {tf_label}: downloading {start_date} → {end_date} …")
        df = client.get_historical_candles(NIFTY_SYMBOL, tf_resolution, start_date, end_date)

        raw_path = RAW_DIR / f"{NIFTY_SAFE}_{tf_label}_raw.csv"
        df.to_csv(raw_path, index=False)
        logger.info(f"NIFTY {tf_label}: saved raw CSV ({len(df)} rows) → {raw_path}")

        df_clean = clean(df, NIFTY_SYMBOL, tf_label, _INTERVAL_MINUTES[tf_label])
        df_clean.to_parquet(clean_path, index=False)
        logger.info(f"NIFTY {tf_label}: saved clean Parquet ({len(df_clean)} rows) → {clean_path}")


def load_nifty(timeframe: str) -> pd.DataFrame:
    """Load the cleaned NIFTY parquet for the given timeframe."""
    path = PROCESSED_DIR / f"{NIFTY_SAFE}_{timeframe}_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"NIFTY {timeframe} data not found at {path}. Run fetch_nifty() first."
        )
    return pd.read_parquet(path)
