"""
Dataset builder — joins features + labels into ML-ready DataFrames.

Session filters applied by default:
  - Drop first 30 minutes of each session (VWAP/OBV unreliable during opening)
  - Drop last 15 minutes of each session (closing auction distorts features;
    these rows are labeled 0 anyway due to horizon truncation)
"""

import logging
from pathlib import Path

import pandas as pd

from app.config import FEATURES_DIR

logger = logging.getLogger(__name__)

LABELS_DIR = Path(__file__).resolve().parent.parent / "data_storage" / "labels"

# NSE session start and end in minutes from midnight
_SESSION_START_MINUTE = 9 * 60 + 15   # 09:15
_SESSION_END_MINUTE   = 15 * 60 + 30  # 15:30


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def build_dataset(
    symbol: str,
    timeframe: str,
    variant: str,
    exclude_opening_minutes: int = 30,
    exclude_closing_minutes: int = 15,
) -> pd.DataFrame:
    """
    Load features + labels for one symbol/timeframe/variant combination.
    Apply session time filters and return ML-ready DataFrame.

    Columns: all 64 feature columns + 'label' column
    Index: datetime (timezone-aware)
    """
    safe = _safe_symbol(symbol)

    feat_path  = FEATURES_DIR / f"{safe}_{timeframe}_features.parquet"
    label_path = LABELS_DIR   / f"{safe}_{timeframe}_{variant}_labels.parquet"

    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    feat_df  = pd.read_parquet(feat_path)
    label_df = pd.read_parquet(label_path)

    # Ensure datetime column is parsed
    feat_df["datetime"]  = pd.to_datetime(feat_df["datetime"])
    label_df["datetime"] = pd.to_datetime(label_df["datetime"])

    feat_df  = feat_df.set_index("datetime")
    label_df = label_df.set_index("datetime")

    # Inner join on datetime index
    df = feat_df.join(label_df[["label"]], how="inner")

    # Apply session time filters
    dt = df.index
    session_minute = dt.hour * 60 + dt.minute

    open_cutoff  = _SESSION_START_MINUTE + exclude_opening_minutes
    close_cutoff = _SESSION_END_MINUTE   - exclude_closing_minutes

    mask = (session_minute >= open_cutoff) & (session_minute < close_cutoff)
    df = df[mask]

    logger.debug(
        f"build_dataset {symbol} {timeframe} {variant}: {len(df)} rows after session filter"
    )
    return df


def build_full_dataset(
    symbols: list[str],
    timeframe: str,
    variant: str,
    exclude_opening_minutes: int = 30,
    exclude_closing_minutes: int = 15,
) -> pd.DataFrame:
    """
    Build and concatenate datasets for all symbols.
    Adds 'symbol' column. Preserves chronological order (no global shuffle).

    Important: temporal order is preserved for train/val/test split in Phase 5.
    """
    parts = []
    for symbol in symbols:
        try:
            df = build_dataset(
                symbol,
                timeframe,
                variant,
                exclude_opening_minutes,
                exclude_closing_minutes,
            )
            df = df.copy()
            df["symbol"] = symbol
            parts.append(df)
            logger.info(f"  Loaded {symbol} {timeframe} {variant}: {len(df)} rows")
        except FileNotFoundError as exc:
            logger.warning(f"  Skipping {symbol}: {exc}")

    if not parts:
        raise ValueError("No datasets could be loaded — check that label files exist.")

    combined = pd.concat(parts, axis=0)
    combined = combined.sort_index()
    return combined
