import logging
from pathlib import Path

import pandas as pd
import pytz

import app.logger  # noqa: F401
from app.config import PROCESSED_DIR, MARKET_OPEN, MARKET_CLOSE

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

RESAMPLE_RULES = {
    "3min": "3min",
    "5min": "5min",
}

OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample(df: pd.DataFrame, symbol: str, target_timeframe: str) -> pd.DataFrame:
    """
    Resample a cleaned 1-min DataFrame to target_timeframe ('3min' or '5min').
    Returns resampled DataFrame and saves to Parquet.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to resample for %s [%s]", symbol, target_timeframe)
        return df

    rule = RESAMPLE_RULES.get(target_timeframe)
    if rule is None:
        raise ValueError(f"Unsupported target timeframe: {target_timeframe}")

    logger.info("Resampling %s → %s (%d rows in)", symbol, target_timeframe, len(df))

    df = df.copy()
    df = df.set_index("datetime")
    df.index = pd.DatetimeIndex(df.index)

    # Closed-left, label-left: the candle timestamp is the open of the period
    resampled = (
        df.resample(rule, closed="left", label="left")
        .agg(OHLCV_AGG)
        .dropna(subset=["open", "close"])
    )

    # Drop partial candles at session boundary
    resampled = _drop_partial_candles(resampled, target_timeframe)

    resampled.reset_index(inplace=True)
    resampled["volume"] = resampled["volume"].astype("int64")

    logger.info("Resampled %s [%s] — %d candles", symbol, target_timeframe, len(resampled))
    _save_parquet(resampled, symbol, target_timeframe)
    return resampled


def resample_all(df_1min: pd.DataFrame, symbol: str) -> dict[str, pd.DataFrame]:
    """Resample a 1-min DataFrame to all supported higher timeframes."""
    results: dict[str, pd.DataFrame] = {}
    for tf_label in RESAMPLE_RULES:
        results[tf_label] = resample(df_1min, symbol, tf_label)
    return results


def _drop_partial_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Remove candles whose open-time would extend past the session close.
    E.g. a 3-min bar at 15:29 would end at 15:32 — beyond 15:30.
    """
    minutes = int(timeframe.replace("min", ""))
    close_t = pd.to_datetime(MARKET_CLOSE, format="%H:%M").time()

    def _would_be_complete(ts: pd.Timestamp) -> bool:
        end = ts + pd.Timedelta(minutes=minutes) - pd.Timedelta(seconds=1)
        return end.time() <= close_t

    mask = pd.Series([_would_be_complete(ts) for ts in df.index], index=df.index)
    dropped = (~mask).sum()
    if dropped:
        logger.debug("Dropped %d partial candles at session boundary", dropped)
    return df[mask]


def _save_parquet(df: pd.DataFrame, symbol: str, timeframe_label: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace(":", "_").replace("-", "_")
    path = PROCESSED_DIR / f"{safe_symbol}_{timeframe_label}_resampled.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %s", path.name)
    return path
