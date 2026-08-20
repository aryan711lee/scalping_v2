import logging
from pathlib import Path

import pandas as pd
import pytz

import app.logger  # noqa: F401
from app.config import PROCESSED_DIR, MARKET_OPEN, MARKET_CLOSE

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# Gap sizes (in candles) that trigger forward-fill vs log-only
GAP_FILL_MAX = 2


def clean(
    df: pd.DataFrame,
    symbol: str,
    timeframe_label: str,
    interval_minutes: int,
) -> pd.DataFrame:
    """
    Full cleaning pipeline. Returns cleaned DataFrame and saves to Parquet.
    """
    logger.info("Cleaning %s [%s] — %d raw rows", symbol, timeframe_label, len(df))
    df = df.copy()

    # 1. Duplicate removal
    before = len(df)
    df.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
    removed = before - len(df)
    if removed:
        logger.debug("Removed %d duplicate rows", removed)

    # 2. Sort by datetime
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 3. Ensure datetime is IST-aware
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(IST)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(IST)

    # 4. Market hours filter (09:15 – 15:30 IST)
    t = df["datetime"].dt.time
    open_t = pd.to_datetime(MARKET_OPEN).time()
    close_t = pd.to_datetime(MARKET_CLOSE).time()
    before = len(df)
    df = df[(t >= open_t) & (t <= close_t)].copy()
    removed = before - len(df)
    if removed:
        logger.debug("Removed %d pre/post-market rows", removed)

    # 5. Zero / null price removal
    price_cols = ["open", "high", "low", "close"]
    before = len(df)
    mask_null = df[price_cols].isnull().any(axis=1)
    mask_zero = (df[price_cols] == 0).any(axis=1)
    df = df[~(mask_null | mask_zero)].copy()
    removed = before - len(df)
    if removed:
        logger.debug("Removed %d rows with null/zero prices", removed)

    # 6. OHLC consistency check
    before = len(df)
    bad_ohlc = (
        (df["high"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    )
    if bad_ohlc.sum():
        logger.warning("Removing %d rows with OHLC inconsistency", bad_ohlc.sum())
    df = df[~bad_ohlc].copy()
    df.reset_index(drop=True, inplace=True)

    # 7. Volume zero logging (keep rows)
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol:
        logger.info("%d candles with zero volume in %s [%s] — kept for review", zero_vol, symbol, timeframe_label)

    # 8. Gap detection & filling
    df = _handle_gaps(df, symbol, timeframe_label, interval_minutes)

    df.reset_index(drop=True, inplace=True)
    logger.info("Cleaned %s [%s] — %d rows remaining", symbol, timeframe_label, len(df))

    _save_parquet(df, symbol, timeframe_label, suffix="clean")
    return df


def _handle_gaps(
    df: pd.DataFrame,
    symbol: str,
    timeframe_label: str,
    interval_minutes: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    expected_delta = pd.Timedelta(minutes=interval_minutes)
    deltas = df["datetime"].diff().iloc[1:]
    gaps = deltas[deltas > expected_delta]

    if gaps.empty:
        return df

    fill_rows = []
    intraday_gap_count = 0

    for idx in gaps.index:
        prev_row = df.loc[idx - 1]
        curr_dt = df.loc[idx, "datetime"]
        prev_dt = prev_row["datetime"]

        # Skip overnight gaps — they are expected, not missing candles
        if curr_dt.date() != prev_dt.date():
            continue

        gap_candles = int((curr_dt - prev_dt) / expected_delta) - 1
        intraday_gap_count += 1

        if gap_candles <= GAP_FILL_MAX:
            for i in range(1, gap_candles + 1):
                synthetic_dt = prev_dt + i * expected_delta
                fill_rows.append({
                    "datetime": synthetic_dt,
                    "open": prev_row["close"],
                    "high": prev_row["close"],
                    "low": prev_row["close"],
                    "close": prev_row["close"],
                    "volume": 0,
                })
            logger.debug("Forward-filled %d candle(s) at %s", gap_candles, prev_dt)
        else:
            logger.warning(
                "Large intraday gap (%d candles) at %s in %s [%s] — left unfilled",
                gap_candles, prev_dt, symbol, timeframe_label,
            )

    if intraday_gap_count:
        logger.info("%d intraday gaps detected in %s [%s]", intraday_gap_count, symbol, timeframe_label)

    if fill_rows:
        fill_df = pd.DataFrame(fill_rows)
        df = pd.concat([df, fill_df], ignore_index=True)
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Injected %d synthetic fill candles", len(fill_rows))

    return df


def _save_parquet(df: pd.DataFrame, symbol: str, timeframe_label: str, suffix: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace(":", "_").replace("-", "_")
    path = PROCESSED_DIR / f"{safe_symbol}_{timeframe_label}_{suffix}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %s", path.name)
    return path
