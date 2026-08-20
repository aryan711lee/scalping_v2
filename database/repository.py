import logging
from datetime import date, datetime

import pandas as pd
from sqlalchemy import text

import app.logger  # noqa: F401
from database.database import get_session, get_engine
from database.models import Candle, DataDownloadLog, DataQualityLog

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000


def upsert_candles(df: pd.DataFrame, symbol: str, timeframe: str, is_clean: bool = True) -> int:
    """
    Insert candles from DataFrame; skip rows that already exist (INSERT OR IGNORE).
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    engine = get_engine()
    inserted = 0
    records = df.to_dict(orient="records")

    sql = text("""
        INSERT OR IGNORE INTO candles
            (symbol, timeframe, datetime, open, high, low, close, volume, is_clean)
        VALUES
            (:symbol, :timeframe, :datetime, :open, :high, :low, :close, :volume, :is_clean)
    """)

    try:
        with engine.begin() as conn:
            for i in range(0, len(records), _BATCH_SIZE):
                batch = records[i : i + _BATCH_SIZE]
                params = [
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "datetime": str(row["datetime"]),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                        "is_clean": int(is_clean),
                    }
                    for row in batch
                ]
                conn.execute(sql, params)
                inserted += len(batch)
                logger.debug("Inserted batch of %d candles (%s [%s])", len(batch), symbol, timeframe)

        logger.info("Upserted %d candles for %s [%s]", inserted, symbol, timeframe)
    except Exception as exc:
        logger.error("Failed to upsert candles for %s [%s]: %s", symbol, timeframe, exc)
        raise

    return inserted


def log_download(
    symbol: str,
    timeframe: str,
    date_from: str,
    date_to: str,
    records_count: int,
    status: str,
    error_message: str | None = None,
) -> None:
    session = get_session()
    try:
        entry = DataDownloadLog(
            symbol=symbol,
            timeframe=timeframe,
            date_from=datetime.strptime(date_from, "%Y-%m-%d").date(),
            date_to=datetime.strptime(date_to, "%Y-%m-%d").date(),
            records_count=records_count,
            status=status,
            error_message=error_message,
            downloaded_at=datetime.utcnow(),
        )
        session.add(entry)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to log download: %s", exc)
    finally:
        session.close()


def log_quality_check(
    symbol: str,
    timeframe: str,
    check_name: str,
    result: str,
    value: float,
    flagged: bool,
) -> None:
    session = get_session()
    try:
        entry = DataQualityLog(
            symbol=symbol,
            timeframe=timeframe,
            check_name=check_name,
            result=result,
            value=value,
            flagged=flagged,
            checked_at=datetime.utcnow(),
        )
        session.add(entry)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to log quality check: %s", exc)
    finally:
        session.close()


def get_candles(
    symbol: str,
    timeframe: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    session = get_session()
    try:
        query = session.query(Candle).filter(
            Candle.symbol == symbol,
            Candle.timeframe == timeframe,
        )
        if date_from:
            query = query.filter(Candle.datetime >= date_from)
        if date_to:
            query = query.filter(Candle.datetime <= date_to)
        query = query.order_by(Candle.datetime)
        rows = query.all()
        if not rows:
            return pd.DataFrame()
        data = [
            {
                "datetime": r.datetime,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    finally:
        session.close()
