from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "datetime", name="uq_candle"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False, index=True)
    datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    is_clean = Column(Boolean, default=True)


class DataDownloadLog(Base):
    __tablename__ = "data_download_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    timeframe = Column(String)
    date_from = Column(Date)
    date_to = Column(Date)
    records_count = Column(Integer)
    status = Column(String)  # 'success', 'failed', 'partial'
    error_message = Column(Text)
    downloaded_at = Column(DateTime, default=datetime.utcnow)


class DataQualityLog(Base):
    __tablename__ = "data_quality_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String)
    timeframe = Column(String)
    check_name = Column(String)
    result = Column(String)
    value = Column(Float)
    flagged = Column(Boolean)
    checked_at = Column(DateTime, default=datetime.utcnow)
