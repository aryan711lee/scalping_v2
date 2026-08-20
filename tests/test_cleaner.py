"""Tests for data/cleaning/cleaner.py"""
from datetime import datetime

import pandas as pd
import pytest
import pytz

IST = pytz.timezone("Asia/Kolkata")


def _make_row(dt_str: str, o=100.0, h=105.0, l=99.0, c=102.0, vol=1000) -> dict:
    return {
        "datetime": IST.localize(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
    }


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


class TestDuplicateRemoval:
    def test_removes_duplicate_datetimes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 09:15:00"),
            _make_row("2024-01-02 09:15:00"),  # duplicate
            _make_row("2024-01-02 09:16:00"),
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        assert len(result) == 2


class TestOHLCConsistency:
    def test_removes_bad_ohlc_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 09:15:00"),          # valid
            _make_row("2024-01-02 09:16:00", h=90.0),  # high < low → removed by OHLC check
            _make_row("2024-01-02 09:17:00"),          # valid
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        # 3 rows: 09:15 (real) + 09:16 (synthetic gap-fill after bad row removed) + 09:17 (real)
        assert len(result) == 3
        filled = result[result["datetime"].dt.strftime("%H:%M") == "09:16"].iloc[0]
        assert filled["volume"] == 0  # synthetic candle has zero volume

    def test_close_above_high_removed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 09:15:00"),
            _make_row("2024-01-02 09:16:00", c=200.0),  # close > high
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        assert len(result) == 1


class TestMarketHoursFilter:
    def test_removes_pre_market_candles(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 08:00:00"),  # pre-market
            _make_row("2024-01-02 09:15:00"),  # open
            _make_row("2024-01-02 15:30:00"),  # close (included)
            _make_row("2024-01-02 16:00:00"),  # post-market
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        times = result["datetime"].dt.strftime("%H:%M").tolist()
        assert "08:00" not in times
        assert "16:00" not in times
        assert "09:15" in times
        assert "15:30" in times


class TestGapFilling:
    def test_fills_single_missing_candle(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 09:15:00", c=100.0),
            # 09:16 is missing
            _make_row("2024-01-02 09:17:00"),
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        times = result["datetime"].dt.strftime("%H:%M").tolist()
        assert "09:16" in times
        # Filled candle should have close of previous candle as OHLC
        filled = result[result["datetime"].dt.strftime("%H:%M") == "09:16"].iloc[0]
        assert filled["close"] == 100.0
        assert filled["volume"] == 0

    def test_fills_two_missing_candles(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 09:15:00"),
            # 09:16 and 09:17 missing
            _make_row("2024-01-02 09:18:00"),
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        assert len(result) == 4  # original 2 + 2 filled

    def test_does_not_fill_large_gaps(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.cleaning.cleaner.PROCESSED_DIR", tmp_path)
        from data.cleaning.cleaner import clean

        rows = [
            _make_row("2024-01-02 09:15:00"),
            # 5 candles missing — should NOT be filled
            _make_row("2024-01-02 09:21:00"),
        ]
        df = _df(*rows)
        result = clean(df, "NSE:RELIANCE-EQ", "1min", 1)
        assert len(result) == 2
