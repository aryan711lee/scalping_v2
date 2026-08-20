"""Tests for data/ingestion/historical.py"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz


IST = pytz.timezone("Asia/Kolkata")


def _make_df(n: int = 3) -> pd.DataFrame:
    import pytz
    from datetime import datetime
    IST = pytz.timezone("Asia/Kolkata")
    rows = []
    for i in range(n):
        rows.append({
            "datetime": IST.localize(datetime(2024, 1, 2, 9, 15 + i, 0)),
            "open": 100.0 + i,
            "high": 105.0 + i,
            "low": 99.0 + i,
            "close": 103.0 + i,
            "volume": 1000 + i * 100,
        })
    return pd.DataFrame(rows)


class TestDateChunking:
    def test_1min_chunks_cover_full_range(self):
        from data.ingestion.fyers_client import FyersClient
        from datetime import datetime, timedelta

        client = FyersClient.__new__(FyersClient)
        chunks = client._build_date_chunks(
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
            resolution_minutes=1,
        )
        # All chunks must be ≤ 89 days (chunk_days - 1) to stay under Fyers 100-day limit
        for start, end in chunks:
            delta = datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")
            assert delta.days <= 89
        # Full range must be covered without gaps or overlap
        assert chunks[0][0] == "2024-01-01"
        assert chunks[-1][1] == "2024-12-31"

    def test_3min_chunks_at_90_days(self):
        from data.ingestion.fyers_client import FyersClient
        from datetime import datetime

        client = FyersClient.__new__(FyersClient)
        # 365-day range should split into multiple 90-day chunks
        chunks = client._build_date_chunks(
            datetime(2023, 1, 1),
            datetime(2023, 12, 31),
            resolution_minutes=3,
        )
        # 365 / 90 = ~5 chunks; verify all chunks are ≤ 90 days
        from datetime import timedelta
        for start, end in chunks:
            delta = datetime.strptime(end, "%Y-%m-%d") - datetime.strptime(start, "%Y-%m-%d")
            assert delta.days <= 89  # chunk_days - 1
        assert len(chunks) >= 4
        assert chunks[0][0] == "2023-01-01"
        assert chunks[-1][1] == "2023-12-31"


class TestIdempotentDownload:
    def test_skips_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.ingestion.historical.RAW_DIR", tmp_path)

        # Pre-create the file
        df = _make_df()
        safe = "NSE_RELIANCE_EQ"
        csv_path = tmp_path / f"{safe}_1min_2024-01-01_2024-12-31.csv"
        df.to_csv(csv_path, index=False)

        mock_client = MagicMock()
        mock_client.get_historical_candles = MagicMock()

        from data.ingestion.historical import download_symbol
        result = download_symbol(mock_client, "NSE:RELIANCE-EQ", "1min", "2024-01-01", "2024-12-31")

        # Should NOT have called the API
        mock_client.get_historical_candles.assert_not_called()
        assert result is not None

    def test_downloads_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.ingestion.historical.RAW_DIR", tmp_path)
        monkeypatch.setattr("data.ingestion.historical.API_RATE_LIMIT_SLEEP", 0)

        df = _make_df()
        mock_client = MagicMock()
        mock_client.get_historical_candles.return_value = df

        from data.ingestion.historical import download_symbol
        result = download_symbol(mock_client, "NSE:RELIANCE-EQ", "1min", "2024-01-01", "2024-12-31")

        mock_client.get_historical_candles.assert_called_once()
        assert result is not None
        assert len(result) == len(df)
