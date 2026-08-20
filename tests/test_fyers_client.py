"""Tests for data/ingestion/fyers_client.py"""
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_epoch(dt_str: str) -> int:
    """Return UTC epoch seconds for an IST datetime string."""
    import pytz
    from datetime import datetime
    IST = pytz.timezone("Asia/Kolkata")
    dt = IST.localize(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S"))
    return int(dt.timestamp())


def _mock_candles():
    return [
        [_make_epoch("2024-01-02 09:15:00"), 100.0, 105.0, 99.0, 103.0, 5000],
        [_make_epoch("2024-01-02 09:16:00"), 103.0, 106.0, 102.0, 104.0, 3000],
        [_make_epoch("2024-01-02 09:17:00"), 104.0, 107.0, 103.0, 105.0, 4000],
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFyersClientCredentials:
    def test_credentials_load_from_env(self, monkeypatch):
        monkeypatch.setenv("FYERS_APP_ID", "test_app_id")
        monkeypatch.setenv("FYERS_ACCESS_TOKEN", "test_token")

        # Re-import config to pick up monkeypatched env
        import importlib
        import app.config as cfg
        importlib.reload(cfg)

        assert cfg.FYERS_APP_ID == "test_app_id"
        assert cfg.FYERS_ACCESS_TOKEN == "test_token"


class TestFyersClientHistoricalFetch:
    @patch("data.ingestion.fyers_client.fyersModel", create=True)
    def test_returns_correct_columns(self, mock_fyers_module):
        mock_instance = MagicMock()
        mock_instance.history.return_value = {
            "code": 200,
            "candles": _mock_candles(),
        }
        mock_fyers_module.FyersModel.return_value = mock_instance

        # Patch the import inside fyers_client
        with patch.dict("sys.modules", {"fyers_apiv3": MagicMock(), "fyers_apiv3.fyersModel": mock_fyers_module}):
            from data.ingestion.fyers_client import FyersClient
            client = FyersClient.__new__(FyersClient)
            client._fyers = mock_instance

            df = client.get_historical_candles("NSE:RELIANCE-EQ", "1", "2024-01-02", "2024-01-02")

        assert set(df.columns) == {"datetime", "open", "high", "low", "close", "volume"}
        assert df.dtypes["open"] == "float64"
        assert df.dtypes["volume"] == "int64"
        assert len(df) == 3

    @patch("data.ingestion.fyers_client.time.sleep", return_value=None)
    def test_retry_logic_on_failure(self, mock_sleep):
        mock_fyers = MagicMock()
        mock_fyers.history.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            {"code": 200, "candles": _mock_candles()},
        ]

        from data.ingestion.fyers_client import FyersClient
        client = FyersClient.__new__(FyersClient)
        client._fyers = mock_fyers

        df = client._fetch_chunk_with_retry("NSE:RELIANCE-EQ", "1", "2024-01-02", "2024-01-02")

        assert mock_fyers.history.call_count == 3
        assert df is not None
        assert len(df) == 3

    @patch("data.ingestion.fyers_client.time.sleep", return_value=None)
    def test_retry_exhausted_returns_none(self, mock_sleep):
        mock_fyers = MagicMock()
        mock_fyers.history.side_effect = Exception("always fails")

        from data.ingestion.fyers_client import FyersClient
        client = FyersClient.__new__(FyersClient)
        client._fyers = mock_fyers

        df = client._fetch_chunk_with_retry("NSE:RELIANCE-EQ", "1", "2024-01-02", "2024-01-02")
        assert df is None
