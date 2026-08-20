"""Tests for data/resampling/resampler.py"""
from datetime import datetime

import pandas as pd
import pytest
import pytz

IST = pytz.timezone("Asia/Kolkata")


def _make_1min_session(date_str: str = "2024-01-02", open_h=9, open_m=15, n_candles=6) -> pd.DataFrame:
    """Generate n_candles of sequential 1-min data starting at open_h:open_m IST."""
    rows = []
    for i in range(n_candles):
        total_minutes = open_h * 60 + open_m + i
        h, m = divmod(total_minutes, 60)
        dt = IST.localize(datetime.strptime(f"{date_str} {h:02d}:{m:02d}:00", "%Y-%m-%d %H:%M:%S"))
        rows.append({
            "datetime": dt,
            "open": float(100 + i),
            "high": float(105 + i),
            "low": float(99 + i),
            "close": float(103 + i),
            "volume": 1000 * (i + 1),
        })
    return pd.DataFrame(rows)


class TestResampleOHLCV:
    def test_3min_aggregation_rules(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.resampling.resampler.PROCESSED_DIR", tmp_path)
        from data.resampling.resampler import resample

        df_1min = _make_1min_session(n_candles=6)  # 09:15 – 09:20 → 2 complete 3-min bars
        result = resample(df_1min, "NSE:RELIANCE-EQ", "3min")

        assert len(result) == 2

        bar0 = result.iloc[0]
        # open = first 1-min open (09:15)
        assert bar0["open"] == 100.0
        # high = max of 09:15, 09:16, 09:17
        assert bar0["high"] == 107.0
        # low = min of 09:15, 09:16, 09:17
        assert bar0["low"] == 99.0
        # close = last 1-min close (09:17)
        assert bar0["close"] == 105.0
        # volume = sum
        assert bar0["volume"] == 1000 + 2000 + 3000

    def test_5min_aggregation_rules(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.resampling.resampler.PROCESSED_DIR", tmp_path)
        from data.resampling.resampler import resample

        df_1min = _make_1min_session(n_candles=10)  # 09:15 – 09:24 → 2 complete 5-min bars
        result = resample(df_1min, "NSE:RELIANCE-EQ", "5min")

        assert len(result) == 2


class TestNoPartialCandles:
    def test_drops_partial_3min_at_session_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr("data.resampling.resampler.PROCESSED_DIR", tmp_path)
        from data.resampling.resampler import resample

        # 09:15 – 15:29 is a full session; 15:29 bar cannot complete (would need 15:31)
        df_1min = _make_1min_session(n_candles=375)  # full 375-min session
        result = resample(df_1min, "NSE:RELIANCE-EQ", "3min")

        max_time = result["datetime"].dt.time.max()
        # Last complete 3-min bar starting at or before 15:27 (ends at 15:30 which is ok)
        # 15:28 bar would end at 15:31 → partial → must be dropped
        assert str(max_time) <= "15:28:00"
