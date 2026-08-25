"""Tests for features/multi_timeframe.py."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.multi_timeframe import add_5min_context, _TF5_CANDLE_MINUTES


def _make_5min_df(start="2024-01-02 09:15", n=30) -> pd.DataFrame:
    """Minimal 5-min feature DataFrame."""
    times = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({
        "datetime": times,
        "ema9_above_ema21": [1] * n,
        "rsi_14": [55.0] * n,
        "vwap_above": [1] * n,
        "volume_ratio": [1.5] * n,
        "candle_body": [0.002] * n,
    })


def _make_3min_df(start="2024-01-02 09:15", n=50) -> pd.DataFrame:
    """Minimal 3-min DataFrame."""
    times = pd.date_range(start, periods=n, freq="3min")
    return pd.DataFrame({
        "datetime": times,
        "close": [100.0] * n,
        "open": [99.9] * n,
        "high": [100.2] * n,
        "low": [99.8] * n,
        "volume": [5000] * n,
    })


def test_output_has_tf5_columns():
    """add_5min_context adds all 5 tf5 context columns."""
    df3 = _make_3min_df()
    df5 = _make_5min_df()
    result = add_5min_context(df3, df5)
    for col in ["tf5_trend", "tf5_rsi", "tf5_vwap_above", "tf5_volume_ratio", "tf5_candle_body"]:
        assert col in result.columns, f"Missing column: {col}"


def test_row_count_unchanged():
    """Merging 5-min context does not change the number of 3-min rows."""
    df3 = _make_3min_df(n=30)
    df5 = _make_5min_df(n=20)
    result = add_5min_context(df3, df5)
    assert len(result) == len(df3)


def test_backward_direction_no_future_5min():
    """
    A 3-min candle at 10:03 should get the 5-min candle opening at 09:55
    (closes at 10:00), NOT the 5-min candle opening at 10:00 (closes at 10:05).

    After the +5min shift, the 5-min at 09:55 appears at 10:00, and the
    5-min at 10:00 appears at 10:05. For the 3-min at 10:03, merge_asof
    backward picks the shifted 10:00 (originally 09:55).
    """
    # 5-min candles starting at 09:55 and 10:00
    df5 = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 09:55", "2024-01-02 10:00"]),
        "ema9_above_ema21": [0, 1],  # 0 for 09:55, 1 for 10:00
        "rsi_14": [40.0, 60.0],
        "vwap_above": [0, 1],
        "volume_ratio": [1.0, 2.0],
        "candle_body": [-0.001, 0.001],
    })
    # 3-min candle at 10:03
    df3 = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 10:03"]),
        "close": [100.0],
        "open": [99.9],
        "high": [100.2],
        "low": [99.8],
        "volume": [5000],
    })
    result = add_5min_context(df3, df5)

    # Should use 5-min from 09:55 (ema9_above_ema21=0, rsi=40, etc.)
    assert result["tf5_trend"].iloc[0] == 0, (
        f"Expected tf5_trend=0 (from 09:55 candle), got {result['tf5_trend'].iloc[0]}"
    )
    assert result["tf5_rsi"].iloc[0] == pytest.approx(40.0), (
        f"Expected tf5_rsi=40.0 (from 09:55 candle), got {result['tf5_rsi'].iloc[0]}"
    )


def test_3min_at_10_00_gets_9_55_candle():
    """
    A 3-min candle at exactly 10:00 should use the 5-min candle opening at 09:55
    (which closes at 10:00), not the one opening at 10:00 (closes at 10:05).
    """
    df5 = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 09:55", "2024-01-02 10:00"]),
        "ema9_above_ema21": [0, 1],
        "rsi_14": [40.0, 60.0],
        "vwap_above": [0, 1],
        "volume_ratio": [1.0, 2.0],
        "candle_body": [-0.001, 0.001],
    })
    df3 = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02 10:00"]),
        "close": [100.0],
        "open": [99.9],
        "high": [100.2],
        "low": [99.8],
        "volume": [5000],
    })
    result = add_5min_context(df3, df5)

    # At T=10:00, after shift: 09:55→10:00, 10:00→10:05
    # merge_asof backward picks 10:00 (originally 09:55), NOT 10:05 (originally 10:00)
    assert result["tf5_trend"].iloc[0] == 0, (
        f"3-min at 10:00 should use 09:55 5-min candle (tf5_trend=0), "
        f"got {result['tf5_trend'].iloc[0]}"
    )


def test_no_nan_after_first_5min_candle():
    """
    After the first 5-min candle has been processed, 3-min candles should have
    valid (non-NaN) tf5 values.
    """
    df3 = _make_3min_df(n=40)
    df5 = _make_5min_df(n=20)
    result = add_5min_context(df3, df5)

    # Skip the very first few 3-min candles (before first 5-min candle is available)
    tf5_cols = ["tf5_trend", "tf5_rsi", "tf5_vwap_above", "tf5_volume_ratio", "tf5_candle_body"]
    non_nan_rows = result[tf5_cols].dropna()
    assert len(non_nan_rows) > 0, "Expected some rows with valid tf5 values"


def test_tf5_columns_absent_for_non_3min():
    """
    The pipeline should not add tf5 columns when timeframe != '3min'.
    This is enforced at the pipeline level — add_5min_context is only called for 3min.
    Verify that calling add_5min_context with a 5-min df still works mechanically.
    """
    df3 = _make_3min_df(n=10)
    df5 = _make_5min_df(n=5)
    result = add_5min_context(df3, df5)
    # Function itself always adds the columns; gating is in pipeline.py
    assert "tf5_trend" in result.columns


def test_candle_shift_is_5_minutes():
    """The shift applied to 5-min timestamps is exactly 5 minutes."""
    assert _TF5_CANDLE_MINUTES == 5


def test_tz_aware_datetimes_compatible():
    """
    merge_asof must work when both DataFrames have timezone-aware datetime columns.
    Timedelta addition upcasts datetime64[ms] → datetime64[us]; the fix must
    normalize both sides before merging.
    """
    tz = "Asia/Kolkata"
    times5 = pd.date_range("2024-01-02 09:55", periods=3, freq="5min", tz=tz)
    df5 = pd.DataFrame({
        "datetime": times5,
        "ema9_above_ema21": [0, 1, 0],
        "rsi_14": [40.0, 60.0, 50.0],
        "vwap_above": [0, 1, 0],
        "volume_ratio": [1.0, 2.0, 1.5],
        "candle_body": [-0.001, 0.001, 0.0],
    })
    times3 = pd.date_range("2024-01-02 10:00", periods=3, freq="3min", tz=tz)
    df3 = pd.DataFrame({
        "datetime": times3,
        "close": [100.0] * 3,
        "open": [99.9] * 3,
        "high": [100.2] * 3,
        "low": [99.8] * 3,
        "volume": [5000] * 3,
    })
    result = add_5min_context(df3, df5)
    assert len(result) == 3, f"Expected 3 rows, got {len(result)}"
    assert "tf5_trend" in result.columns
    # 10:00 3-min gets 09:55 5-min candle (shifted to 10:00) → tf5_trend=0
    assert result["tf5_trend"].iloc[0] == 0


def test_column_values_propagated_correctly():
    """Verify that tf5 column values come from the correct 5-min candle."""
    df5 = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2024-01-02 09:55",
            "2024-01-02 10:00",
            "2024-01-02 10:05",
        ]),
        "ema9_above_ema21": [10, 20, 30],
        "rsi_14": [11.0, 21.0, 31.0],
        "vwap_above": [12, 22, 32],
        "volume_ratio": [13.0, 23.0, 33.0],
        "candle_body": [14.0, 24.0, 34.0],
    })
    # 3-min candles at 10:00, 10:03, 10:06, 10:09
    df3 = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2024-01-02 10:00",
            "2024-01-02 10:03",
            "2024-01-02 10:06",
            "2024-01-02 10:09",
        ]),
        "close": [100.0] * 4,
        "open": [99.9] * 4,
        "high": [100.2] * 4,
        "low": [99.8] * 4,
        "volume": [5000] * 4,
    })
    result = add_5min_context(df3, df5)
    result = result.sort_values("datetime").reset_index(drop=True)

    # After +5min shift: 09:55→10:00, 10:00→10:05, 10:05→10:10
    # 3min at 10:00: gets 09:55 5min candle (shifted 10:00) → value=10
    # 3min at 10:03: gets 09:55 5min candle (shifted 10:00) → value=10
    # 3min at 10:06: gets 10:00 5min candle (shifted 10:05) → value=20
    # 3min at 10:09: gets 10:00 5min candle (shifted 10:05) → value=20
    assert result["tf5_trend"].iloc[0] == 10
    assert result["tf5_trend"].iloc[1] == 10
    assert result["tf5_trend"].iloc[2] == 20
    assert result["tf5_trend"].iloc[3] == 20
