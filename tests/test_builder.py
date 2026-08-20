"""Tests for datasets/builder.py"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from datasets.builder import build_dataset, build_full_dataset, _safe_symbol


# ---------------------------------------------------------------------------
# Fixtures — synthetic feature + label DataFrames
# ---------------------------------------------------------------------------

def _make_feat_df(n: int = 200, start_hour: int = 9, start_min: int = 15) -> pd.DataFrame:
    """Synthetic feature DataFrame matching the real schema (simplified)."""
    times = pd.date_range(
        start=pd.Timestamp(f"2024-01-15 {start_hour:02d}:{start_min:02d}:00",
                           tz="Asia/Kolkata"),
        periods=n, freq="3min",
    )
    rng = np.random.default_rng(42)
    data = {"datetime": times}
    # 64 feature-like columns
    for col in [
        "open", "high", "low", "close", "volume",
        "price_return_1", "ema_9", "ema_21", "ema_50", "vwap", "rsi_14",
        "atr_14", "bb_upper", "bb_lower", "volume_ratio", "obv",
        "nifty_trend", "trend_regime_enc", "vol_regime_enc", "time_sin",
    ]:
        data[col] = rng.uniform(90, 110, n)
    return pd.DataFrame(data)


def _make_label_df(feat_df: pd.DataFrame, variant: str = "L1") -> pd.DataFrame:
    n = len(feat_df)
    rng = np.random.default_rng(0)
    labels = rng.choice([-1, 0, 1], size=n, p=[0.08, 0.84, 0.08]).astype(np.int8)
    return pd.DataFrame({
        "datetime": feat_df["datetime"],
        "label":    labels,
        "variant":  variant,
    })


@pytest.fixture
def mock_data(tmp_path):
    """Write synthetic parquet files and patch FEATURES_DIR + LABELS_DIR."""
    feat_df  = _make_feat_df(200)
    label_df = _make_label_df(feat_df)

    feat_dir  = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feat_dir.mkdir()
    label_dir.mkdir()

    symbol = "NSE:RELIANCE-EQ"
    safe   = _safe_symbol(symbol)
    tf     = "3min"
    variant = "L1"

    feat_df.to_parquet(feat_dir / f"{safe}_{tf}_features.parquet", index=False)
    label_df.to_parquet(label_dir / f"{safe}_{tf}_{variant}_labels.parquet", index=False)

    return {
        "symbol": symbol, "timeframe": tf, "variant": variant,
        "feat_dir": feat_dir, "label_dir": label_dir,
        "feat_df": feat_df, "label_df": label_df,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_dataset_has_label_column(mock_data):
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        df = build_dataset(mock_data["symbol"], mock_data["timeframe"], mock_data["variant"])
    assert "label" in df.columns


def test_build_dataset_no_nan(mock_data):
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        df = build_dataset(mock_data["symbol"], mock_data["timeframe"], mock_data["variant"])
    assert not df.isnull().any().any(), "Dataset contains NaN values"


def test_opening_minutes_excluded(mock_data):
    """No rows should fall within the first 30 minutes of the session (09:15-09:44)."""
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        df = build_dataset(mock_data["symbol"], mock_data["timeframe"], mock_data["variant"],
                           exclude_opening_minutes=30)
    session_minute = df.index.hour * 60 + df.index.minute
    assert (session_minute >= 9 * 60 + 45).all(), (
        "Some rows fall in the first 30 minutes (before 09:45)"
    )


def test_closing_minutes_excluded(mock_data):
    """No rows should fall in last 15 minutes of session (15:15-15:30)."""
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        df = build_dataset(mock_data["symbol"], mock_data["timeframe"], mock_data["variant"],
                           exclude_closing_minutes=15)
    session_minute = df.index.hour * 60 + df.index.minute
    assert (session_minute < 15 * 60 + 15).all(), (
        "Some rows fall in the last 15 minutes (>= 15:15)"
    )


def test_chronological_order_preserved(mock_data):
    """Datetime index must be monotonically increasing."""
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        df = build_dataset(mock_data["symbol"], mock_data["timeframe"], mock_data["variant"])
    assert df.index.is_monotonic_increasing, "Datetime index is not monotonically increasing"


def test_multi_symbol_adds_symbol_column(tmp_path):
    """build_full_dataset must add a 'symbol' column."""
    feat_dir  = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feat_dir.mkdir()
    label_dir.mkdir()

    symbols = ["NSE:TCS-EQ", "NSE:INFY-EQ"]
    tf, variant = "3min", "L1"

    for symbol in symbols:
        safe = _safe_symbol(symbol)
        feat_df  = _make_feat_df(200)
        label_df = _make_label_df(feat_df, variant)
        feat_df.to_parquet(feat_dir  / f"{safe}_{tf}_features.parquet", index=False)
        label_df.to_parquet(label_dir / f"{safe}_{tf}_{variant}_labels.parquet", index=False)

    with (
        patch("datasets.builder.FEATURES_DIR", feat_dir),
        patch("datasets.builder.LABELS_DIR",   label_dir),
    ):
        df = build_full_dataset(symbols, tf, variant)

    assert "symbol" in df.columns, "build_full_dataset did not add 'symbol' column"
    assert set(df["symbol"].unique()) == set(symbols)


def test_multi_symbol_chronological_order(tmp_path):
    """build_full_dataset result must be sorted by datetime."""
    feat_dir  = tmp_path / "features"
    label_dir = tmp_path / "labels"
    feat_dir.mkdir()
    label_dir.mkdir()

    symbols = ["NSE:TCS-EQ", "NSE:INFY-EQ"]
    tf, variant = "3min", "L1"

    for symbol in symbols:
        safe = _safe_symbol(symbol)
        feat_df  = _make_feat_df(200)
        label_df = _make_label_df(feat_df, variant)
        feat_df.to_parquet(feat_dir  / f"{safe}_{tf}_features.parquet", index=False)
        label_df.to_parquet(label_dir / f"{safe}_{tf}_{variant}_labels.parquet", index=False)

    with (
        patch("datasets.builder.FEATURES_DIR", feat_dir),
        patch("datasets.builder.LABELS_DIR",   label_dir),
    ):
        df = build_full_dataset(symbols, tf, variant)

    assert df.index.is_monotonic_increasing, "build_full_dataset result is not sorted"


def test_missing_feature_file_raises(mock_data):
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        with pytest.raises(FileNotFoundError, match="Feature file not found"):
            build_dataset("NSE:FAKE-EQ", "3min", "L1")


def test_missing_label_file_raises(mock_data):
    with (
        patch("datasets.builder.FEATURES_DIR", mock_data["feat_dir"]),
        patch("datasets.builder.LABELS_DIR",   mock_data["label_dir"]),
    ):
        with pytest.raises(FileNotFoundError, match="Label file not found"):
            build_dataset(mock_data["symbol"], mock_data["timeframe"], "L9")
