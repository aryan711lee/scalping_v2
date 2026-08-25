"""Tests for features/momentum_quality.py."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.momentum_quality import add_momentum_quality


def _base_df(n=20) -> pd.DataFrame:
    """Minimal DataFrame with all columns momentum_quality depends on."""
    dates = pd.date_range("2024-01-02 09:45", periods=n, freq="3min")
    close = pd.Series([100.0 + i * 0.1 for i in range(n)])
    open_ = close - 0.05  # slightly below close (bullish candles)
    return pd.DataFrame({
        "datetime": dates,
        "open": open_.values,
        "high": (close + 0.1).values,
        "low": (close - 0.2).values,
        "close": close.values,
        "volume": [10000] * n,
        # Dependencies from add_price_action
        "price_return_1": close.pct_change(1).values,
        # Dependencies from add_volume
        "volume_ratio": [1.2] * n,
        # Dependencies from add_trend
        "ema_9": close.ewm(span=9, adjust=False).mean().values,
    })


def test_candle_consistency_all_bullish():
    """All bullish candles → candle_consistency_3 == +1 after warmup."""
    df = _base_df(10)
    result = add_momentum_quality(df)
    # Rows 2+ (index 2 onward) should be +1 since all candles are bullish
    assert result["candle_consistency_3"].iloc[4] == 1
    assert result["candle_consistency_3"].iloc[9] == 1


def test_candle_consistency_all_bearish():
    """All bearish candles → candle_consistency_3 == -1 after warmup."""
    df = _base_df(10)
    # Make candles bearish: close < open
    df["close"] = df["open"] - 0.10
    df["price_return_1"] = df["close"].pct_change(1)
    result = add_momentum_quality(df)
    assert result["candle_consistency_3"].iloc[4] == -1
    assert result["candle_consistency_3"].iloc[9] == -1


def test_candle_consistency_mixed():
    """Mixed bullish/bearish → candle_consistency_3 == 0."""
    df = _base_df(10)
    # Alternate bullish/bearish
    for i in range(0, 10, 2):
        df.loc[i, "close"] = df.loc[i, "open"] - 0.05  # bearish
    df["price_return_1"] = df["close"].pct_change(1)
    result = add_momentum_quality(df)
    # At least some rows should be 0 (mixed)
    assert (result["candle_consistency_3"] == 0).any()


def test_volume_price_trend_positive_when_price_up_high_volume():
    """Positive price return + volume_ratio > 1 → volume_price_trend > 0."""
    df = _base_df(10)
    df["price_return_1"] = 0.005  # positive return
    df["volume_ratio"] = 2.0
    result = add_momentum_quality(df)
    assert (result["volume_price_trend"].iloc[1:] > 0).all()


def test_volume_price_trend_clipped_at_limits():
    """volume_price_trend is clipped to [-5, +5]."""
    df = _base_df(10)
    df["price_return_1"] = 0.01
    df["volume_ratio"] = 100.0  # extreme ratio
    result = add_momentum_quality(df)
    assert result["volume_price_trend"].max() <= 5.0
    assert result["volume_price_trend"].min() >= -5.0


def test_volume_price_trend_negative_when_price_down():
    """Negative price return + volume_ratio > 1 → volume_price_trend < 0."""
    df = _base_df(10)
    df["price_return_1"] = -0.005
    df["volume_ratio"] = 2.0
    result = add_momentum_quality(df)
    assert (result["volume_price_trend"].iloc[1:] < 0).all()


def test_ema9_slope_positive_when_rising():
    """Steadily rising EMA9 → ema9_slope > 0."""
    df = _base_df(20)
    # EMA already computed from rising prices
    result = add_momentum_quality(df)
    # After warmup, slope should be positive
    assert result["ema9_slope"].iloc[5:].dropna().mean() > 0


def test_ema9_slope_near_zero_for_flat():
    """Flat EMA9 → ema9_slope near zero."""
    df = _base_df(20)
    df["ema_9"] = 100.0  # completely flat
    result = add_momentum_quality(df)
    flat_slopes = result["ema9_slope"].iloc[5:].dropna()
    assert (flat_slopes.abs() < 1e-10).all()


def test_no_nan_after_warmup():
    """After warmup period (rows 2+), no NaN in any new feature."""
    df = _base_df(30)
    result = add_momentum_quality(df)
    new_cols = ["candle_consistency_3", "volume_price_trend", "ema9_slope"]
    # Rows 3+ (index >= 3) should have no NaN
    subset = result[new_cols].iloc[5:]
    assert not subset.isnull().any().any(), f"NaN found: {subset.isnull().sum()}"


def test_output_has_all_three_features():
    """Output DataFrame contains all three new feature columns."""
    df = _base_df(10)
    result = add_momentum_quality(df)
    for col in ["candle_consistency_3", "volume_price_trend", "ema9_slope"]:
        assert col in result.columns, f"Missing column: {col}"


def test_candle_consistency_values_in_set():
    """candle_consistency_3 must only contain -1, 0, or +1."""
    df = _base_df(30)
    result = add_momentum_quality(df)
    valid = {-1, 0, 1}
    unique = set(result["candle_consistency_3"].dropna().unique())
    assert unique.issubset(valid), f"Unexpected values: {unique - valid}"
