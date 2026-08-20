"""
Integration test for the full feature pipeline using synthetic candle data.
Does NOT require a Fyers connection or real Parquet files on disk.
"""
import numpy as np
import pandas as pd
import pytz
import pytest

from features.market_context import add_nifty_context, add_time_features, compute_nifty_features
from features.momentum import add_momentum
from features.price_action import add_price_action
from features.regime import add_regime
from features.trend import add_trend
from features.volatility import add_volatility
from features.volume import add_volume

IST = pytz.timezone("Asia/Kolkata")

_BASE_COLS = {"datetime", "open", "high", "low", "close", "volume"}

# Columns that are categorical strings — excluded from numeric NaN checks
_STRING_COLS = {"trend_regime", "vol_regime"}


def _synthetic_candles(n: int = 200, start: str = "2023-01-02 09:15:00") -> pd.DataFrame:
    """Generate n 1-min synthetic candles spread over multiple days."""
    rng = np.random.default_rng(99)
    base = pd.Timestamp(start, tz=IST)
    rows = []
    market_open = pd.Timedelta(hours=9, minutes=15)
    market_close = pd.Timedelta(hours=15, minutes=30)
    session_len = int((market_close - market_open).total_seconds() / 60)  # 375

    candle = 0
    day = 0
    while candle < n:
        day_start = (base + pd.Timedelta(days=day)).normalize().tz_localize(None)
        day_start = pd.Timestamp(day_start, tz=IST) + market_open
        for m in range(session_len):
            if candle >= n:
                break
            close = 100 + rng.standard_normal() * 0.5
            rows.append(
                dict(
                    datetime=day_start + pd.Timedelta(minutes=m),
                    open=close - abs(rng.standard_normal() * 0.1),
                    high=close + abs(rng.standard_normal() * 0.2),
                    low=close - abs(rng.standard_normal() * 0.2),
                    close=close,
                    volume=int(1000 + rng.integers(0, 500)),
                )
            )
            candle += 1
        day += 1

    return pd.DataFrame(rows)


def _run_pipeline(n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _synthetic_candles(n)
    nifty_raw = _synthetic_candles(n, start="2023-01-02 09:15:00")
    nifty_feat = compute_nifty_features(nifty_raw)

    df = add_price_action(df)
    df = add_trend(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume(df)
    df = add_time_features(df)
    df = add_nifty_context(df, nifty_feat)
    df = add_regime(df)

    feature_cols = [c for c in df.columns if c not in _BASE_COLS]
    before = len(df)
    df_clean = df.dropna(subset=feature_cols)
    return df_clean, df


def test_pipeline_runs_without_error():
    out, _ = _run_pipeline(200)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_no_nan_after_dropna():
    out, _ = _run_pipeline(200)
    numeric_cols = [c for c in out.columns if c not in _BASE_COLS | _STRING_COLS]
    assert not out[numeric_cols].isnull().any(axis=None).any()


def test_expected_column_count():
    out, _ = _run_pipeline(200)
    feature_cols = [c for c in out.columns if c not in _BASE_COLS]
    # Spec targets ~54 features; allow ±5 for minor implementation differences
    assert 49 <= len(feature_cols) <= 59, f"Got {len(feature_cols)} feature columns"


def test_nan_warmup_rows_dropped():
    out, raw = _run_pipeline(200)
    # Pipeline should drop rows that had NaN from indicator warmup
    assert len(out) < len(raw), "No NaN rows were dropped — warmup not working"
    # But not all rows should be dropped
    assert len(out) > 100


def test_rsi_in_bounds():
    out, _ = _run_pipeline(200)
    assert (out["rsi_14"] >= 0).all() and (out["rsi_14"] <= 100).all()


def test_regime_enc_valid_values():
    out, _ = _run_pipeline(200)
    assert set(out["trend_regime_enc"].unique()).issubset({-1, 0, 1})
    assert set(out["vol_regime_enc"].unique()).issubset({-1, 0, 1})


def test_close_position_in_unit_interval():
    out, _ = _run_pipeline(200)
    assert (out["close_position"] >= 0).all() and (out["close_position"] <= 1).all()
