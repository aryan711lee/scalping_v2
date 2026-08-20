import numpy as np
import pandas as pd
import pytest

from features.price_action import add_price_action


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for testing."""
    n = kwargs.pop("n", 20)
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    base = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    dt = [base + pd.Timedelta(minutes=i) for i in range(n)]
    defaults = dict(
        open=100.0, high=102.0, low=99.0, close=101.0, volume=1000
    )
    defaults.update(kwargs)
    data = {k: [v] * n for k, v in defaults.items()}
    data["datetime"] = dt
    return pd.DataFrame(data)


def test_price_return_1_zero_when_closes_equal():
    df = _make_df(close=100.0)
    out = add_price_action(df)
    # first row is NaN (pct_change); rest should be 0
    assert out["price_return_1"].iloc[1:].eq(0.0).all()


def test_candle_body_positive_for_bullish():
    df = _make_df(open=100.0, close=102.0)
    out = add_price_action(df)
    assert (out["candle_body"] > 0).all()


def test_candle_body_negative_for_bearish():
    df = _make_df(open=102.0, close=100.0)
    out = add_price_action(df)
    assert (out["candle_body"] < 0).all()


def test_upper_wick_zero_when_close_equals_high():
    # close == high → no upper shadow
    df = _make_df(open=99.0, high=101.0, low=98.0, close=101.0)
    out = add_price_action(df)
    assert np.allclose(out["upper_wick"], 0.0, atol=1e-9)


def test_close_position_one_when_close_equals_high():
    df = _make_df(open=99.0, high=101.0, low=98.0, close=101.0)
    out = add_price_action(df)
    assert np.allclose(out["close_position"], 1.0, atol=1e-6)


def test_close_position_zero_when_close_equals_low():
    df = _make_df(open=99.0, high=101.0, low=98.0, close=98.0)
    out = add_price_action(df)
    assert np.allclose(out["close_position"], 0.0, atol=1e-6)
