import numpy as np
import pandas as pd
import pytz
import pytest

from features.trend import add_trend

IST = pytz.timezone("Asia/Kolkata")


def _make_df(closes, opens=None, highs=None, lows=None, volumes=None, start="2023-01-02 09:15") -> pd.DataFrame:
    n = len(closes)
    base = pd.Timestamp(start, tz=IST)
    dt = [base + pd.Timedelta(minutes=i) for i in range(n)]
    closes = list(closes)
    opens = opens or closes
    highs = highs or [c + 1 for c in closes]
    lows = lows or [c - 1 for c in closes]
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        dict(datetime=dt, open=opens, high=highs, low=lows, close=closes, volume=volumes)
    )


def test_ema_9_approaches_constant_close():
    """After many identical candles, EMA(9) should converge to that price."""
    price = 100.0
    df = _make_df([price] * 200)
    out = add_trend(df)
    assert abs(out["ema_9"].iloc[-1] - price) < 1e-6


def test_vwap_equals_typical_price_of_first_candle():
    """First candle of the session: VWAP = (high+low+close)/3 = typical_price."""
    df = _make_df([100.0], highs=[102.0], lows=[98.0])
    out = add_trend(df)
    expected_tp = (102.0 + 98.0 + 100.0) / 3
    assert abs(out["vwap"].iloc[0] - expected_tp) < 1e-9


def test_vwap_resets_on_new_trading_day():
    """VWAP must restart at each day's first candle, not continue from yesterday."""
    # Build two days of data
    day1_start = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    day2_start = pd.Timestamp("2023-01-03 09:15:00", tz=IST)

    rows = []
    for i in range(375):
        rows.append(dict(
            datetime=day1_start + pd.Timedelta(minutes=i),
            open=100.0, high=102.0, low=98.0, close=100.0 + i * 0.01, volume=1000,
        ))
    for i in range(5):
        rows.append(dict(
            datetime=day2_start + pd.Timedelta(minutes=i),
            open=200.0, high=202.0, low=198.0, close=200.0, volume=500,
        ))

    df = pd.DataFrame(rows)
    out = add_trend(df)

    # First candle of day 2: VWAP should equal its own typical price
    day2_first = out[out["datetime"].dt.date == pd.Timestamp("2023-01-03").date()].iloc[0]
    expected_tp = (202.0 + 198.0 + 200.0) / 3
    assert abs(day2_first["vwap"] - expected_tp) < 1e-6, (
        f"VWAP did not reset: got {day2_first['vwap']:.4f}, expected {expected_tp:.4f}"
    )


def test_ema9_above_ema21_flag():
    """Flag is 1 when EMA(9) > EMA(21), 0 otherwise."""
    # Use a constant series so EMAs converge to same value → flag = 0
    df = _make_df([100.0] * 100)
    out = add_trend(df)
    # After convergence both EMAs equal 100; ema9 > ema21 is False → flag should be 0
    assert out["ema9_above_ema21"].iloc[-1] == 0

    # Rising series: short EMA reacts faster → ema9 > ema21
    closes = list(range(1, 101))
    df2 = _make_df(closes)
    out2 = add_trend(df2)
    assert out2["ema9_above_ema21"].iloc[-1] == 1
