import numpy as np
import pandas as pd
import pytz
import pytest

from features.volume import add_volume

IST = pytz.timezone("Asia/Kolkata")


def _make_df(closes, volumes=None) -> pd.DataFrame:
    n = len(closes)
    base = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    dt = [base + pd.Timedelta(minutes=i) for i in range(n)]
    volumes = volumes or [1000] * n
    return pd.DataFrame(
        dict(
            datetime=dt,
            open=closes,
            high=[c + 1 for c in closes],
            low=[c - 1 for c in closes],
            close=closes,
            volume=volumes,
        )
    )


def test_volume_ratio_one_when_volume_equals_ma20():
    """If every candle has the same volume, volume_ratio should be 1."""
    df = _make_df([100.0] * 50, volumes=[500] * 50)
    out = add_volume(df)
    valid = out["volume_ratio"].dropna()
    assert np.allclose(valid, 1.0, atol=1e-9)


def test_obv_increases_on_up_candle():
    """OBV should rise when close > prev_close."""
    closes = [100.0, 101.0, 102.0, 103.0]
    df = _make_df(closes, volumes=[1000] * 4)
    out = add_volume(df)
    obv = out["obv"].tolist()
    # first candle: sign(NaN) = 0 → OBV = 0
    # subsequent: close rises → +volume each step
    assert obv[0] == 0
    assert obv[1] > obv[0]
    assert obv[2] > obv[1]


def test_obv_decreases_on_down_candle():
    """OBV should fall when close < prev_close."""
    closes = [103.0, 102.0, 101.0, 100.0]
    df = _make_df(closes, volumes=[1000] * 4)
    out = add_volume(df)
    obv = out["obv"].tolist()
    assert obv[1] < obv[0]
    assert obv[2] < obv[1]


def test_obv_resets_each_day():
    """OBV must restart at 0 on a new trading day."""
    day1 = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    day2 = pd.Timestamp("2023-01-03 09:15:00", tz=IST)
    rows = [
        dict(datetime=day1, open=100, high=101, low=99, close=101, volume=1000),
        dict(datetime=day1 + pd.Timedelta(minutes=1), open=101, high=102, low=100, close=102, volume=1000),
        dict(datetime=day2, open=200, high=201, low=199, close=201, volume=500),
    ]
    df = pd.DataFrame(rows)
    out = add_volume(df)
    # Day 2 first candle: diff vs previous candle is NaN within day → OBV = 0
    day2_obv = out[out["datetime"].dt.date == pd.Timestamp("2023-01-03").date()]["obv"].iloc[0]
    assert day2_obv == 0, f"OBV did not reset: got {day2_obv}"


def test_volume_above_avg_flag():
    vols = [1000] * 20 + [5000]  # last one is way above average
    closes = [100.0] * 21
    df = _make_df(closes, volumes=vols)
    out = add_volume(df)
    assert out["volume_above_avg"].iloc[-1] == 1
    assert out["volume_above_avg"].iloc[0] == 0  # first row has NaN ma → 0
