import numpy as np
import pandas as pd
import pytz
import pytest

from features.market_context import add_nifty_context, add_time_features

IST = pytz.timezone("Asia/Kolkata")


def _row(dt_str: str, close: float = 100.0) -> dict:
    dt = pd.Timestamp(dt_str, tz=IST)
    return dict(datetime=dt, open=close, high=close + 1, low=close - 1, close=close, volume=1000)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_session_minute_zero_at_market_open():
    df = _make_df([_row("2023-01-02 09:15:00")])
    out = add_time_features(df)
    assert out["session_minute"].iloc[0] == 0


def test_session_minute_374_at_last_candle():
    df = _make_df([_row("2023-01-02 15:29:00")])
    out = add_time_features(df)
    assert out["session_minute"].iloc[0] == 374


def test_is_opening_30min_flag():
    rows = [
        _row("2023-01-02 09:15:00"),  # session_minute=0  → opening
        _row("2023-01-02 09:44:00"),  # session_minute=29 → still opening
        _row("2023-01-02 09:45:00"),  # session_minute=30 → NOT opening
        _row("2023-01-02 10:00:00"),  # session_minute=45 → NOT opening
    ]
    out = add_time_features(_make_df(rows))
    flags = out["is_opening_30min"].tolist()
    assert flags == [1, 1, 0, 0]


def test_is_closing_30min_flag():
    rows = [
        _row("2023-01-02 15:00:00"),  # session_minute=345 → NOT closing
        _row("2023-01-02 15:01:00"),  # session_minute=346 → closing
        _row("2023-01-02 15:29:00"),  # session_minute=374 → closing
    ]
    out = add_time_features(_make_df(rows))
    flags = out["is_closing_30min"].tolist()
    assert flags == [0, 1, 1]


def test_nifty_join_forward_fills_missing_candle():
    """If NIFTY is missing one candle, the stock row should get the previous NIFTY value."""
    stock_rows = [
        _row("2023-01-02 09:15:00", close=100.0),
        _row("2023-01-02 09:16:00", close=101.0),  # NIFTY missing at this time
        _row("2023-01-02 09:17:00", close=102.0),
    ]
    stock_df = _make_df(stock_rows)
    # Add dummy feature columns expected by add_time_features
    stock_df = add_time_features(stock_df)

    nifty_rows = [
        dict(datetime=pd.Timestamp("2023-01-02 09:15:00", tz=IST),
             nifty_return_1=0.001, nifty_return_3=0.002, nifty_return_5=0.003,
             nifty_rsi=55.0, nifty_trend=1),
        # 09:16 missing intentionally
        dict(datetime=pd.Timestamp("2023-01-02 09:17:00", tz=IST),
             nifty_return_1=0.002, nifty_return_3=0.004, nifty_return_5=0.006,
             nifty_rsi=58.0, nifty_trend=1),
    ]
    nifty_feat = pd.DataFrame(nifty_rows)

    out = add_nifty_context(stock_df, nifty_feat)

    # Row 1 (09:16) should be forward-filled from row 0 (09:15)
    assert out["nifty_rsi"].iloc[1] == pytest.approx(55.0)
    assert out["nifty_trend"].iloc[1] == 1


def test_day_sin_cos_range():
    rows = [_row(f"2023-01-0{d} 09:15:00") for d in range(2, 7)]  # Mon–Fri
    out = add_time_features(_make_df(rows))
    assert (out["day_sin"].abs() <= 1.0).all()
    assert (out["day_cos"].abs() <= 1.0).all()
