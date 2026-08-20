import numpy as np
import pandas as pd

from features.momentum import wilder_rsi


def compute_nifty_features(nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive NIFTY context columns from cleaned NIFTY candles.
    Returns a DataFrame with datetime + 5 nifty feature columns.
    """
    df = nifty_df.copy()
    close = df["close"]

    df["nifty_return_1"] = close.pct_change(1)
    df["nifty_return_3"] = close.pct_change(3)
    df["nifty_return_5"] = close.pct_change(5)
    df["nifty_rsi"] = wilder_rsi(close, 14)

    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    df["nifty_trend"] = 0
    df.loc[ema9 > ema21, "nifty_trend"] = 1
    df.loc[ema9 < ema21, "nifty_trend"] = -1

    cols = ["datetime", "nifty_return_1", "nifty_return_3", "nifty_return_5", "nifty_rsi", "nifty_trend"]
    return df[cols].copy()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]

    # minutes elapsed since 09:15 on the same calendar day
    market_open = dt.dt.normalize() + pd.Timedelta(hours=9, minutes=15)
    session_minute = (dt - market_open).dt.total_seconds() / 60
    df["session_minute"] = session_minute.clip(0, 374).astype(int)

    df["time_sin"] = np.sin(2 * np.pi * df["session_minute"] / 375)
    df["time_cos"] = np.cos(2 * np.pi * df["session_minute"] / 375)

    # first 30 min: 09:15–09:44  → session_minute 0–29
    df["is_opening_30min"] = (df["session_minute"] < 30).astype(int)
    # last 30 min: 15:01–15:30  → session_minute 346–374
    df["is_closing_30min"] = (df["session_minute"] >= 346).astype(int)

    dow = dt.dt.dayofweek  # 0=Monday, 4=Friday
    df["day_sin"] = np.sin(2 * np.pi * dow / 5)
    df["day_cos"] = np.cos(2 * np.pi * dow / 5)

    return df


def add_nifty_context(df: pd.DataFrame, nifty_features: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join pre-computed NIFTY features onto stock DataFrame on datetime.
    Forward-fills any stock candles where NIFTY data is absent.
    """
    df = df.copy()
    nifty_cols = ["nifty_return_1", "nifty_return_3", "nifty_return_5", "nifty_rsi", "nifty_trend"]

    nifty_indexed = nifty_features.set_index("datetime")[nifty_cols]

    # Normalise timezone representation before joining so UTC+05:30 and Asia/Kolkata match.
    stock_index = df["datetime"]
    if hasattr(stock_index.dtype, "tz") and stock_index.dtype.tz is not None:
        nifty_indexed.index = nifty_indexed.index.tz_convert(stock_index.dt.tz)

    df = df.set_index("datetime").join(nifty_indexed, how="left")
    df[nifty_cols] = df[nifty_cols].ffill()
    df = df.reset_index()

    return df
