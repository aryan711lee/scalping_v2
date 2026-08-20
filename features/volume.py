import numpy as np
import pandas as pd


def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    df["volume_ma_20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / df["volume_ma_20"]
    df["volume_return_1"] = volume.diff() / (volume.shift(1) + 1)
    df["volume_above_avg"] = (volume > df["volume_ma_20"]).astype(int)

    # OBV resets every trading day so intraday accumulation doesn't carry over.
    # diff() is computed within each day so the first candle of a new day never
    # compares against the previous day's close — it contributes 0 to that day's OBV.
    date_key = df["datetime"].dt.date
    daily_diff = df.groupby(date_key)["close"].diff()
    sign = np.sign(daily_diff).fillna(0)
    df["_sv"] = sign * volume
    df["obv"] = df.groupby(date_key)["_sv"].cumsum()
    df = df.drop(columns=["_sv"])

    df["obv_slope"] = df["obv"] - df["obv"].shift(5)

    return df
