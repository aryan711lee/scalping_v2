import pandas as pd


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    df["ema9_distance"] = (close - df["ema_9"]) / close
    df["ema21_distance"] = (close - df["ema_21"]) / close
    df["ema50_distance"] = (close - df["ema_50"]) / close

    df["ema9_above_ema21"] = (df["ema_9"] > df["ema_21"]).astype(int)
    df["ema21_above_ema50"] = (df["ema_21"] > df["ema_50"]).astype(int)

    # VWAP — resets every trading day at 09:15.
    # groupby(date).cumsum() ensures the cumulative sums restart at zero each day.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["_tp_vol"] = typical_price * df["volume"]
    date_key = df["datetime"].dt.date
    df["vwap"] = df.groupby(date_key)["_tp_vol"].cumsum() / df.groupby(date_key)["volume"].cumsum()
    df = df.drop(columns=["_tp_vol"])

    df["vwap_distance"] = (close - df["vwap"]) / close
    df["vwap_above"] = (close > df["vwap"]).astype(int)

    return df
