import pandas as pd


def add_price_action(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["price_return_1"] = close.pct_change(1)
    df["price_return_3"] = close.pct_change(3)
    df["price_return_5"] = close.pct_change(5)
    df["price_return_10"] = close.pct_change(10)

    df["candle_body"] = (df["close"] - df["open"]) / df["open"]
    df["candle_body_abs"] = df["candle_body"].abs()

    hl_range = df["high"] - df["low"] + 1e-9
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / hl_range
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / hl_range
    df["candle_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_position"] = (df["close"] - df["low"]) / hl_range

    return df
