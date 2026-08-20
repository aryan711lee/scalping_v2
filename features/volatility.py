import numpy as np
import pandas as pd


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    prev_close = close.shift(1)

    df["true_range"] = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # ATR uses Wilder's smoothing — same recursive formula as RSI.
    df["atr_14"] = df["true_range"].ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / close

    # Bollinger Bands (20-period, 2 std)
    rolling_20 = close.rolling(20)
    bb_mean = rolling_20.mean()
    bb_std = rolling_20.std()
    df["bb_upper"] = bb_mean + 2 * bb_std
    df["bb_lower"] = bb_mean - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mean
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Realized volatility (use price_return_1 if already computed, else recompute)
    ret1 = df["price_return_1"] if "price_return_1" in df.columns else close.pct_change(1)
    df["realized_vol_10"] = ret1.rolling(10).std() * np.sqrt(10)
    df["realized_vol_20"] = ret1.rolling(20).std() * np.sqrt(20)

    return df
