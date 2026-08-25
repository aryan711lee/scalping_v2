import numpy as np
import pandas as pd


def add_momentum_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three conviction-based features that measure the *quality* of a move,
    not just its direction. Designed to address Phase 6 finding that
    time-of-day features dominated over price/momentum features.
    """
    df = df.copy()

    # Feature 1: candle_consistency_3
    # +1 if last 3 candles are all bullish (close > open)
    # -1 if last 3 candles are all bearish (close < open)
    #  0 otherwise (mixed direction)
    is_bull = (df["close"] > df["open"]).astype(int)
    is_bear = (df["close"] < df["open"]).astype(int)

    bull_3 = (is_bull == 1) & (is_bull.shift(1) == 1) & (is_bull.shift(2) == 1)
    bear_3 = (is_bear == 1) & (is_bear.shift(1) == 1) & (is_bear.shift(2) == 1)

    df["candle_consistency_3"] = np.where(bull_3, 1, np.where(bear_3, -1, 0)).astype(float)

    # Feature 2: volume_price_trend
    # sign(price_return_1) × volume_ratio — positive when price moving up with above-avg volume
    # Clipped to [-5, +5] to prevent extreme values from distorting the model
    price_sign = np.sign(df["price_return_1"].fillna(0))
    df["volume_price_trend"] = (price_sign * df["volume_ratio"]).clip(-5, 5)

    # Feature 3: ema9_slope
    # (ema_9 - ema_9.shift(3)) / ema_9.shift(3) — rate of change of the short-term trend
    # Distinguishes a fresh crossover (high slope) from a stale one (slope near zero).
    # Stale crossovers are a major source of false signals in EMA-based systems.
    ema9_lag = df["ema_9"].shift(3)
    df["ema9_slope"] = (df["ema_9"] - ema9_lag) / ema9_lag.replace(0, np.nan)

    return df
