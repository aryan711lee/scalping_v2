import pandas as pd


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trend_strength"] = df["ema9_distance"].abs()

    uptrend = (
        (df["ema_9"] > df["ema_21"])
        & (df["ema_21"] > df["ema_50"])
        & (df["close"] > df["ema_9"])
    )
    downtrend = (
        (df["ema_9"] < df["ema_21"])
        & (df["ema_21"] < df["ema_50"])
        & (df["close"] < df["ema_9"])
    )

    df["trend_regime"] = "ranging"
    df.loc[uptrend, "trend_regime"] = "uptrend"
    df.loc[downtrend, "trend_regime"] = "downtrend"

    df["trend_regime_enc"] = 0
    df.loc[uptrend, "trend_regime_enc"] = 1
    df.loc[downtrend, "trend_regime_enc"] = -1

    # ATR percentile within a backward-looking 50-candle rolling window.
    df["atr_percentile"] = df["atr_14"].rolling(50).rank(pct=True)

    high_vol = df["atr_percentile"] > 0.75
    low_vol = df["atr_percentile"] < 0.25

    df["vol_regime"] = "normal_vol"
    df.loc[high_vol, "vol_regime"] = "high_vol"
    df.loc[low_vol, "vol_regime"] = "low_vol"

    df["vol_regime_enc"] = 0
    df.loc[high_vol, "vol_regime_enc"] = 1
    df.loc[low_vol, "vol_regime_enc"] = -1

    return df
