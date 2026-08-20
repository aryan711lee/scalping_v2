import pandas as pd


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's exponential smoothing (alpha=1/period, not simple rolling mean)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # ewm with alpha=1/period and adjust=False implements the recursive formula:
    #   avg = (prev_avg * (period-1) + current) / period  — identical to Wilder's smoothing.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    # When avg_loss == 0 all gains → RSI should be 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    df["rsi_14"] = wilder_rsi(close, 14)
    df["rsi_slope"] = df["rsi_14"] - df["rsi_14"].shift(3)
    df["roc_5"] = close.pct_change(5) * 100
    df["roc_10"] = close.pct_change(10) * 100

    return df
