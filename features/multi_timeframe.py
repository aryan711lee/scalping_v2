"""
Multi-timeframe context features — adds 5-minute candle information to 3-minute model.

Critical alignment note: 5-minute candle timestamps are OPEN times (label='left').
A 5-min candle at 09:55 covers 09:55–10:00 and is NOT complete until 10:00.
We shift 5-min timestamps forward by 5 minutes before merge_asof so that
each 5-min candle is only available to 3-min candles AFTER it has closed.

Effect:
  5-min originally at 09:55 (closes at 10:00) → shifted to 10:00
  3-min at 10:00: merge_asof backward picks shifted 10:00 (orig 09:55) ✓
  3-min at 10:03: merge_asof backward picks shifted 10:00 (orig 09:55) ✓
  3-min at 10:06: merge_asof backward picks shifted 10:05 (orig 10:00) ✓
"""

import pandas as pd

_TF5_CANDLE_MINUTES = 5
_CONTEXT_COLS = ["ema9_above_ema21", "rsi_14", "vwap_above", "volume_ratio", "candle_body"]
_RENAME = {
    "ema9_above_ema21": "tf5_trend",
    "rsi_14": "tf5_rsi",
    "vwap_above": "tf5_vwap_above",
    "volume_ratio": "tf5_volume_ratio",
    "candle_body": "tf5_candle_body",
}


def add_5min_context(df_3min: pd.DataFrame, df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Add 5-minute context columns to a 3-minute feature DataFrame.

    df_3min: 3-min feature DataFrame with 'datetime' as a column (not index).
    df_5min: 5-min feature DataFrame with 'datetime' as a column (not index).

    Returns df_3min with 5 new columns: tf5_trend, tf5_rsi, tf5_vwap_above,
    tf5_volume_ratio, tf5_candle_body. Rows without a prior 5-min candle receive NaN.
    """
    # Select and rename context columns from 5-min feature set
    for col in _CONTEXT_COLS:
        if col not in df_5min.columns:
            raise ValueError(f"Expected column '{col}' in 5-min feature DataFrame")

    ctx = df_5min[["datetime"] + _CONTEXT_COLS].copy()
    ctx = ctx.rename(columns=_RENAME)

    # Shift 5-min timestamps forward by one candle duration so each candle is
    # only available after it has fully closed (prevents look-ahead).
    # Note: Timedelta addition may upcast datetime resolution (e.g. ms→us).
    # Cast back to the original dtype to keep merge_asof keys compatible.
    orig_dt_dtype = ctx["datetime"].dtype
    ctx["datetime"] = ctx["datetime"] + pd.Timedelta(minutes=_TF5_CANDLE_MINUTES)
    ctx["datetime"] = ctx["datetime"].astype(orig_dt_dtype)

    # merge_asof requires both inputs sorted by the key column and matching dtypes.
    left = df_3min.sort_values("datetime").reset_index(drop=True)
    right = ctx.sort_values("datetime").reset_index(drop=True)

    # Unify datetime dtype in case 3-min and 5-min parquets have different resolutions
    if left["datetime"].dtype != right["datetime"].dtype:
        right["datetime"] = right["datetime"].astype(left["datetime"].dtype)

    merged = pd.merge_asof(
        left,
        right,
        on="datetime",
        direction="backward",  # only use 5-min candles that have already closed
    )

    # Restore original row order from df_3min
    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged
