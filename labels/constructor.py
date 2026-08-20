"""
Label construction logic for three-class labels (+1 long, -1 short, 0 no-trade).

Critical rules (enforced throughout):
  1. No overnight look-ahead — stop at trading day boundary.
  2. Entry price = next candle's open (matches backtester fill model).
  3. Candle-level resolution — if high and low both cross in one candle, stop wins.
  4. Last `horizon` candles of each day are labeled 0.
  5. Vectorized-friendly implementation using numpy.
"""

import numpy as np
import pandas as pd


def construct_labels(
    df: pd.DataFrame,
    target_pct: float,
    stop_pct: float,
    horizon: int,
) -> pd.Series:
    """
    For each candle at index i, look forward up to `horizon` candles.
    Determine whether a long, short, or no-trade label applies.

    Uses the NEXT candle's open as the simulated entry price.
    (Consistent with backtester: signal at T, fill at T+1 open)

    Returns pd.Series of int8 with values: +1, -1, 0
    Index matches input df index.
    Last `horizon` rows of each trading day are labeled 0 (no trade).
    """
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)

    # Extract date from datetime for boundary checks
    dt = pd.to_datetime(df["datetime"])
    dates = dt.dt.date
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)

    date_arr = dates.to_numpy()  # numpy array of date objects for fast indexing

    for i in range(n - 1):
        # Entry is at candle i+1 open — need candle i+1 to exist
        entry_idx = i + 1
        if entry_idx >= n:
            break

        # Entry price: next candle's open
        entry_price = opens[entry_idx]
        current_date = date_arr[i]

        long_target  = entry_price * (1.0 + target_pct)
        long_stop    = entry_price * (1.0 - stop_pct)
        short_target = entry_price * (1.0 - target_pct)
        short_stop   = entry_price * (1.0 + stop_pct)

        long_success  = False
        short_success = False
        long_resolved  = False
        short_resolved = False

        # Scan forward candles i+1 .. i+horizon (same trading day only)
        end_idx = min(n, i + 1 + horizon)
        for j in range(i + 1, end_idx):
            if date_arr[j] != current_date:
                # Day boundary — stop scanning entirely
                break

            h = highs[j]
            lo = lows[j]

            # Long resolution
            if not long_resolved:
                tp_hit = h >= long_target
                sl_hit = lo <= long_stop

                if tp_hit and sl_hit:
                    # Both in same candle — stop wins (conservative)
                    long_resolved = True
                    long_success = False
                elif sl_hit:
                    long_resolved = True
                    long_success = False
                elif tp_hit:
                    long_resolved = True
                    long_success = True

            # Short resolution
            if not short_resolved:
                tp_hit = lo <= short_target
                sl_hit = h >= short_stop

                if tp_hit and sl_hit:
                    # Both in same candle — stop wins (conservative)
                    short_resolved = True
                    short_success = False
                elif sl_hit:
                    short_resolved = True
                    short_success = False
                elif tp_hit:
                    short_resolved = True
                    short_success = True

            if long_resolved and short_resolved:
                break

        # Final label assignment
        if long_success and short_success:
            labels[i] = 0   # conflicted — no trade
        elif long_success:
            labels[i] = 1
        elif short_success:
            labels[i] = -1
        else:
            labels[i] = 0

    # Rule 4: last `horizon` candles of each trading day are labeled 0.
    # We must do this per-day so that late-day candles aren't mislabeled.
    for date_val, grp in df.groupby(dates):
        idx = grp.index.to_numpy()
        if len(idx) <= horizon:
            # Entire day has fewer candles than horizon — label all 0
            labels[idx] = 0
        else:
            labels[idx[-horizon:]] = 0

    return pd.Series(labels, index=df.index, dtype=np.int8)
