"""
Tests for labels/constructor.py

Each test crafts a synthetic DataFrame with controlled OHLC values
to verify label assignment logic precisely.
"""

import numpy as np
import pandas as pd
import pytest

from labels.constructor import construct_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict], date: str = "2024-01-15") -> pd.DataFrame:
    """
    Build a minimal DataFrame with datetime, open, high, low, close columns.
    Each dict in rows should have: minute (int HHmm), open, high, low, close.
    """
    records = []
    for r in rows:
        hh, mm = divmod(r["minute"], 100)
        dt = pd.Timestamp(f"{date} {hh:02d}:{mm:02d}:00", tz="Asia/Kolkata")
        records.append({
            "datetime": dt,
            "open":  float(r["open"]),
            "high":  float(r["high"]),
            "low":   float(r["low"]),
            "close": float(r.get("close", r["open"])),
        })
    return pd.DataFrame(records)


def _flat(n: int, price: float = 100.0, start_minute: int = 915) -> list[dict]:
    """Generate n flat candles starting at start_minute, 3-min apart."""
    rows = []
    hh = start_minute // 100
    mm = start_minute % 100
    for i in range(n):
        total_min = hh * 60 + mm + i * 3
        new_hh = total_min // 60
        new_mm = total_min % 60
        rows.append({
            "minute": new_hh * 100 + new_mm,
            "open": price,
            "high": price + 0.01,
            "low":  price - 0.01,
            "close": price,
        })
    return rows


# ---------------------------------------------------------------------------
# Test 1: Long label fires correctly
# ---------------------------------------------------------------------------

def test_long_label_fires():
    """Candles where high hits long target before low hits long stop → label = +1."""
    # Entry at candle 0 (signal), fill at candle 1 open = 100
    # target = 100 * 1.004 = 100.4; stop = 100 * 0.998 = 99.8
    # Use horizon=2 with 5 candles so candle 0 is NOT in last-2 zone.
    rows = [
        {"minute": 915,  "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # signal (idx 0)
        {"minute": 918,  "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # entry
        {"minute": 921,  "open": 100, "high": 100.5, "low": 99.9, "close": 100},  # target hit
        {"minute": 924,  "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # padding
        {"minute": 927,  "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # padding
    ]
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=2)
    assert labels.iloc[0] == 1, f"Expected +1, got {labels.iloc[0]}"


# ---------------------------------------------------------------------------
# Test 2: Short label fires correctly
# ---------------------------------------------------------------------------

def test_short_label_fires():
    """Candles where low hits short target before high hits short stop → label = -1."""
    # Entry open = 100; short_target = 100 * 0.996 = 99.6; short_stop = 100 * 1.002 = 100.2
    # Use horizon=2 with 5 candles so candle 0 is NOT in last-2 zone.
    rows = [
        {"minute": 915, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # signal (idx 0)
        {"minute": 918, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # entry
        {"minute": 921, "open": 100, "high": 100.1, "low": 99.5, "close": 100},  # short target
        {"minute": 924, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # padding
        {"minute": 927, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # padding
    ]
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=2)
    assert labels.iloc[0] == -1, f"Expected -1, got {labels.iloc[0]}"


# ---------------------------------------------------------------------------
# Test 3: No-trade when neither hits within horizon
# ---------------------------------------------------------------------------

def test_no_trade_when_neither_hits():
    """Flat candles that never reach target or stop → label = 0."""
    rows = _flat(25, price=100.0)
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=10)
    # candle 0 should be 0 (no resolution within 10 flat candles at ±0.01 range)
    assert labels.iloc[0] == 0, f"Expected 0, got {labels.iloc[0]}"


# ---------------------------------------------------------------------------
# Test 4: No-trade when both long and short resolve in same first candle
# ---------------------------------------------------------------------------

def test_conflicted_label_is_no_trade():
    """If both long and short targets are hit → label = 0 (conflicted)."""
    # entry_price = 100; long_target = 100.4; short_target = 99.6
    # Next candle has high >= 100.4 AND low <= 99.6 → conflicted
    rows = [
        {"minute": 915, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # signal
        {"minute": 918, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # entry open=100
        {"minute": 921, "open": 100, "high": 100.5, "low": 99.5, "close": 100},  # both hit
        {"minute": 924, "open": 100, "high": 100.1, "low": 99.9, "close": 100},
    ]
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=20)
    assert labels.iloc[0] == 0, f"Expected 0 (conflicted), got {labels.iloc[0]}"


# ---------------------------------------------------------------------------
# Test 5: Stop wins when target and stop both hit in same candle
# ---------------------------------------------------------------------------

def test_stop_wins_over_target_in_same_candle():
    """
    When a single forward candle's HIGH crosses long_target AND LOW crosses long_stop,
    the stop is assumed to hit first (conservative) → long_success = False.
    If only long would conflict and short doesn't resolve → label = 0 or -1.
    """
    # entry open = 100; long_target = 100.4; long_stop = 99.8
    # short_target = 99.6; short_stop = 100.2
    # Forward candle: high = 100.5 (long target), low = 99.7 (long stop)
    # Long: same candle — stop wins → long_success = False
    # Short: low = 99.7 <= 99.6? No. short never resolves within 1 horizon candle.
    rows = [
        {"minute": 915, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # signal
        {"minute": 918, "open": 100, "high": 100.1, "low": 99.9, "close": 100},  # entry
        {"minute": 921, "open": 100, "high": 100.5, "low": 99.7, "close": 100},  # both hit → stop wins
    ]
    df = _make_df(rows)
    # horizon=1 so only candle at 921 is looked at
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=1)
    # long stop wins → no long success; short not resolved → label = 0
    assert labels.iloc[0] == 0, f"Expected 0 (stop wins for long), got {labels.iloc[0]}"


# ---------------------------------------------------------------------------
# Test 6: No overnight look-ahead
# ---------------------------------------------------------------------------

def test_no_overnight_lookahead():
    """Entry candle near end of session; forward candles on next day are not scanned."""
    # Day 1: last 2 candles at 15:24 and 15:27
    # Day 2: candles where target would be hit
    rows_day1 = [
        {"minute": 1527, "open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"minute": 1530, "open": 100, "high": 100.1, "low": 99.9, "close": 100},
    ]
    rows_day2 = [
        {"minute": 915, "open": 100, "high": 100.5, "low": 99.5, "close": 100},
        {"minute": 918, "open": 100, "high": 100.5, "low": 99.5, "close": 100},
        {"minute": 921, "open": 100, "high": 100.5, "low": 99.5, "close": 100},
    ]
    recs = []
    for r in rows_day1:
        hh, mm = divmod(r["minute"], 100)
        dt = pd.Timestamp(f"2024-01-15 {hh:02d}:{mm:02d}:00", tz="Asia/Kolkata")
        recs.append({"datetime": dt, "open": r["open"], "high": r["high"],
                     "low": r["low"], "close": r["close"]})
    for r in rows_day2:
        hh, mm = divmod(r["minute"], 100)
        dt = pd.Timestamp(f"2024-01-16 {hh:02d}:{mm:02d}:00", tz="Asia/Kolkata")
        recs.append({"datetime": dt, "open": r["open"], "high": r["high"],
                     "low": r["low"], "close": r["close"]})
    df = pd.DataFrame(recs)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=20)
    # Day 1 candle at index 0 (15:27 signal): entry would be 15:30 open=100
    # The only forward same-day candle is 15:30 which has range ±0.1 — neither resolves
    # → label = 0 (day boundary stops the scan)
    assert labels.iloc[0] == 0, f"Expected 0 (no overnight), got {labels.iloc[0]}"


# ---------------------------------------------------------------------------
# Test 7: Entry price is next candle open
# ---------------------------------------------------------------------------

def test_entry_price_is_next_candle_open():
    """
    Targets computed from next candle's open, not current close.
    If close=100 but next open=105, target must be based on 105.
    """
    # Signal candle close=100; next candle open=105; long_target = 105*1.004 = 105.42
    # Forward candle: high=105.3 (< 105.42) → should NOT hit target
    rows = [
        {"minute": 915, "open": 100, "high": 100.5, "low": 99.5, "close": 100},  # signal
        {"minute": 918, "open": 105, "high": 105.3, "low": 104.9, "close": 105},  # entry open=105
        {"minute": 921, "open": 105, "high": 105.3, "low": 104.9, "close": 105},  # high<target
        {"minute": 924, "open": 105, "high": 105.3, "low": 104.9, "close": 105},
    ] + _flat(20, price=105, start_minute=927)
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=20)
    assert labels.iloc[0] == 0, (
        f"Expected 0 (target based on next open=105, not close=100), got {labels.iloc[0]}"
    )


# ---------------------------------------------------------------------------
# Test 8: Last horizon candles of day are labeled 0
# ---------------------------------------------------------------------------

def test_last_horizon_candles_labeled_zero():
    """Final `horizon` candles of each trading day must always be labeled 0."""
    horizon = 5
    rows = _flat(30, price=100.0)  # 30 flat candles
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=horizon)
    assert all(labels.iloc[-horizon:] == 0), (
        f"Expected last {horizon} candles to be 0, got: {labels.iloc[-horizon:].tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 9: Output length matches input
# ---------------------------------------------------------------------------

def test_output_length_matches_input():
    rows = _flat(50)
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=10)
    assert len(labels) == len(df), f"Expected {len(df)}, got {len(labels)}"


# ---------------------------------------------------------------------------
# Test 10: No NaN in output
# ---------------------------------------------------------------------------

def test_no_nan_in_output():
    rows = _flat(50)
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=10)
    assert not labels.isna().any(), "Labels contain NaN values"


# ---------------------------------------------------------------------------
# Test 11: Only valid label values
# ---------------------------------------------------------------------------

def test_only_valid_label_values():
    rows = _flat(50)
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=10)
    assert set(labels.unique()).issubset({-1, 0, 1}), (
        f"Unexpected label values: {set(labels.unique())}"
    )


# ---------------------------------------------------------------------------
# Test 12: dtype is int8
# ---------------------------------------------------------------------------

def test_label_dtype_is_int8():
    rows = _flat(20)
    df = _make_df(rows)
    labels = construct_labels(df, target_pct=0.004, stop_pct=0.002, horizon=5)
    assert labels.dtype == np.int8, f"Expected int8, got {labels.dtype}"
