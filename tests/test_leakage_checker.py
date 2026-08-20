"""Tests for validation/leakage_checker.py"""

import numpy as np
import pandas as pd
import pytest

from validation.leakage_checker import (
    check_fold_leakage,
    check_test_isolation,
    check_feature_leakage,
)


def _make_df(start: str, end: str, freq: str = "3min") -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq=freq, tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"label": rng.choice([-1, 0, 1], len(idx))}, index=idx)


# --- check_fold_leakage ---

def test_check_fold_leakage_clean_split_returns_true():
    train = _make_df("2023-01-02 09:15", "2024-06-28 15:30")
    val   = _make_df("2024-07-01 09:15", "2024-12-31 15:30")
    assert check_fold_leakage(train, val, fold_id=1) is True


def test_check_fold_leakage_raises_if_train_includes_val_dates():
    # Train goes into July — overlaps with val
    train = _make_df("2023-01-02 09:15", "2024-07-05 15:30")
    val   = _make_df("2024-07-01 09:15", "2024-12-31 15:30")
    with pytest.raises(ValueError, match="leakage"):
        check_fold_leakage(train, val, fold_id=1)


def test_check_fold_leakage_raises_on_shared_timestamps():
    # Same date range → overlapping timestamps
    train = _make_df("2023-01-02 09:15", "2024-07-31 15:30")
    val   = _make_df("2024-07-01 09:15", "2024-12-31 15:30")
    with pytest.raises(ValueError):
        check_fold_leakage(train, val, fold_id=1)


def test_check_fold_leakage_raises_if_val_within_train_period():
    train = _make_df("2023-01-02 09:15", "2025-06-30 15:30")
    val   = _make_df("2024-07-01 09:15", "2024-12-31 15:30")
    with pytest.raises(ValueError):
        check_fold_leakage(train, val, fold_id=1)


def test_check_fold_leakage_raises_on_empty_dataframe():
    empty = pd.DataFrame(index=pd.DatetimeIndex([], tz="Asia/Kolkata"))
    val   = _make_df("2024-07-01 09:15", "2024-12-31 15:30")
    with pytest.raises(ValueError):
        check_fold_leakage(empty, val, fold_id=1)


# --- check_test_isolation ---

def test_check_test_isolation_clean_returns_true():
    full_train = _make_df("2023-01-02 09:15", "2025-12-31 15:30")
    test       = _make_df("2026-01-03 09:15", "2026-08-19 15:30")
    assert check_test_isolation(full_train, test) is True


def test_check_test_isolation_raises_if_test_dates_in_train():
    full_train = _make_df("2023-01-02 09:15", "2026-03-31 15:30")
    test       = _make_df("2026-01-03 09:15", "2026-08-19 15:30")
    with pytest.raises(ValueError, match="[Ii]solation"):
        check_test_isolation(full_train, test)


def test_check_test_isolation_raises_on_date_boundary_overlap():
    # train runs right up to test_start date
    full_train = _make_df("2023-01-02 09:15", "2026-01-02 15:30")
    test       = _make_df("2026-01-03 09:15", "2026-08-19 15:30")
    # This should pass — train ends 2026-01-02, test starts 2026-01-03
    assert check_test_isolation(full_train, test) is True


# --- check_feature_leakage ---

def test_check_feature_leakage_flags_future_column():
    df = pd.DataFrame({"vwap_distance": [1.0], "future_return": [0.5], "rsi_14": [50.0]})
    flags = check_feature_leakage(df)
    assert "future_return" in flags


def test_check_feature_leakage_flags_forward_looking_names():
    df = pd.DataFrame({
        "next_close":    [1.0],
        "ahead_price":   [1.0],
        "target_price":  [1.0],
        "exit_price":    [1.0],
        "forward_vol":   [1.0],
    })
    flags = check_feature_leakage(df)
    assert len(flags) == 5


def test_check_feature_leakage_returns_empty_for_clean_features():
    df = pd.DataFrame({
        "vwap_distance":    [1.0],
        "ema9_distance":    [0.5],
        "rsi_14":           [50.0],
        "volume_ratio":     [1.2],
        "trend_regime_enc": [1.0],
    })
    flags = check_feature_leakage(df)
    assert flags == []
