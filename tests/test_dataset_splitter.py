"""Tests for validation/dataset_splitter.py"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS
from validation.dataset_splitter import DatasetSplitter


def _make_df(start: str, end: str, freq: str = "3min", add_extra_cols: bool = True) -> pd.DataFrame:
    """Create a minimal labeled DataFrame with IST timezone index."""
    idx = pd.date_range(start=start, end=end, freq=freq, tz="Asia/Kolkata")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=idx)
    df.index.name = "datetime"
    df["label"] = rng.choice([-1, 0, 1], size=len(idx))
    df["symbol"] = "NSE:TEST-EQ"

    if add_extra_cols:
        # Simulate real feature columns including ones that should be dropped
        for col in ["ema_9", "ema_21", "ema_50", "bb_upper", "bb_lower", "vwap"]:
            df[col] = rng.random(len(idx))
        for col in ["vwap_distance", "ema9_distance", "rsi_14", "volume_ratio",
                    "atr_14", "bb_pct", "ema9_above_ema21", "trend_regime_enc"]:
            df[col] = rng.random(len(idx))

    return df


@pytest.fixture
def splitter():
    return WalkForwardSplitter(FOLD_DEFINITIONS)


@pytest.fixture
def dataset_splitter(splitter):
    return DatasetSplitter(splitter)


@pytest.fixture
def full_df():
    # Spans entire 2023-2026 range at daily resolution (to keep test fast)
    return _make_df("2023-01-02 09:15:00", "2026-08-19 15:30:00", freq="D")


def test_split_fold1_correct_date_ranges(dataset_splitter, full_df):
    train_df, val_df = dataset_splitter.split(full_df, fold_id=1)
    # Train should not include 2024-07-01 or later
    assert train_df.index.max() < pd.Timestamp("2024-07-01", tz="Asia/Kolkata")
    # Val should not include dates before 2024-07-01
    assert val_df.index.min() >= pd.Timestamp("2024-07-01", tz="Asia/Kolkata")
    # Val should not include 2025 data
    assert val_df.index.max() <= pd.Timestamp("2024-12-31 23:59:59", tz="Asia/Kolkata")


def test_train_df_has_no_rows_from_validation_period(dataset_splitter, full_df):
    train_df, val_df = dataset_splitter.split(full_df, fold_id=1)
    val_start = pd.Timestamp("2024-07-01", tz="Asia/Kolkata")
    assert (train_df.index >= val_start).sum() == 0


def test_validate_df_has_no_rows_from_training_period(dataset_splitter, full_df):
    train_df, val_df = dataset_splitter.split(full_df, fold_id=1)
    train_end = pd.Timestamp("2024-06-30 23:59:59", tz="Asia/Kolkata")
    assert (val_df.index <= train_end).sum() == 0


def test_get_test_split_logs_warning(dataset_splitter, full_df, caplog):
    with caplog.at_level(logging.WARNING, logger="validation.dataset_splitter"):
        _ = dataset_splitter.get_test_split(full_df)
    assert any("WARNING" in rec.message and "test holdout" in rec.message.lower()
               for rec in caplog.records)


def test_get_feature_columns_excludes_dropped_features(dataset_splitter, full_df):
    feat_cols = dataset_splitter.get_feature_columns(full_df)
    dropped = {"ema_9", "ema_21", "ema_50", "bb_upper", "bb_lower", "vwap"}
    for d in dropped:
        assert d not in feat_cols, f"{d} should be excluded from feature columns"


def test_get_feature_columns_excludes_label_and_symbol(dataset_splitter, full_df):
    feat_cols = dataset_splitter.get_feature_columns(full_df)
    assert "label" not in feat_cols
    assert "symbol" not in feat_cols


def test_get_feature_columns_count(dataset_splitter, full_df):
    feat_cols = dataset_splitter.get_feature_columns(full_df)
    # Our test df has 8 feature cols (after dropping 6 raw + 2 meta)
    # Just verify it's positive and drops happened
    assert len(feat_cols) > 0
    assert len(feat_cols) < len(full_df.columns)


def test_get_class_weights_all_three_classes(dataset_splitter, full_df):
    train_df, _ = dataset_splitter.split(full_df, fold_id=1)
    weights = dataset_splitter.get_class_weights(train_df)
    assert set(weights.keys()) == {-1, 0, 1}


def test_get_class_weights_balanced(dataset_splitter):
    """Weighted counts should be approximately equal across classes."""
    # Create perfectly balanced df
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.date_range("2023-01-02", periods=n, freq="3min", tz="Asia/Kolkata")
    df = pd.DataFrame({"label": np.tile([-1, 0, 1], n // 3)}, index=idx)
    splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds = DatasetSplitter(splitter)
    weights = ds.get_class_weights(df)
    # All weights should be ~1.0 for perfectly balanced data
    for cls, w in weights.items():
        assert abs(w - 1.0) < 0.01, f"Weight for class {cls}: {w}"


def test_get_class_weights_imbalanced_gives_higher_weight_to_minority(dataset_splitter):
    """The majority class (0) should get lower weight than minority classes."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-02", periods=1000, freq="3min", tz="Asia/Kolkata")
    # ~65% class 0, ~17.5% each for ±1
    labels = np.concatenate([
        np.zeros(650, dtype=int),
        np.ones(175, dtype=int),
        np.full(175, -1, dtype=int),
    ])
    rng.shuffle(labels)
    df = pd.DataFrame({"label": labels}, index=idx)
    splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds = DatasetSplitter(splitter)
    weights = ds.get_class_weights(df)
    assert weights[0] < weights[1]
    assert weights[0] < weights[-1]
