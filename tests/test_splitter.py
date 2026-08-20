"""Tests for validation/splitter.py"""

import pytest
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS, TEST_PERIOD


@pytest.fixture
def splitter():
    return WalkForwardSplitter(FOLD_DEFINITIONS)


def test_fold1_train_end_before_validate_start(splitter):
    f = splitter.get_fold(1)
    assert f["train_end"] < f["validate_start"]


def test_fold2_train_end_before_validate_start(splitter):
    f = splitter.get_fold(2)
    assert f["train_end"] < f["validate_start"]


def test_fold3_train_end_before_validate_start(splitter):
    f = splitter.get_fold(3)
    assert f["train_end"] < f["validate_start"]


def test_fold2_train_includes_fold1_train(splitter):
    f1 = splitter.get_fold(1)
    f2 = splitter.get_fold(2)
    # Expanding window: fold 2 train_start is same, train_end is later
    assert f2["train_start"] == f1["train_start"]
    assert f2["train_end"] > f1["train_end"]


def test_fold3_train_includes_fold1_and_fold2(splitter):
    f1 = splitter.get_fold(1)
    f2 = splitter.get_fold(2)
    f3 = splitter.get_fold(3)
    assert f3["train_start"] == f1["train_start"]
    assert f3["train_end"] > f2["train_end"]


def test_validate_periods_do_not_overlap(splitter):
    folds = splitter.get_all_folds()
    for i in range(len(folds) - 1):
        assert folds[i]["validate_end"] < folds[i + 1]["validate_start"]


def test_test_period_does_not_overlap_any_fold(splitter):
    tp = splitter.get_test_period()
    for f in splitter.get_all_folds():
        assert f["validate_end"] < tp["test_start"]


def test_get_all_folds_returns_exactly_three(splitter):
    assert len(splitter.get_all_folds()) == 3


def test_summary_contains_all_fold_ids(splitter):
    s = splitter.summary()
    assert "Fold 1" in s
    assert "Fold 2" in s
    assert "Fold 3" in s
    assert "Test" in s or "HOLDOUT" in s


def test_get_fold_invalid_id_raises(splitter):
    with pytest.raises(KeyError):
        splitter.get_fold(99)


def test_get_test_period_returns_correct_dates(splitter):
    tp = splitter.get_test_period()
    assert tp["test_start"] == "2026-01-01"
    assert tp["test_end"] == "2026-08-19"
