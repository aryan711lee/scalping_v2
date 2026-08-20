"""Tests for validation/evaluator.py"""

import numpy as np
import pytest

from validation.evaluator import evaluate_predictions, aggregate_fold_results


def _make_result(**kwargs):
    """Helper to create an evaluate_predictions result with overrides."""
    base_true = np.array([1, 1, -1, -1, 0, 0, 1, -1, 0, 1])
    base_pred = np.array([1, 0, -1, 0, 0, 0, 1, -1, 0, 1])
    true = kwargs.pop("y_true", base_true)
    pred = kwargs.pop("y_pred", base_pred)
    return evaluate_predictions(
        y_true=true, y_pred=pred,
        fold_id=kwargs.pop("fold_id", 1),
        model_name=kwargs.pop("model_name", "test"),
        variant=kwargs.pop("variant", "L1"),
    )


def test_perfect_predictions_precision_one():
    y = np.array([1, -1, 0, 1, -1])
    result = evaluate_predictions(y, y, fold_id=1, model_name="m", variant="L1")
    assert result["precision_long"]  == pytest.approx(1.0)
    assert result["precision_short"] == pytest.approx(1.0)
    assert result["combined_precision"] == pytest.approx(1.0)


def test_all_zero_predictions_signal_rate_zero():
    y_true = np.array([1, -1, 0, 1, -1, 0])
    y_pred = np.zeros(6, dtype=int)
    result = evaluate_predictions(y_true, y_pred, fold_id=1, model_name="m", variant="L1")
    assert result["signal_rate"] == pytest.approx(0.0)


def test_combined_precision_none_when_no_signals():
    y_true = np.array([1, -1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    result = evaluate_predictions(y_true, y_pred, fold_id=1, model_name="m", variant="L1")
    assert result["combined_precision"] is None


def test_signal_rate_only_class_zero_model():
    y_true = np.array([1, -1, 0, 1, -1, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0])
    result = evaluate_predictions(y_true, y_pred, fold_id=1, model_name="m", variant="L1")
    assert result["signal_rate"] == pytest.approx(0.0)


def test_confusion_matrix_shape_always_3x3():
    y_true = np.array([1, 0, -1])
    y_pred = np.array([0, 0, 0])
    result = evaluate_predictions(y_true, y_pred, fold_id=1, model_name="m", variant="L1")
    cm = result["confusion_matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)


def test_combined_precision_formula():
    # 4 long signals: 3 correct; 2 short signals: 1 correct → combined = 4/6
    y_true = np.array([1, 1, 1, 0,  -1, 0])
    y_pred = np.array([1, 1, 1, 1,  -1, -1])
    result = evaluate_predictions(y_true, y_pred, fold_id=1, model_name="m", variant="L1")
    # correct_long=3, correct_short=1, total_signals=4+2=6
    assert result["combined_precision"] == pytest.approx(4 / 6)


def test_signal_rate_formula():
    y_true = np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    result = evaluate_predictions(y_true, y_pred, fold_id=1, model_name="m", variant="L1")
    assert result["signal_rate"] == pytest.approx(2 / 10)


def test_aggregate_fold_results_returns_mean_std():
    r1 = _make_result(fold_id=1)
    r2 = _make_result(fold_id=2)
    r3 = _make_result(fold_id=3)
    agg = aggregate_fold_results([r1, r2, r3])
    assert "signal_rate" in agg
    assert "mean" in agg["signal_rate"]
    assert "std"  in agg["signal_rate"]
    assert "min"  in agg["signal_rate"]
    assert "max"  in agg["signal_rate"]


def test_aggregate_identical_folds_std_zero():
    y_true = np.array([1, 1, -1, 0, 0])
    y_pred = np.array([1, 0, -1, 0, 0])
    results = [
        evaluate_predictions(y_true, y_pred, fold_id=i, model_name="m", variant="L1")
        for i in [1, 2, 3]
    ]
    agg = aggregate_fold_results(results)
    assert agg["signal_rate"]["std"] == pytest.approx(0.0)
    assert agg["macro_f1"]["std"] == pytest.approx(0.0)


def test_aggregate_empty_list_returns_empty_dict():
    assert aggregate_fold_results([]) == {}


def test_metrics_keys_present():
    result = _make_result()
    required_keys = [
        "precision_long", "recall_long", "f1_long",
        "precision_short", "recall_short", "f1_short",
        "precision_notrade", "recall_notrade",
        "macro_f1", "weighted_f1",
        "confusion_matrix",
        "signal_rate", "long_precision", "short_precision",
        "combined_precision", "expected_trades_per_day",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
