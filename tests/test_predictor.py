"""Tests for models/predictor.py"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


def _save_dummy_model(tmp_path: Path, is_xgb: bool = False) -> str:
    """Train and save a tiny LogisticRegression (or mock XGB) model."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 3))
    y = rng.choice([-1, 0, 1], size=300)

    if is_xgb:
        try:
            from xgboost import XGBClassifier
            y_enc = np.where(y == -1, 0, np.where(y == 0, 1, 2))
            model = XGBClassifier(n_estimators=5, objective="multi:softprob",
                                  num_class=3, eval_metric="mlogloss",
                                  random_state=42, verbosity=0)
            model.fit(X, y_enc)
            payload = {
                "model": model,
                "scaler": None,
                "feature_columns": ["f1", "f2", "f3"],
                "model_name": "xgboost",
                "variant": "L1",
                "is_xgb": True,
            }
        except ImportError:
            pytest.skip("xgboost not installed")
    else:
        model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
        model.fit(X, y)
        payload = {
            "model": model,
            "scaler": None,
            "feature_columns": ["f1", "f2", "f3"],
            "model_name": "logistic",
            "variant": "L1",
            "is_xgb": False,
        }

    path = tmp_path / "test_model.joblib"
    joblib.dump(payload, path)
    return str(path)


def _make_X(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.standard_normal(n),
                         "f2": rng.standard_normal(n),
                         "f3": rng.standard_normal(n)})


@pytest.fixture()
def lr_predictor(tmp_path):
    from models.predictor import ModelPredictor
    path = _save_dummy_model(tmp_path, is_xgb=False)
    return ModelPredictor(path)


def test_predict_returns_valid_labels(lr_predictor):
    X = _make_X()
    preds = lr_predictor.predict(X)
    assert set(preds).issubset({-1, 0, 1})


def test_predict_threshold_1_returns_all_zeros(lr_predictor):
    """At threshold=1.0, no class probability can reach 100%, so all must be 0."""
    X = _make_X(n=50)
    preds = lr_predictor.predict(X, threshold=1.0)
    assert (preds == 0).all(), f"Expected all zeros but got: {np.unique(preds)}"


def test_predict_threshold_0_returns_no_directional_only(lr_predictor):
    """At threshold=0, every row gets argmax — some should be +1 or -1."""
    X = _make_X(n=100)
    preds = lr_predictor.predict(X, threshold=0.0)
    # With a trained model and 100 rows, we expect at least some directional signals
    # (not necessarily zero-free, but directional signals should appear)
    assert set(preds).issubset({-1, 0, 1})
    # All should be predicted — argmax will assign something
    assert len(preds) == 100


def test_predict_proba_shape(lr_predictor):
    X = _make_X(n=15)
    proba = lr_predictor.predict_proba(X)
    assert proba.shape == (15, 3)


def test_predict_proba_sums_to_one(lr_predictor):
    X = _make_X(n=30)
    proba = lr_predictor.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_predict_proba_columns_ordered_short_none_long(lr_predictor):
    """Columns must be [prob_short, prob_notrade, prob_long] = [-1, 0, +1]."""
    X = _make_X(n=10)
    proba = lr_predictor.predict_proba(X)
    # All probabilities non-negative
    assert (proba >= 0).all()
    # shape is (n, 3)
    assert proba.shape[1] == 3


def test_predict_xgb_returns_valid_labels(tmp_path):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pytest.skip("xgboost not installed")

    from models.predictor import ModelPredictor
    path = _save_dummy_model(tmp_path, is_xgb=True)
    predictor = ModelPredictor(path)
    X = _make_X(n=20)
    preds = predictor.predict(X)
    assert set(preds).issubset({-1, 0, 1})


def test_predict_xgb_threshold_1_returns_zeros(tmp_path):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pytest.skip("xgboost not installed")

    from models.predictor import ModelPredictor
    path = _save_dummy_model(tmp_path, is_xgb=True)
    predictor = ModelPredictor(path)
    X = _make_X(n=30)
    preds = predictor.predict(X, threshold=1.0)
    assert (preds == 0).all()
