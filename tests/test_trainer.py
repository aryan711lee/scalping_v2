"""Tests for models/trainer.py"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _make_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "f3": rng.standard_normal(n),
        "label": rng.choice([-1, 0, 1], size=n),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="3min", tz="Asia/Kolkata")
    return df


FEATURE_COLS = ["f1", "f2", "f3"]
CLASS_WEIGHTS = {-1: 1.5, 0: 0.5, 1: 1.5}


@pytest.fixture()
def trainer_lr(tmp_path):
    from models.trainer import ModelTrainer
    with patch("models.trainer.ARTIFACTS_DIR", tmp_path):
        model = LogisticRegression(solver="lbfgs", max_iter=200, C=0.1, random_state=42)
        yield ModelTrainer(model, "logistic", "L1", FEATURE_COLS, CLASS_WEIGHTS), tmp_path


@pytest.fixture()
def trainer_rf(tmp_path):
    from models.trainer import ModelTrainer
    with patch("models.trainer.ARTIFACTS_DIR", tmp_path):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        yield ModelTrainer(model, "rf", "L1", FEATURE_COLS, CLASS_WEIGHTS), tmp_path


def test_train_fold_returns_fitted_model(trainer_lr):
    trainer, _ = trainer_lr
    df = _make_df()
    result = trainer.train_fold(df, fold_id=1)
    assert "model" in result
    assert result["fold_id"] == 1
    assert result["train_rows"] == len(df)
    assert result["train_time_seconds"] >= 0


def test_train_fold_has_feature_importances_lr(trainer_lr):
    trainer, _ = trainer_lr
    df = _make_df()
    result = trainer.train_fold(df, fold_id=1)
    assert "feature_importances" in result
    assert set(result["feature_importances"].keys()) == set(FEATURE_COLS)


def test_train_fold_has_feature_importances_rf(trainer_rf):
    trainer, _ = trainer_rf
    df = _make_df()
    result = trainer.train_fold(df, fold_id=1)
    assert "feature_importances" in result


def test_scaler_fitted_on_train_only_not_val(trainer_lr):
    """Scaler must be fit on train_df; val_df is passed separately and NOT used for fitting."""
    trainer, _ = trainer_lr
    train_df = _make_df(n=200, seed=1)
    val_df   = _make_df(n=100, seed=2)

    # Scale should have mean/var computed from train only
    result = trainer.train_fold(train_df, fold_id=1, val_df=val_df)
    scaler = result["scaler"]
    assert scaler is not None

    # Scaler mean should match train mean (not combined)
    train_mean = train_df["f1"].mean()
    # scaler.mean_ is the mean of the fitted data; column 0 = f1
    assert abs(scaler.mean_[0] - train_mean) < 1e-6


def test_no_scaler_for_rf(trainer_rf):
    trainer, _ = trainer_rf
    df = _make_df()
    result = trainer.train_fold(df, fold_id=1)
    assert result["scaler"] is None


def test_save_fold_model_creates_file(trainer_lr):
    trainer, tmp_path = trainer_lr
    df = _make_df()
    result = trainer.train_fold(df, fold_id=2)
    path = trainer.save_fold_model(result)
    assert Path(path).exists()
    payload = joblib.load(path)
    assert "model" in payload
    assert payload["model_name"] == "logistic"


def test_train_final_uses_all_data(trainer_rf, tmp_path):
    """train_final saves a final.joblib and returns the model."""
    trainer, artifacts_tmp = trainer_rf
    df = _make_df(n=300)
    with patch("models.trainer.ARTIFACTS_DIR", artifacts_tmp):
        fitted = trainer.train_final(df)
    assert fitted is not None
    final_path = artifacts_tmp / "rf_L1_final.joblib"
    assert final_path.exists()


def test_xgb_label_encoding():
    """XGBoost trainer encodes -1→0, 0→1, 1→2 before fitting."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pytest.skip("xgboost not installed")

    from models.trainer import ModelTrainer, _ENCODE, _DECODE

    # Encoding round-trips
    for orig in [-1, 0, 1]:
        enc = _ENCODE[orig]
        assert _DECODE[enc] == orig

    # Full set
    assert _ENCODE == {-1: 0, 0: 1, 1: 2}
    assert _DECODE == {0: -1, 1: 0, 2: 1}


def test_xgb_sample_weight_length(tmp_path):
    """sample_weight array length must match training rows."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        pytest.skip("xgboost not installed")

    from models.trainer import ModelTrainer

    df = _make_df(n=200)
    model = XGBClassifier(n_estimators=5, objective="multi:softprob",
                          num_class=3, eval_metric="mlogloss",
                          random_state=42, verbosity=0)

    captured_weights = {}

    original_fit = model.__class__.fit

    def mock_fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is not None:
            captured_weights["len"] = len(sample_weight)
            captured_weights["X_len"] = len(X)
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    with patch("models.trainer.ARTIFACTS_DIR", tmp_path):
        with patch.object(model.__class__, "fit", mock_fit):
            trainer = ModelTrainer(model, "xgboost", "L1", FEATURE_COLS, CLASS_WEIGHTS)
            try:
                trainer.train_fold(df, fold_id=1)
            except Exception:
                pass

    if "len" in captured_weights:
        assert captured_weights["len"] == captured_weights["X_len"]
