"""
Model trainer — fits any sklearn-compatible estimator through walk-forward folds.

XGBoost label encoding note: XGBoost requires labels 0, 1, 2 internally.
This module encodes -1→0, 0→1, 1→2 before training and the predictor
decodes 0→-1, 1→0, 2→+1 on output. The rest of the system always uses -1/0/+1.

StandardScaler is fitted ONLY on training data, then applied to train+val.
The fitted scaler is saved alongside the Logistic Regression model.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Canonical label encoding for XGBoost
_ENCODE = {-1: 0, 0: 1, 1: 2}
_DECODE = {0: -1, 1: 0, 2: 1}

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _requires_scaling(model) -> bool:
    return "LogisticRegression" in type(model).__name__


def _requires_xgb_encoding(model) -> bool:
    return "XGB" in type(model).__name__


class ModelTrainer:
    def __init__(
        self,
        model,
        model_name: str,
        variant: str,
        feature_columns: list,
        class_weights: dict,
    ):
        self.model = model
        self.model_name = model_name
        self.variant = variant
        self.feature_columns = feature_columns
        self.class_weights = class_weights
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def train_fold(
        self,
        train_df: pd.DataFrame,
        fold_id: int,
        val_df: pd.DataFrame = None,
    ) -> dict:
        """
        Trains model on train_df. val_df is optional — used only by XGBoost
        for early stopping (never for any tuning decision; it is the fold val set).

        Returns dict with fold metadata and the fitted model.
        """
        import copy

        X_train = train_df[self.feature_columns].values
        y_train = train_df["label"].values

        scaler = None
        if _requires_scaling(self.model):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        t0 = time.perf_counter()
        fitted_model = copy.deepcopy(self.model)

        if _requires_xgb_encoding(fitted_model):
            y_enc = np.array([_ENCODE[int(y)] for y in y_train])
            sample_weight = np.array([self.class_weights[int(y)] for y in y_train])

            fit_kwargs = {"sample_weight": sample_weight}
            if val_df is not None and hasattr(fitted_model, "early_stopping_rounds"):
                X_val = val_df[self.feature_columns].values
                y_val_enc = np.array([_ENCODE[int(y)] for y in val_df["label"].values])
                fit_kwargs["eval_set"] = [(X_val, y_val_enc)]

            fitted_model.fit(X_train, y_enc, **fit_kwargs)
        else:
            fitted_model.fit(X_train, y_train)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Trained %s fold %d in %.1fs on %d rows",
            self.model_name, fold_id, elapsed, len(X_train),
        )

        result = {
            "fold_id": fold_id,
            "model": fitted_model,
            "scaler": scaler,
            "train_rows": len(X_train),
            "train_time_seconds": elapsed,
        }

        if hasattr(fitted_model, "feature_importances_"):
            importances = dict(zip(
                self.feature_columns,
                fitted_model.feature_importances_.tolist(),
            ))
            result["feature_importances"] = importances
        elif hasattr(fitted_model, "coef_"):
            # Logistic Regression: use mean absolute coefficient across classes
            abs_coef = np.abs(fitted_model.coef_).mean(axis=0)
            importances = dict(zip(self.feature_columns, abs_coef.tolist()))
            result["feature_importances"] = importances

        return result

    def train_final(self, full_train_df: pd.DataFrame) -> object:
        """
        Trains final model on ALL non-test data (fold 1+2+3 training sets combined).
        Saves to models/artifacts/{model_name}_{variant}_final.joblib.
        Returns the fitted model.
        """
        import copy

        X = full_train_df[self.feature_columns].values
        y = full_train_df["label"].values

        scaler = None
        if _requires_scaling(self.model):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        t0 = time.perf_counter()
        fitted_model = copy.deepcopy(self.model)

        if _requires_xgb_encoding(fitted_model):
            y_enc = np.array([_ENCODE[int(lbl)] for lbl in y])
            sample_weight = np.array([self.class_weights[int(lbl)] for lbl in y])
            fitted_model.fit(X, y_enc, sample_weight=sample_weight)
        else:
            fitted_model.fit(X, y)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Trained final %s model in %.1fs on %d rows",
            self.model_name, elapsed, len(X),
        )

        # Save
        path = ARTIFACTS_DIR / f"{self.model_name}_{self.variant}_final.joblib"
        payload = {
            "model": fitted_model,
            "scaler": scaler,
            "feature_columns": self.feature_columns,
            "model_name": self.model_name,
            "variant": self.variant,
            "is_xgb": _requires_xgb_encoding(fitted_model),
        }
        joblib.dump(payload, path)
        logger.info("Saved final model to %s", path)
        return fitted_model

    def save_fold_model(self, fold_result: dict) -> str:
        """
        Saves fold model (and scaler) to models/artifacts/.
        Returns the file path as a string.
        """
        fold_id = fold_result["fold_id"]
        path = ARTIFACTS_DIR / f"{self.model_name}_{self.variant}_fold{fold_id}.joblib"
        payload = {
            "model": fold_result["model"],
            "scaler": fold_result.get("scaler"),
            "feature_columns": self.feature_columns,
            "model_name": self.model_name,
            "variant": self.variant,
            "is_xgb": _requires_xgb_encoding(fold_result["model"]),
        }
        joblib.dump(payload, path)
        logger.debug("Saved fold %d model to %s", fold_id, path)
        return str(path)
