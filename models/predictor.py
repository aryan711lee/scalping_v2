"""
Model predictor — loads a saved model and generates predictions.

The threshold parameter controls signal selectivity:
  - Default 0.5: predict whichever class has highest probability
  - Higher threshold (e.g. 0.6): only predict +1/-1 if that class
    probability exceeds the threshold, otherwise predict 0

XGBoost label decoding: model internally uses 0/1/2 → this module
decodes back to -1/0/+1 before returning. The caller always sees -1/0/+1.

predict_proba columns are always ordered [prob_short, prob_notrade, prob_long]
corresponding to labels [-1, 0, +1] regardless of model type.
"""

import logging

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# XGBoost internal label → system label
_DECODE = {0: -1, 1: 0, 2: 1}
# System label → XGBoost column index in predict_proba output
# XGBoost trains with classes 0(→-1), 1(→0), 2(→+1)
# predict_proba returns columns in class order: [col0=prob(-1), col1=prob(0), col2=prob(+1)]
_SHORT_IDX = 0   # column for -1 (short)
_NONE_IDX  = 1   # column for 0 (no-trade)
_LONG_IDX  = 2   # column for +1 (long)


class ModelPredictor:
    def __init__(self, model_path: str, scaler_path: str = None):
        """
        Loads saved model (and optional scaler) from disk.
        model_path: path to a .joblib file saved by ModelTrainer.
        scaler_path: ignored — scaler is bundled inside the .joblib payload.
        """
        payload = joblib.load(model_path)
        self._model = payload["model"]
        self._scaler = payload.get("scaler")
        self._feature_columns = payload["feature_columns"]
        self._model_name = payload.get("model_name", "unknown")
        self._is_xgb = payload.get("is_xgb", False)
        logger.info("Loaded model %s from %s", self._model_name, model_path)

    @property
    def feature_columns(self) -> list:
        return self._feature_columns

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        arr = X[self._feature_columns].values
        if self._scaler is not None:
            arr = self._scaler.transform(arr)
        return arr

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Returns predicted labels as array of -1, 0, +1.

        When threshold > 0: a directional signal (+1 or -1) is emitted only if
        the winning class probability strictly exceeds the threshold; otherwise 0.

        When threshold == 0: every row gets the highest-probability class
        (including 0/no-trade if that's the max), but for clarity: threshold=0
        means we never withhold a signal based on probability confidence.
        """
        arr = self._prepare(X)
        proba = self._model.predict_proba(arr)  # shape (n, 3)

        # Map proba columns to short/none/long
        if self._is_xgb:
            # XGBoost: columns ordered by encoded class 0,1,2 → short,none,long
            p_short = proba[:, _SHORT_IDX]
            p_none  = proba[:, _NONE_IDX]
            p_long  = proba[:, _LONG_IDX]
        else:
            # sklearn classifiers: columns ordered by model.classes_
            classes = list(self._model.classes_)
            p_short = proba[:, classes.index(-1)]
            p_none  = proba[:, classes.index(0)]
            p_long  = proba[:, classes.index(1)]

        n = len(proba)
        labels = np.zeros(n, dtype=int)

        if threshold <= 0.0:
            # Pure argmax across all three classes
            for i in range(n):
                idx = np.argmax([p_short[i], p_none[i], p_long[i]])
                labels[i] = [-1, 0, 1][idx]
        else:
            for i in range(n):
                if p_long[i] > p_short[i] and p_long[i] > threshold:
                    labels[i] = 1
                elif p_short[i] > p_long[i] and p_short[i] > threshold:
                    labels[i] = -1
                else:
                    labels[i] = 0

        return labels

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns raw probability array shape (n_rows, 3).
        Columns: [prob_short, prob_notrade, prob_long] (order: -1, 0, +1).
        """
        arr = self._prepare(X)
        raw = self._model.predict_proba(arr)

        if self._is_xgb:
            return raw  # already in [short, none, long] order
        else:
            classes = list(self._model.classes_)
            out = np.zeros_like(raw)
            out[:, 0] = raw[:, classes.index(-1)]
            out[:, 1] = raw[:, classes.index(0)]
            out[:, 2] = raw[:, classes.index(1)]
            return out
