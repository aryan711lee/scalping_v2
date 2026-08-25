"""
EXP_013: Confidence-weighted signal filtering.

Instead of a fixed threshold applied uniformly, this filter ranks each row's
max-probability by its position within a rolling window. Only the top N% most
confident predictions in each window are allowed through as signals.

Rationale: a fixed threshold is too blunt across varying market conditions.
In choppy periods, the model is uniformly uncertain and a fixed threshold lets
too many marginal signals through. Ranking within a window naturally suppresses
trades during low-confidence regimes.
"""

import logging

import numpy as np
import pandas as pd

from models.predictor import ModelPredictor

logger = logging.getLogger(__name__)


class ConfidenceFilter:
    """
    Wraps a ModelPredictor and filters its output by confidence rank.

    Parameters
    ----------
    base_predictor : ModelPredictor
        The underlying model to generate raw probability scores.
    top_pct : float
        Fraction of highest-confidence rows to keep as signals (e.g. 0.20 = top 20%).
    window_candles : int
        Rolling window size in rows for ranking confidence scores.
    min_confidence : float
        Hard floor — never emit a signal if max_prob < min_confidence.
    """

    def __init__(
        self,
        base_predictor: ModelPredictor,
        top_pct: float = 0.20,
        window_candles: int = 100,
        min_confidence: float = 0.40,
    ):
        if not (0 < top_pct <= 1.0):
            raise ValueError(f"top_pct must be in (0, 1], got {top_pct}")
        if window_candles < 10:
            raise ValueError(f"window_candles must be >= 10, got {window_candles}")
        if not (0 <= min_confidence < 1):
            raise ValueError(f"min_confidence must be in [0, 1), got {min_confidence}")

        self.base_predictor = base_predictor
        self.top_pct = top_pct
        self.window_candles = window_candles
        self.min_confidence = min_confidence

    @property
    def feature_columns(self) -> list:
        return self.base_predictor.feature_columns

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Generate filtered predictions using confidence ranking.

        Steps:
          1. Get raw probabilities from base_predictor.predict_proba()
          2. Compute max_prob for each row (confidence score)
          3. Hard floor: if max_prob < min_confidence → emit 0
          4. For each row, rank its max_prob within the last window_candles rows
             If rank is in top top_pct fraction → emit the base prediction
             Otherwise → emit 0
          5. Return array of -1, 0, +1

        The window uses only past observations (causal) — row i looks back at
        rows max(0, i-window_candles) to i (inclusive).
        Uses vectorized pandas rolling quantile for performance.
        """
        proba = self.base_predictor.predict_proba(X)   # shape (n, 3) → [short, none, long]
        base_preds = self.base_predictor.predict(X, threshold=threshold)

        max_prob = proba.max(axis=1)                   # shape (n,)

        # Hard floor mask
        floor_pass = max_prob >= self.min_confidence

        # Vectorized rolling quantile — causal (closed='left' uses rows up to and
        # including current, which is what we want for the inclusive window)
        sr = pd.Series(max_prob)
        rolling_cutoff = sr.rolling(
            window=self.window_candles, min_periods=1
        ).quantile(1.0 - self.top_pct)
        rank_pass = max_prob >= rolling_cutoff.values

        # Emit signal only when both floor and rank conditions are met
        active = floor_pass & rank_pass
        labels = np.where(active, base_preds, 0).astype(int)

        return labels

    def signal_rate(self, X: pd.DataFrame, threshold: float = 0.5) -> float:
        """Fraction of rows where a non-zero signal is emitted."""
        preds = self.predict(X, threshold=threshold)
        return float((preds != 0).mean())

    @classmethod
    def sweep(
        cls,
        base_predictor: ModelPredictor,
        X: pd.DataFrame,
        y_true: np.ndarray,
        top_pcts: list[float] = None,
        window_candles_list: list[int] = None,
        min_confidences: list[float] = None,
    ) -> list[dict]:
        """
        Sweep hyperparameters and return results sorted by combined_precision
        subject to signal_rate >= 1.5%.

        Returns list of dicts with keys:
          top_pct, window_candles, min_confidence,
          combined_precision, signal_rate, viable
        """
        from validation.evaluator import evaluate_predictions

        top_pcts = top_pcts or [0.10, 0.15, 0.20, 0.25, 0.30]
        window_candles_list = window_candles_list or [50, 100, 200]
        min_confidences = min_confidences or [0.35, 0.40, 0.45]

        rows = []
        for tp in top_pcts:
            for wc in window_candles_list:
                for mc in min_confidences:
                    filt = cls(
                        base_predictor=base_predictor,
                        top_pct=tp,
                        window_candles=wc,
                        min_confidence=mc,
                    )
                    y_pred = filt.predict(X)
                    metrics = evaluate_predictions(
                        y_pred=y_pred,
                        y_true=y_true,
                        fold_id=0,
                        model_name="confidence_filter",
                        variant="L1",
                    )
                    cp = (metrics.get("combined_precision") or 0.0)
                    sr = metrics.get("signal_rate", 0.0)
                    rows.append({
                        "top_pct": tp,
                        "window_candles": wc,
                        "min_confidence": mc,
                        "combined_precision": cp,
                        "signal_rate": sr,
                        "viable": sr >= 0.015,
                    })

        return sorted(rows, key=lambda r: (r["viable"], r["combined_precision"]), reverse=True)
