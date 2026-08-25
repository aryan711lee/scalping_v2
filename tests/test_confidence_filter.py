"""Tests for models/confidence_filter.py."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.confidence_filter import ConfidenceFilter


class _MockPredictor:
    """Minimal predictor stub for testing ConfidenceFilter in isolation."""

    def __init__(self, probas: np.ndarray):
        """
        probas: shape (n, 3) — [prob_short, prob_notrade, prob_long]
        """
        self._probas = probas

    @property
    def feature_columns(self) -> list:
        return ["f1", "f2"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._probas

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        n = len(self._probas)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            p_short, p_none, p_long = self._probas[i]
            if p_long > p_short and p_long > threshold:
                labels[i] = 1
            elif p_short > p_long and p_short > threshold:
                labels[i] = -1
        return labels


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f1": [1.0] * n, "f2": [2.0] * n})


def test_top_20pct_signals_pass():
    """Top 20% of rows by max_prob should emit non-zero signals (assuming above floor)."""
    n = 100
    # Varying confidence: row i has max_prob = i/n
    # All signals are long (p_long highest)
    probas = np.zeros((n, 3))
    probas[:, 2] = np.linspace(0.01, 0.99, n)   # p_long increases
    probas[:, 1] = 1.0 - probas[:, 2]            # p_notrade fills the rest

    predictor = _MockPredictor(probas)
    filt = ConfidenceFilter(predictor, top_pct=0.20, window_candles=n, min_confidence=0.0)
    X = _make_df(n)
    preds = filt.predict(X)

    # Approximately top 20% of rows should be non-zero
    signal_rate = (preds != 0).mean()
    assert 0.15 <= signal_rate <= 0.25, f"signal_rate={signal_rate:.2f} outside [0.15, 0.25]"


def test_min_confidence_floor_suppresses_low_confidence():
    """Rows with max_prob < min_confidence must always return 0."""
    n = 20
    # All probas below 0.40 floor
    probas = np.zeros((n, 3))
    probas[:, 2] = 0.35   # p_long = 0.35 < 0.40 floor
    probas[:, 1] = 0.65

    predictor = _MockPredictor(probas)
    filt = ConfidenceFilter(predictor, top_pct=0.50, window_candles=n, min_confidence=0.40)
    X = _make_df(n)
    preds = filt.predict(X)

    assert (preds == 0).all(), f"Expected all zeros below floor, got {preds}"


def test_output_only_contains_valid_labels():
    """Output must only contain -1, 0, or +1."""
    n = 50
    rng = np.random.default_rng(42)
    probas = rng.dirichlet([1, 1, 1], size=n)

    predictor = _MockPredictor(probas)
    filt = ConfidenceFilter(predictor, top_pct=0.30, window_candles=20, min_confidence=0.30)
    X = _make_df(n)
    preds = filt.predict(X)

    unique = set(preds.tolist())
    assert unique.issubset({-1, 0, 1}), f"Unexpected labels: {unique - {-1, 0, 1}}"


def test_lower_top_pct_reduces_signal_rate():
    """Decreasing top_pct should reduce or maintain signal_rate."""
    n = 200
    rng = np.random.default_rng(7)
    probas = rng.dirichlet([1, 3, 1], size=n)  # no-trade dominant

    predictor = _MockPredictor(probas)
    X = _make_df(n)

    prev_sr = 1.0
    for tp in [0.30, 0.20, 0.10]:
        filt = ConfidenceFilter(predictor, top_pct=tp, window_candles=50, min_confidence=0.25)
        sr = filt.signal_rate(X)
        assert sr <= prev_sr + 0.05, f"signal_rate({tp}) = {sr:.3f} > prev {prev_sr:.3f}"
        prev_sr = sr


def test_top_pct_1_matches_base_predictor():
    """With top_pct=1.0 and min_confidence=0, output equals base predictor."""
    n = 30
    rng = np.random.default_rng(99)
    probas = rng.dirichlet([2, 1, 2], size=n)

    predictor = _MockPredictor(probas)
    filt = ConfidenceFilter(predictor, top_pct=1.0, window_candles=n, min_confidence=0.0)
    X = _make_df(n)
    preds_filt = filt.predict(X, threshold=0.5)
    preds_base = predictor.predict(X, threshold=0.5)

    np.testing.assert_array_equal(preds_filt, preds_base)


def test_invalid_top_pct_raises():
    """top_pct outside (0, 1] should raise ValueError."""
    n = 10
    probas = np.full((n, 3), 1 / 3)
    predictor = _MockPredictor(probas)
    with pytest.raises(ValueError, match="top_pct"):
        ConfidenceFilter(predictor, top_pct=0.0, window_candles=50, min_confidence=0.40)


def test_invalid_min_confidence_raises():
    """min_confidence outside (0, 1) should raise ValueError."""
    n = 10
    probas = np.full((n, 3), 1 / 3)
    predictor = _MockPredictor(probas)
    with pytest.raises(ValueError, match="min_confidence"):
        ConfidenceFilter(predictor, top_pct=0.20, window_candles=50, min_confidence=1.5)


def test_rolling_window_causal():
    """
    Filter is causal: each row's rank is computed within [max(0, i-window), i].
    The first few rows are ranked within a small window, not the full dataset.
    """
    n = 50
    # First 10 rows have low confidence, last 40 have high confidence
    probas = np.zeros((n, 3))
    probas[:10, 1] = 0.9   # no-trade wins, max_prob=0.9 for first 10
    probas[10:, 2] = 0.95  # long wins, max_prob=0.95 for rest
    probas[:10, 2] = 0.05
    probas[10:, 1] = 0.05

    predictor = _MockPredictor(probas)
    filt = ConfidenceFilter(predictor, top_pct=0.20, window_candles=20, min_confidence=0.0)
    X = _make_df(n)
    preds = filt.predict(X)

    # In first 10 rows (window builds up), the "top 20%" of [0.9, 0.9, ...] is all 0.9
    # so rows 2+ in the first 10 should sometimes get signals (no-trade is predicted 0 by base)
    # After row 10, long signals (=1) should appear since p_long=0.95 > 0.5 threshold
    signals_after_10 = preds[10:]
    assert (signals_after_10 == 1).sum() > 0, "Expected some long signals after row 10"
