"""Tests for strategy/ml_strategy.py"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


def _save_dummy_predictor(tmp_path: Path) -> str:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 3))
    y = rng.choice([-1, 0, 1], size=300)
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


@pytest.fixture()
def predictor(tmp_path):
    from models.predictor import ModelPredictor
    path = _save_dummy_predictor(tmp_path)
    return ModelPredictor(path)


FEATURE_COLS = ["f1", "f2", "f3"]


def _make_row(**kwargs) -> dict:
    base = {"f1": 0.1, "f2": -0.2, "f3": 0.3, "is_closing_30min": 0}
    base.update(kwargs)
    return base


def test_inherits_base_strategy(predictor):
    from strategy.base import BaseStrategy
    from strategy.ml_strategy import MLStrategy
    strategy = MLStrategy(predictor=predictor, feature_columns=FEATURE_COLS,
                          threshold=0.5, avoid_afternoon=True)
    assert isinstance(strategy, BaseStrategy)


def test_generate_signal_returns_zero_afternoon_filter(predictor):
    """When avoid_afternoon=True and is_closing_30min==1, must return 0."""
    from strategy.ml_strategy import MLStrategy
    strategy = MLStrategy(predictor=predictor, feature_columns=FEATURE_COLS,
                          threshold=0.5, avoid_afternoon=True)
    row = _make_row(is_closing_30min=1)
    signal = strategy.generate_signal(row)
    assert signal == 0


def test_generate_signal_passes_through_when_afternoon_disabled(predictor):
    """When avoid_afternoon=False, is_closing_30min==1 does NOT filter."""
    from strategy.ml_strategy import MLStrategy
    strategy = MLStrategy(predictor=predictor, feature_columns=FEATURE_COLS,
                          threshold=0.5, avoid_afternoon=False)
    # Even if is_closing_30min==1, the model prediction runs
    row = _make_row(is_closing_30min=1)
    signal = strategy.generate_signal(row)
    assert signal in {-1, 0, 1}


def test_generate_signal_returns_model_prediction_when_filters_pass(predictor):
    """Normal morning candle should return whatever the model predicts."""
    from strategy.ml_strategy import MLStrategy
    strategy = MLStrategy(predictor=predictor, feature_columns=FEATURE_COLS,
                          threshold=0.5, avoid_afternoon=True)
    row = _make_row(is_closing_30min=0)
    signal = strategy.generate_signal(row)
    assert signal in {-1, 0, 1}


def test_generate_signal_valid_for_pd_series(predictor):
    """generate_signal must also accept a pd.Series (used in tests/manual use)."""
    from strategy.ml_strategy import MLStrategy
    strategy = MLStrategy(predictor=predictor, feature_columns=FEATURE_COLS,
                          threshold=0.5, avoid_afternoon=False)
    row = pd.Series({"f1": 0.1, "f2": -0.2, "f3": 0.3, "is_closing_30min": 0})
    signal = strategy.generate_signal(row)
    assert signal in {-1, 0, 1}


def test_ml_strategy_plugs_into_backtester_engine(predictor, tmp_path):
    """
    Run a short synthetic backtest using MLStrategy — verify no errors raised.
    Uses a minimal in-memory DataFrame with required OHLCV columns.
    """
    from backtester.engine import BacktestEngine
    from backtester.portfolio import Portfolio
    from strategy.ml_strategy import MLStrategy

    strategy = MLStrategy(predictor=predictor, feature_columns=FEATURE_COLS,
                          threshold=0.5, avoid_afternoon=False)

    portfolio = Portfolio(
        capital=50000,
        leverage=5,
        sizing_mode="fixed_pct",
        position_size_pct=0.20,
    )

    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        target_pct=0.004,
        stoploss_pct=0.002,
        time_exit_minute=15 * 60 + 15,
        max_trades_per_symbol_per_day=3,
    )

    # Build synthetic price data with feature columns attached
    n = 60
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-02-01 09:30", periods=n, freq="3min")

    rows = []
    price = 1000.0
    for dt in dates:
        ret = rng.normal(0, 0.003)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + rng.uniform(0, 0.001))
        l = min(o, c) * (1 - rng.uniform(0, 0.001))
        rows.append({
            "datetime": dt,
            "open": o, "high": h, "low": l, "close": c, "volume": 10000,
            "f1": rng.standard_normal(), "f2": rng.standard_normal(), "f3": rng.standard_normal(),
            "is_closing_30min": 0,
        })
        price = c

    df = pd.DataFrame(rows)

    data = {"NSE:RELIANCE-EQ": df}
    engine.run(data)

    # No exception — verify trades list is accessible
    assert isinstance(portfolio.closed_trades, list)
