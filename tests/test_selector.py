"""Tests for models/selector.py"""

import pytest
from models.selector import ModelSelector


def _make_exp(exp_id: str, cp_mean: float, cp_std: float, sr_mean: float) -> dict:
    return {
        "experiment_id": exp_id,
        "model_name": exp_id.lower(),
        "variant": "L1",
        "agg": {
            "combined_precision": {"mean": cp_mean, "std": cp_std},
            "signal_rate":        {"mean": sr_mean, "std": 0.01},
        },
        "fold_results": [],
    }


def test_rank_models_returns_sorted_df():
    experiments = [
        _make_exp("EXP_004", cp_mean=0.28, cp_std=0.02, sr_mean=0.06),
        _make_exp("EXP_005", cp_mean=0.35, cp_std=0.03, sr_mean=0.07),
        _make_exp("EXP_006", cp_mean=0.42, cp_std=0.04, sr_mean=0.06),
    ]
    selector = ModelSelector(experiments)
    ranked = selector.rank_models()
    assert list(ranked["experiment_id"]) == ["EXP_006", "EXP_005", "EXP_004"]


def test_stability_penalty_zero_when_consistent():
    """std/mean <= 0.10 → penalty = 0."""
    # cp_mean=0.40, cp_std=0.04 → consistency=0.10 → penalty=0
    exp = _make_exp("EXP_X", cp_mean=0.40, cp_std=0.04, sr_mean=0.05)
    selector = ModelSelector([exp])
    score, breakdown = selector._score(exp)
    assert breakdown["penalty_instability"] == pytest.approx(0.0)


def test_stability_penalty_increases_above_10pct():
    """std/mean = 0.20 → penalty = (0.20 - 0.10) * 20 = 2.0."""
    exp = _make_exp("EXP_X", cp_mean=0.40, cp_std=0.08, sr_mean=0.05)  # 0.08/0.40 = 0.20
    selector = ModelSelector([exp])
    score, breakdown = selector._score(exp)
    assert breakdown["penalty_instability"] == pytest.approx(2.0, abs=1e-6)


def test_signal_rate_penalty_zero_when_above_2pct():
    """signal_rate >= 0.02 → penalty = 0."""
    exp = _make_exp("EXP_X", cp_mean=0.40, cp_std=0.03, sr_mean=0.05)
    selector = ModelSelector([exp])
    score, breakdown = selector._score(exp)
    assert breakdown["penalty_signal_rate"] == pytest.approx(0.0)


def test_signal_rate_penalty_increases_below_2pct():
    """signal_rate = 0.01 → penalty = (0.02 - 0.01) * 100 = 1.0."""
    exp = _make_exp("EXP_X", cp_mean=0.40, cp_std=0.03, sr_mean=0.01)
    selector = ModelSelector([exp])
    score, breakdown = selector._score(exp)
    assert breakdown["penalty_signal_rate"] == pytest.approx(1.0, abs=1e-6)


def test_select_best_returns_top_ranked():
    experiments = [
        _make_exp("EXP_004", cp_mean=0.28, cp_std=0.02, sr_mean=0.06),
        _make_exp("EXP_005", cp_mean=0.35, cp_std=0.03, sr_mean=0.07),
        _make_exp("EXP_006", cp_mean=0.42, cp_std=0.04, sr_mean=0.06),
    ]
    selector = ModelSelector(experiments)
    best = selector.select_best()
    assert best["experiment_id"] == "EXP_006"


def test_select_best_single_experiment():
    experiments = [_make_exp("EXP_004", cp_mean=0.30, cp_std=0.02, sr_mean=0.05)]
    selector = ModelSelector(experiments)
    best = selector.select_best()
    assert best["experiment_id"] == "EXP_004"


def test_select_best_penalises_unstable_model():
    """A high-precision but unstable model should lose to a lower-precision stable one."""
    # EXP_A: cp=0.50, std/mean=0.50 → penalty=(0.50-0.10)*20=8.0 → score=0.42
    # EXP_B: cp=0.40, std/mean=0.10 → penalty=0 → score=0.40
    # EXP_A still wins here (0.50 - 8.0 = -7.5 vs 0.40), let's use a narrower case
    # EXP_A: cp=0.44, std/mean=0.30 → penalty=(0.30-0.10)*20=4.0 → score=0.40
    # EXP_B: cp=0.41, std/mean=0.05 → penalty=0 → score=0.41
    exp_a = _make_exp("EXP_A", cp_mean=0.44, cp_std=0.132, sr_mean=0.05)  # 0.132/0.44=0.30
    exp_b = _make_exp("EXP_B", cp_mean=0.41, cp_std=0.020, sr_mean=0.05)
    selector = ModelSelector([exp_a, exp_b])
    best = selector.select_best()
    assert best["experiment_id"] == "EXP_B"


def test_rank_models_columns():
    experiments = [_make_exp("EXP_004", cp_mean=0.30, cp_std=0.02, sr_mean=0.05)]
    selector = ModelSelector(experiments)
    ranked = selector.rank_models()
    assert "composite_score" in ranked.columns
    assert "combined_precision" in ranked.columns
    assert "signal_rate" in ranked.columns
