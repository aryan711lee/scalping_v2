"""Tests for labels/variants.py"""
import pytest
from labels.variants import LABEL_VARIANTS


def test_all_variants_present():
    assert set(LABEL_VARIANTS.keys()) == {"L1", "L2", "L3", "L4"}


def test_required_keys():
    for name, cfg in LABEL_VARIANTS.items():
        assert "target_pct" in cfg, f"{name} missing target_pct"
        assert "stop_pct"   in cfg, f"{name} missing stop_pct"
        assert "horizon"    in cfg, f"{name} missing horizon"
        assert "description" in cfg, f"{name} missing description"


def test_target_greater_than_stop_for_2to1_variants():
    """L1, L2, L4 should have target_pct > stop_pct (reward > risk)."""
    for variant in ("L1", "L2", "L4"):
        cfg = LABEL_VARIANTS[variant]
        assert cfg["target_pct"] > cfg["stop_pct"], (
            f"{variant}: target_pct {cfg['target_pct']} not > stop_pct {cfg['stop_pct']}"
        )


def test_l3_symmetric():
    """L3 has equal target_pct and stop_pct (1:1 RR)."""
    cfg = LABEL_VARIANTS["L3"]
    assert cfg["target_pct"] == cfg["stop_pct"]


def test_horizons_are_positive_integers():
    for name, cfg in LABEL_VARIANTS.items():
        h = cfg["horizon"]
        assert isinstance(h, int) and h > 0, f"{name}: horizon must be positive int, got {h}"


def test_percentages_are_float():
    for name, cfg in LABEL_VARIANTS.items():
        assert isinstance(cfg["target_pct"], float), f"{name}: target_pct not float"
        assert isinstance(cfg["stop_pct"], float),   f"{name}: stop_pct not float"


def test_percentages_in_reasonable_range():
    for name, cfg in LABEL_VARIANTS.items():
        assert 0 < cfg["target_pct"] < 0.05, f"{name}: target_pct out of range"
        assert 0 < cfg["stop_pct"]   < 0.05, f"{name}: stop_pct out of range"
