"""Tests for labels/validator.py"""

import numpy as np
import pandas as pd
import pytest

from labels.validator import (
    _distribution,
    _flag_distribution,
    cross_variant_check,
    validate_labels,
)


def _make_labels(n_long: int, n_short: int, n_none: int) -> pd.Series:
    vals = ([1] * n_long) + ([-1] * n_short) + ([0] * n_none)
    return pd.Series(vals, dtype=np.int8)


# ---------------------------------------------------------------------------
# Distribution checks
# ---------------------------------------------------------------------------

def test_distribution_counts():
    labels = _make_labels(100, 80, 820)
    dist = _distribution(labels)
    assert dist["count_long"]  == 100
    assert dist["count_short"] == 80
    assert dist["count_none"]  == 820
    assert dist["total"]       == 1000


def test_distribution_percentages():
    labels = _make_labels(100, 80, 820)
    dist = _distribution(labels)
    assert abs(dist["long_pct"]  - 10.0) < 0.01
    assert abs(dist["short_pct"] -  8.0) < 0.01
    assert abs(dist["none_pct"]  - 82.0) < 0.01


def test_flag_fires_when_long_pct_too_low():
    labels = _make_labels(1, 80, 919)  # long_pct = 0.1%
    dist = _distribution(labels)
    flags = _flag_distribution(dist)
    assert any("long_pct" in f for f in flags)


def test_flag_fires_when_short_pct_too_low():
    labels = _make_labels(80, 1, 919)
    dist = _distribution(labels)
    flags = _flag_distribution(dist)
    assert any("short_pct" in f for f in flags)


def test_flag_fires_when_none_pct_too_low():
    # none < 30% is suspicious
    labels = _make_labels(400, 400, 200)
    dist = _distribution(labels)
    flags = _flag_distribution(dist)
    assert any("none_pct" in f for f in flags)


def test_no_flag_when_distribution_normal():
    labels = _make_labels(90, 85, 825)  # ~9%, 8.5%, 82.5%
    dist = _distribution(labels)
    flags = _flag_distribution(dist)
    assert flags == [], f"Unexpected flags: {flags}"


def test_flag_fires_when_long_pct_too_high():
    labels = _make_labels(450, 50, 500)  # long_pct = 45%
    dist = _distribution(labels)
    flags = _flag_distribution(dist)
    assert any("long_pct" in f for f in flags)


# ---------------------------------------------------------------------------
# Cross-variant check
# ---------------------------------------------------------------------------

def _make_result(symbol, tf, variant, long_pct, short_pct):
    total = 1000
    n_long  = int(long_pct  / 100 * total)
    n_short = int(short_pct / 100 * total)
    n_none  = total - n_long - n_short
    return {
        "symbol": symbol, "timeframe": tf, "variant": variant,
        "dist": {
            "total": total, "count_long": n_long, "count_short": n_short, "count_none": n_none,
            "long_pct": long_pct, "short_pct": short_pct, "none_pct": 100 - long_pct - short_pct,
        },
        "flags": [], "leakage_errors": [], "flagged": False,
    }


def test_cross_variant_detects_identical_distributions():
    results = [
        _make_result("NSE:TCS-EQ", "3min", "L1", 10.0, 8.0),
        _make_result("NSE:TCS-EQ", "3min", "L2", 10.0, 8.0),  # identical to L1 — bug!
    ]
    warnings = cross_variant_check(results)
    assert len(warnings) > 0, "Expected a warning for identical L1/L2 distributions"


def test_cross_variant_no_warning_when_different():
    results = [
        _make_result("NSE:TCS-EQ", "3min", "L1", 10.0, 8.0),
        _make_result("NSE:TCS-EQ", "3min", "L2",  5.0, 4.0),  # different — correct
    ]
    warnings = cross_variant_check(results)
    assert warnings == [], f"Unexpected warnings: {warnings}"
