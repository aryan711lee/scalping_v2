"""
Tests for BaselineStrategy — each condition is tested individually and in combination.
"""
import pytest
import pandas as pd
from strategy.baseline import BaselineStrategy


def make_long_row(**overrides) -> pd.Series:
    """Row satisfying all 8 long conditions."""
    base = {
        "ema9_above_ema21": 1,
        "ema21_above_ema50": 1,
        "rsi_14": 52.0,
        "vwap_above": 1,
        "volume_ratio": 1.5,
        "vol_regime_enc": 0,
        "is_opening_30min": 0,
        "nifty_trend": 1,
    }
    base.update(overrides)
    return pd.Series(base)


def make_short_row(**overrides) -> pd.Series:
    """Row satisfying all 8 short conditions."""
    base = {
        "ema9_above_ema21": 0,
        "ema21_above_ema50": 0,
        "rsi_14": 48.0,
        "vwap_above": 0,
        "volume_ratio": 1.5,
        "vol_regime_enc": 0,
        "is_opening_30min": 0,
        "nifty_trend": -1,
    }
    base.update(overrides)
    return pd.Series(base)


strategy = BaselineStrategy()


class TestLongSignal:
    def test_long_signal_when_all_conditions_met(self):
        assert strategy.generate_signal(make_long_row()) == 1

    def test_no_long_when_ema9_not_above_ema21(self):
        assert strategy.generate_signal(make_long_row(ema9_above_ema21=0)) != 1

    def test_no_long_when_ema21_not_above_ema50(self):
        assert strategy.generate_signal(make_long_row(ema21_above_ema50=0)) != 1

    def test_no_long_when_rsi_below_40(self):
        assert strategy.generate_signal(make_long_row(rsi_14=35.0)) != 1

    def test_no_long_when_rsi_above_65(self):
        assert strategy.generate_signal(make_long_row(rsi_14=66.0)) != 1

    def test_no_long_when_rsi_at_lower_boundary(self):
        # RSI = 40 should qualify
        assert strategy.generate_signal(make_long_row(rsi_14=40.0)) == 1

    def test_no_long_when_rsi_at_upper_boundary(self):
        # RSI = 65 should qualify
        assert strategy.generate_signal(make_long_row(rsi_14=65.0)) == 1

    def test_no_long_when_price_below_vwap(self):
        assert strategy.generate_signal(make_long_row(vwap_above=0)) != 1

    def test_no_long_when_volume_ratio_below_1_2(self):
        assert strategy.generate_signal(make_long_row(volume_ratio=1.1)) != 1

    def test_no_long_when_nifty_bearish(self):
        assert strategy.generate_signal(make_long_row(nifty_trend=-1)) != 1

    def test_long_allowed_when_nifty_flat(self):
        assert strategy.generate_signal(make_long_row(nifty_trend=0)) == 1


class TestShortSignal:
    def test_short_signal_when_all_conditions_met(self):
        assert strategy.generate_signal(make_short_row()) == -1

    def test_no_short_when_ema9_above_ema21(self):
        assert strategy.generate_signal(make_short_row(ema9_above_ema21=1)) != -1

    def test_no_short_when_ema21_above_ema50(self):
        assert strategy.generate_signal(make_short_row(ema21_above_ema50=1)) != -1

    def test_no_short_when_rsi_below_35(self):
        assert strategy.generate_signal(make_short_row(rsi_14=30.0)) != -1

    def test_no_short_when_rsi_above_60(self):
        assert strategy.generate_signal(make_short_row(rsi_14=61.0)) != -1

    def test_no_short_when_price_above_vwap(self):
        assert strategy.generate_signal(make_short_row(vwap_above=1)) != -1

    def test_no_short_when_volume_ratio_low(self):
        assert strategy.generate_signal(make_short_row(volume_ratio=1.0)) != -1

    def test_no_short_when_nifty_bullish(self):
        assert strategy.generate_signal(make_short_row(nifty_trend=1)) != -1

    def test_short_allowed_when_nifty_flat(self):
        assert strategy.generate_signal(make_short_row(nifty_trend=0)) == -1


class TestUniversalBlockers:
    def test_no_signal_during_opening_30min_regardless_of_other_conditions(self):
        long_row = make_long_row(is_opening_30min=1)
        short_row = make_short_row(is_opening_30min=1)
        assert strategy.generate_signal(long_row) == 0
        assert strategy.generate_signal(short_row) == 0

    def test_no_signal_in_high_vol_regime_regardless_of_conditions(self):
        long_row = make_long_row(vol_regime_enc=1)
        short_row = make_short_row(vol_regime_enc=1)
        assert strategy.generate_signal(long_row) == 0
        assert strategy.generate_signal(short_row) == 0

    def test_zero_signal_when_no_conditions_met(self):
        neutral = pd.Series({
            "ema9_above_ema21": 1,
            "ema21_above_ema50": 0,
            "rsi_14": 70.0,
            "vwap_above": 0,
            "volume_ratio": 0.8,
            "vol_regime_enc": 0,
            "is_opening_30min": 0,
            "nifty_trend": 0,
        })
        assert strategy.generate_signal(neutral) == 0
