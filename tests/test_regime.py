import numpy as np
import pandas as pd
import pytz
import pytest

from features.price_action import add_price_action
from features.regime import add_regime
from features.trend import add_trend
from features.volatility import add_volatility

IST = pytz.timezone("Asia/Kolkata")


def _make_df(n: int = 100, close: float = 100.0) -> pd.DataFrame:
    base = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    dt = [base + pd.Timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame(
        dict(datetime=dt, open=close, high=close + 1, low=close - 1, close=close, volume=1000)
    )


def _build(closes) -> pd.DataFrame:
    n = len(closes)
    base = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    dt = [base + pd.Timedelta(minutes=i) for i in range(n)]
    df = pd.DataFrame(
        dict(
            datetime=dt,
            open=closes,
            high=[c + 0.5 for c in closes],
            low=[c - 0.5 for c in closes],
            close=closes,
            volume=[1000] * n,
        )
    )
    df = add_price_action(df)
    df = add_trend(df)
    df = add_volatility(df)
    return add_regime(df)


def test_trend_regime_uptrend():
    """Strongly rising series → ema9 > ema21 > ema50 and close > ema9 → uptrend."""
    closes = list(range(1, 201))
    out = _build(closes)
    assert out["trend_regime_enc"].iloc[-1] == 1
    assert out["trend_regime"].iloc[-1] == "uptrend"


def test_trend_regime_downtrend():
    """Strongly falling series → ema9 < ema21 < ema50 and close < ema9 → downtrend."""
    closes = list(range(200, 0, -1))
    out = _build(closes)
    assert out["trend_regime_enc"].iloc[-1] == -1
    assert out["trend_regime"].iloc[-1] == "downtrend"


def test_trend_regime_enc_only_valid_values():
    rng = np.random.default_rng(11)
    closes = (100 + rng.standard_normal(300).cumsum()).tolist()
    out = _build(closes)
    assert set(out["trend_regime_enc"].unique()).issubset({-1, 0, 1})


def test_vol_regime_enc_high_when_atr_in_top_quartile():
    """After computing regime, vol_regime_enc=1 should appear when ATR is high."""
    rng = np.random.default_rng(42)
    closes = (100 + rng.standard_normal(300).cumsum()).tolist()
    out = _build(closes)
    # Check: wherever atr_percentile > 0.75, vol_regime_enc == 1
    mask = out["atr_percentile"] > 0.75
    if mask.any():
        assert (out.loc[mask, "vol_regime_enc"] == 1).all()


def test_vol_regime_enc_only_valid_values():
    rng = np.random.default_rng(7)
    closes = (100 + rng.standard_normal(200).cumsum()).tolist()
    out = _build(closes)
    assert set(out["vol_regime_enc"].dropna().unique()).issubset({-1, 0, 1})
