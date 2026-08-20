import numpy as np
import pandas as pd
import pytz
import pytest

from features.momentum import add_momentum, wilder_rsi

IST = pytz.timezone("Asia/Kolkata")


def _make_df(closes) -> pd.DataFrame:
    n = len(closes)
    base = pd.Timestamp("2023-01-02 09:15:00", tz=IST)
    dt = [base + pd.Timedelta(minutes=i) for i in range(n)]
    return pd.DataFrame(
        dict(datetime=dt, open=closes, high=closes, low=closes, close=closes, volume=1000)
    )


def test_rsi_100_when_all_gains():
    """When every period is a gain, RSI should converge to 100."""
    # strictly increasing prices → avg_loss = 0 → RSI = 100
    closes = [100 + i for i in range(50)]
    df = _make_df(closes)
    out = add_momentum(df)
    assert out["rsi_14"].iloc[-1] == pytest.approx(100.0, abs=1e-6)


def test_rsi_0_when_all_losses():
    """When every period is a loss, RSI should converge to 0."""
    closes = [100 - i * 0.5 for i in range(50)]
    df = _make_df(closes)
    out = add_momentum(df)
    assert out["rsi_14"].iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_between_0_and_100_on_random_data():
    rng = np.random.default_rng(42)
    closes = (100 + rng.standard_normal(500).cumsum()).tolist()
    df = _make_df(closes)
    out = add_momentum(df)
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_uses_wilder_not_simple_rolling():
    """
    A simple 14-period rolling average gives different results than Wilder's smoothing.
    Verify that our RSI matches the Wilder formula, not the simple average.
    """
    rng = np.random.default_rng(7)
    closes = pd.Series(100 + rng.standard_normal(100).cumsum())

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder (our implementation)
    avg_gain_w = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss_w = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rsi_wilder = 100 - 100 / (1 + avg_gain_w / avg_loss_w.replace(0, float("nan")))

    # Simple rolling (wrong)
    avg_gain_s = gain.rolling(14).mean()
    avg_loss_s = loss.rolling(14).mean()
    rsi_simple = 100 - 100 / (1 + avg_gain_s / avg_loss_s.replace(0, float("nan")))

    # They must differ after enough bars
    diff = (rsi_wilder - rsi_simple).dropna().abs()
    assert diff.max() > 0.1, "Wilder and simple RSI should differ"

    # And our function matches Wilder
    rsi_ours = wilder_rsi(closes, 14)
    assert np.allclose(rsi_ours.dropna(), rsi_wilder.dropna(), atol=1e-9)
