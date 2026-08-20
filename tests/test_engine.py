"""
Tests for BacktestEngine — execution timing, stop/target logic, time exits,
same-candle gap scenarios, and daily trade limits.
"""
import pytest
from datetime import datetime, timezone
import pandas as pd

from backtester.engine import BacktestEngine
from backtester.portfolio import Portfolio
from strategy.base import BaseStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IST = "Asia/Kolkata"


def make_portfolio(capital=100000.0):
    return Portfolio(
        capital=capital,
        leverage=1.0,
        sizing_mode="full_capital",
        position_size_pct=1.0,
    )


def make_engine(strategy, portfolio, target_pct=0.01, stoploss_pct=0.005,
                time_exit_minute=915, max_trades=3):
    return BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        target_pct=target_pct,
        stoploss_pct=stoploss_pct,
        time_exit_minute=time_exit_minute,
        max_trades_per_symbol_per_day=max_trades,
    )


def make_candle(dt: datetime, open_: float, high: float, low: float, close: float,
                signal_override=0) -> dict:
    return {
        "datetime": pd.Timestamp(dt, tz=IST),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000,
        # Feature columns required by baseline but we use a custom strategy here
        "ema9_above_ema21": signal_override,
        "ema21_above_ema50": signal_override,
        "rsi_14": 50.0,
        "vwap_above": signal_override,
        "volume_ratio": 1.5 if signal_override else 0.5,
        "vol_regime_enc": 0,
        "is_opening_30min": 0,
        "nifty_trend": signal_override,
    }


def make_df(candles: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(candles)


class AlwaysLong(BaseStrategy):
    """Generates long signal on every candle (used to test execution timing)."""
    def generate_signal(self, row):
        return 1


class AlwaysShort(BaseStrategy):
    def generate_signal(self, row):
        return -1


class NeverSignal(BaseStrategy):
    def generate_signal(self, row):
        return 0


class SignalOnFirst(BaseStrategy):
    """Signals long only on the very first call, then never again."""
    def __init__(self):
        self._fired = False

    def generate_signal(self, row):
        if not self._fired:
            self._fired = True
            return 1
        return 0


# ---------------------------------------------------------------------------
# Execution timing: signal at T fills at T+1 open
# ---------------------------------------------------------------------------

class TestExecutionTiming:
    def test_signal_at_candle_T_fills_at_T_plus_1_open(self):
        """
        Candle 0: close triggers long signal
        Candle 1: entry should be at open=110, NOT at candle 0 close=100
        """
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 102, 99, 100),   # T=0, signal fires
            make_candle(datetime(2024, 1, 2, 9, 21), 110, 115, 108, 112),   # T=1, fill at open=110
            make_candle(datetime(2024, 1, 2, 15, 15), 112, 113, 111, 112),  # time exit
        ]
        portfolio = make_portfolio()
        engine = make_engine(AlwaysLong(), portfolio, target_pct=0.10, stoploss_pct=0.05)
        engine.run({"SYM": make_df(candles)})

        assert len(portfolio.closed_trades) >= 1
        trade = portfolio.closed_trades[0]
        assert trade.entry_price == pytest.approx(110.0), \
            f"Expected fill at T+1 open (110), got {trade.entry_price}"


# ---------------------------------------------------------------------------
# Stop loss triggers
# ---------------------------------------------------------------------------

class TestStopLoss:
    def test_long_stop_triggered_when_low_lte_stop_price(self):
        """
        Entry at 100, stop at 99.5 (0.5%). Candle low = 99.0 → stop triggered, fill at 99.5.
        """
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 101, 99, 100),  # signal
            make_candle(datetime(2024, 1, 2, 9, 21), 100, 101, 99, 100),  # entry at open=100
            make_candle(datetime(2024, 1, 2, 9, 24), 100, 100.2, 99.0, 100),  # low hits SL
        ]
        portfolio = make_portfolio()
        engine = make_engine(SignalOnFirst(), portfolio, target_pct=0.01, stoploss_pct=0.005)
        engine.run({"SYM": make_df(candles)})

        assert len(portfolio.closed_trades) == 1
        trade = portfolio.closed_trades[0]
        assert trade.exit_reason == "stoploss"
        assert trade.exit_price == pytest.approx(100.0 * (1 - 0.005))

    def test_short_stop_triggered_when_high_gte_stop_price(self):
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 101, 99, 100),  # signal
            make_candle(datetime(2024, 1, 2, 9, 21), 100, 101, 99, 100),  # entry
            make_candle(datetime(2024, 1, 2, 9, 24), 100, 100.6, 99.5, 100),  # high hits SL
        ]
        portfolio = make_portfolio()
        engine = make_engine(SignalOnFirst(), portfolio, target_pct=0.01, stoploss_pct=0.005)
        # Override to short
        engine.strategy = type("Short", (BaseStrategy,), {"generate_signal": lambda self, r: -1 if not getattr(self, "_fired", False) else setattr(self, "_fired", True) or 0})()
        engine.strategy._done = False

        class ShortOnFirst(BaseStrategy):
            def __init__(self):
                self._fired = False
            def generate_signal(self, row):
                if not self._fired:
                    self._fired = True
                    return -1
                return 0

        portfolio2 = make_portfolio()
        engine2 = make_engine(ShortOnFirst(), portfolio2, target_pct=0.01, stoploss_pct=0.005)
        engine2.run({"SYM": make_df(candles)})

        assert len(portfolio2.closed_trades) == 1
        trade = portfolio2.closed_trades[0]
        assert trade.exit_reason == "stoploss"
        assert trade.exit_price == pytest.approx(100.0 * (1 + 0.005))


# ---------------------------------------------------------------------------
# Target triggers
# ---------------------------------------------------------------------------

class TestTarget:
    def test_long_target_triggered_when_high_gte_target_price(self):
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 101, 99, 100),  # signal
            make_candle(datetime(2024, 1, 2, 9, 21), 100, 101, 99, 100),  # entry at 100
            make_candle(datetime(2024, 1, 2, 9, 24), 100, 102.0, 99.8, 101),  # high >= target 101; low > stop (99.5)
        ]
        portfolio = make_portfolio()
        engine = make_engine(SignalOnFirst(), portfolio, target_pct=0.01, stoploss_pct=0.005)
        engine.run({"SYM": make_df(candles)})

        assert len(portfolio.closed_trades) == 1
        trade = portfolio.closed_trades[0]
        assert trade.exit_reason == "target"
        assert trade.exit_price == pytest.approx(100.0 * 1.01)


# ---------------------------------------------------------------------------
# Time exit
# ---------------------------------------------------------------------------

class TestTimeExit:
    def test_time_exit_triggers_at_or_after_1515(self):
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 101, 99, 100),  # signal
            make_candle(datetime(2024, 1, 2, 9, 21), 100, 101, 99, 100),  # entry
            make_candle(datetime(2024, 1, 2, 15, 15), 100, 100.5, 99.5, 100),  # 15:15 = 915 min
        ]
        portfolio = make_portfolio()
        engine = make_engine(SignalOnFirst(), portfolio, target_pct=0.10, stoploss_pct=0.05, time_exit_minute=915)
        engine.run({"SYM": make_df(candles)})

        assert len(portfolio.closed_trades) == 1
        assert portfolio.closed_trades[0].exit_reason == "time_exit"

    def test_no_time_exit_before_1515(self):
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 101, 99, 100),  # signal
            make_candle(datetime(2024, 1, 2, 9, 21), 100, 101, 99, 100),  # entry
            make_candle(datetime(2024, 1, 2, 15, 14), 100, 100.5, 99.5, 100),  # 15:14 = 914 < 915
        ]
        portfolio = make_portfolio()
        engine = make_engine(SignalOnFirst(), portfolio, target_pct=0.10, stoploss_pct=0.05, time_exit_minute=915)
        engine.run({"SYM": make_df(candles)})

        # Either still open (closed as end_of_data) or time_exit at last candle
        if portfolio.closed_trades:
            # should not be time_exit since minute < 915
            assert portfolio.closed_trades[0].exit_reason != "time_exit"


# ---------------------------------------------------------------------------
# Same-candle gap: both SL and TP hit → SL wins
# ---------------------------------------------------------------------------

class TestGapScenario:
    def test_when_both_sl_and_tp_hit_same_candle_sl_wins(self):
        # Entry at 100, target at 101, stop at 99.5
        # Candle: low=99.0 (SL hit), high=101.5 (TP hit) — both triggered, SL wins
        candles = [
            make_candle(datetime(2024, 1, 2, 9, 18), 100, 101, 99, 100),
            make_candle(datetime(2024, 1, 2, 9, 21), 100, 101, 99, 100),  # entry at 100
            make_candle(datetime(2024, 1, 2, 9, 24), 100, 101.5, 99.0, 100),  # both hit
        ]
        portfolio = make_portfolio()
        engine = make_engine(SignalOnFirst(), portfolio, target_pct=0.01, stoploss_pct=0.005)
        engine.run({"SYM": make_df(candles)})

        assert portfolio.closed_trades[0].exit_reason == "stoploss"


# ---------------------------------------------------------------------------
# Daily trade limit
# ---------------------------------------------------------------------------

class TestDailyTradeLimit:
    def test_daily_limit_prevents_more_than_max_trades(self):
        """With max_trades=2, only 2 trades should be opened per day."""
        candles = []
        # Create many signal candles in one day
        base_minutes = [9*60+18, 9*60+21, 9*60+24, 9*60+27, 9*60+30,
                        9*60+33, 9*60+36, 9*60+39, 9*60+42, 9*60+45]
        for i, m in enumerate(base_minutes):
            h, mi = divmod(m, 60)
            candles.append(make_candle(
                datetime(2024, 1, 2, h, mi), 100, 101, 99.5, 100
            ))
        # Add time exit candle
        candles.append(make_candle(datetime(2024, 1, 2, 15, 15), 100, 101, 99, 100))

        portfolio = make_portfolio(capital=1_000_000.0)
        engine = make_engine(AlwaysLong(), portfolio, target_pct=0.10, stoploss_pct=0.05, max_trades=2)
        engine.run({"SYM": make_df(candles)})

        assert len(portfolio.closed_trades) <= 2
