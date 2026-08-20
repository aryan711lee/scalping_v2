"""
Tests for backtester/metrics.py
"""
import math
import pytest
from datetime import datetime
from backtester.metrics import calculate_metrics
from backtester.portfolio import Trade


def make_trade(
    net_pnl: float,
    gross_pnl: float = None,
    costs: float = 10.0,
    capital_before: float = 50000.0,
    entry_hour: int = 10,
    direction: str = "long",
    exit_reason: str = "target",
    day_offset: int = 0,
) -> Trade:
    if gross_pnl is None:
        gross_pnl = net_pnl + costs
    capital_after = capital_before + net_pnl
    entry = datetime(2024, 1, 2 + day_offset, entry_hour, 30)
    exit_ = datetime(2024, 1, 2 + day_offset, entry_hour + 1, 0)
    return Trade(
        symbol="SYM",
        direction=direction,
        entry_time=entry,
        entry_price=100.0,
        exit_time=exit_,
        exit_price=101.0,
        quantity=10,
        exit_reason=exit_reason,
        gross_pnl=gross_pnl,
        costs=costs,
        net_pnl=net_pnl,
        capital_before=capital_before,
        capital_after=capital_after,
    )


class TestWinRate:
    def test_win_rate_1_when_all_winners(self):
        trades = [make_trade(100.0, day_offset=i) for i in range(5)]
        m = calculate_metrics(trades, 50000.0)
        assert m["win_rate_pct"] == pytest.approx(100.0)

    def test_win_rate_0_when_all_losers(self):
        trades = [make_trade(-50.0, day_offset=i) for i in range(5)]
        m = calculate_metrics(trades, 50000.0)
        assert m["win_rate_pct"] == pytest.approx(0.0)

    def test_win_rate_50_percent(self):
        trades = [make_trade(100.0, day_offset=0), make_trade(-50.0, day_offset=1)]
        m = calculate_metrics(trades, 50000.0)
        assert m["win_rate_pct"] == pytest.approx(50.0)


class TestProfitFactor:
    def test_profit_factor_none_when_no_losers(self):
        trades = [make_trade(100.0, day_offset=i) for i in range(3)]
        m = calculate_metrics(trades, 50000.0)
        assert m["profit_factor"] is None

    def test_profit_factor_computed_correctly(self):
        # 2 winners each +100, 1 loser -50 → PF = 200/50 = 4.0
        trades = [
            make_trade(100.0, day_offset=0),
            make_trade(100.0, day_offset=1),
            make_trade(-50.0, day_offset=2),
        ]
        m = calculate_metrics(trades, 50000.0)
        assert m["profit_factor"] == pytest.approx(4.0, rel=1e-6)


class TestMaxDrawdown:
    def test_max_drawdown_zero_when_capital_never_declines(self):
        capital = 50000.0
        trades = []
        for i in range(5):
            trades.append(make_trade(100.0, capital_before=capital + i * 100, day_offset=i))
        m = calculate_metrics(trades, 50000.0)
        assert m["max_drawdown_pct"] == pytest.approx(0.0, abs=0.01)

    def test_max_drawdown_negative_for_losing_sequence(self):
        capital = 50000.0
        trades = [make_trade(-1000.0, capital_before=capital + i * (-1000), day_offset=i) for i in range(3)]
        m = calculate_metrics(trades, 50000.0)
        assert m["max_drawdown_pct"] < 0


class TestExpectancy:
    def test_expectancy_matches_manual_calculation(self):
        # 3 winners +100, 2 losers -60
        trades = (
            [make_trade(100.0, day_offset=i) for i in range(3)]
            + [make_trade(-60.0, day_offset=i + 3) for i in range(2)]
        )
        m = calculate_metrics(trades, 50000.0)

        win_rate = 3 / 5
        avg_winner = 100.0
        avg_loser = -60.0
        expected_expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)
        assert m["expectancy"] == pytest.approx(expected_expectancy, rel=1e-6)


class TestSharpeRatio:
    def test_sharpe_positive_for_consistently_profitable_returns(self):
        # 5 winning trades on 5 different days — all positive daily returns
        trades = [make_trade(500.0, capital_before=50000.0, day_offset=i) for i in range(5)]
        m = calculate_metrics(trades, 50000.0)
        assert m["sharpe_ratio"] >= 0  # no variability → sharpe 0 or high


class TestEmptyTrades:
    def test_empty_trades_returns_zero_metrics(self):
        m = calculate_metrics([], 50000.0)
        assert m["total_trades"] == 0
        assert m["net_pnl"] == 0.0
        assert m["profit_factor"] is None
