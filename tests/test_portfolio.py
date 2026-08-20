import pytest
from datetime import datetime, date
from backtester.portfolio import Portfolio


def make_portfolio(capital=50000.0, leverage=5.0, sizing_mode="fixed_pct", pct=0.20):
    return Portfolio(
        capital=capital,
        leverage=leverage,
        sizing_mode=sizing_mode,
        position_size_pct=pct,
    )


class TestFixedPctSizing:
    def test_quantity_uses_correct_percentage_of_capital(self):
        p = make_portfolio(capital=100000.0, leverage=5.0, sizing_mode="fixed_pct", pct=0.10)
        # allocated = 100000 * 0.10 = 10000; buying_power = 10000 * 5 = 50000; qty = floor(50000 / 100) = 500
        qty = p.calculate_quantity("SYM", entry_price=100.0)
        assert qty == 500

    def test_quantity_different_pct(self):
        p = make_portfolio(capital=50000.0, leverage=5.0, sizing_mode="fixed_pct", pct=0.20)
        # allocated = 50000 * 0.20 = 10000; bp = 50000; qty = floor(50000 / 200) = 250
        qty = p.calculate_quantity("SYM", entry_price=200.0)
        assert qty == 250


class TestFullCapitalSizing:
    def test_full_capital_uses_entire_capital_times_leverage(self):
        p = make_portfolio(capital=10000.0, leverage=5.0, sizing_mode="full_capital")
        # bp = 10000 * 5 = 50000; qty = floor(50000 / 50) = 1000
        qty = p.calculate_quantity("SYM", entry_price=50.0)
        assert qty == 1000


class TestMinQuantity:
    def test_quantity_is_at_least_1_for_valid_price(self):
        p = make_portfolio(capital=100.0, leverage=1.0, sizing_mode="fixed_pct", pct=0.01)
        # allocated = 1; bp = 1; price=1; qty = max(1, floor(1/1)) = 1
        qty = p.calculate_quantity("SYM", entry_price=1.0)
        assert qty >= 1

    def test_quantity_is_zero_when_capital_too_small(self):
        # capital=10, sizing=fixed_pct 1%, leverage=1 → allocated=0.10, bp=0.10, price=100 → 0
        p = make_portfolio(capital=10.0, leverage=1.0, sizing_mode="fixed_pct", pct=0.01)
        qty = p.calculate_quantity("SYM", entry_price=100.0)
        assert qty == 0


class TestDailyTradeCount:
    def test_can_trade_within_limit(self):
        p = make_portfolio()
        d = date(2024, 1, 2)
        assert p.can_trade("SYM", d, max_trades=3) is True

    def test_cannot_trade_when_limit_reached(self):
        p = make_portfolio()
        d = date(2024, 1, 2)
        entry = datetime(2024, 1, 2, 9, 30)
        for _ in range(3):
            p.open_position("SYM", "long", entry, 100.0, 10, 0.004, 0.002)
            p.close_position("SYM", entry, 101.0, "target", 50.0)
        assert p.can_trade("SYM", d, max_trades=3) is False

    def test_daily_count_resets(self):
        p = make_portfolio()
        d = date(2024, 1, 2)
        entry = datetime(2024, 1, 2, 9, 30)
        for _ in range(3):
            p.open_position("SYM", "long", entry, 100.0, 10, 0.004, 0.002)
            p.close_position("SYM", entry, 101.0, "target", 50.0)
        assert p.can_trade("SYM", d, max_trades=3) is False
        p.reset_daily_counts()
        assert p.can_trade("SYM", d, max_trades=3) is True


class TestCapitalUpdate:
    def test_capital_increases_after_winning_trade(self):
        p = make_portfolio(capital=50000.0)
        entry = datetime(2024, 1, 2, 9, 30)
        p.open_position("SYM", "long", entry, 100.0, 100, 0.004, 0.002)
        # gross PnL = (105 - 100) * 100 = 500; costs = 50 (mocked); net = 450
        p.close_position("SYM", entry, 105.0, "target", 50.0)
        assert p.current_capital > 50000.0

    def test_capital_decreases_after_losing_trade(self):
        p = make_portfolio(capital=50000.0)
        entry = datetime(2024, 1, 2, 9, 30)
        p.open_position("SYM", "long", entry, 100.0, 100, 0.004, 0.002)
        # gross PnL = (98 - 100) * 100 = -200; costs = 50; net = -250
        p.close_position("SYM", entry, 98.0, "stoploss", 50.0)
        assert p.current_capital < 50000.0

    def test_capital_after_matches_trade_record(self):
        p = make_portfolio(capital=50000.0)
        entry = datetime(2024, 1, 2, 9, 30)
        p.open_position("SYM", "long", entry, 100.0, 100, 0.004, 0.002)
        trade = p.close_position("SYM", entry, 105.0, "target", 50.0)
        assert trade.capital_after == pytest.approx(p.current_capital)
