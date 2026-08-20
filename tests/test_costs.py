import pytest
from backtester.costs import calculate_trade_costs


def _costs(entry=100.0, exit=101.0, qty=100, direction="long"):
    return calculate_trade_costs(entry, exit, qty, direction)


class TestBrokerage:
    def test_brokerage_is_40_flat_per_round_trip_long(self):
        c = _costs(entry=100.0, exit=101.0, qty=1, direction="long")
        assert c["brokerage_entry"] + c["brokerage_exit"] == pytest.approx(40.0)

    def test_brokerage_is_40_flat_per_round_trip_short(self):
        c = _costs(entry=100.0, exit=99.0, qty=1, direction="short")
        assert c["brokerage_entry"] + c["brokerage_exit"] == pytest.approx(40.0)

    def test_brokerage_unchanged_for_large_trade(self):
        c = _costs(entry=1000.0, exit=1010.0, qty=10000, direction="long")
        assert c["brokerage_entry"] + c["brokerage_exit"] == pytest.approx(40.0)


class TestSTT:
    def test_stt_applied_on_sell_side_only_long(self):
        # For long: sell side is exit
        qty = 100
        exit_price = 101.0
        c = _costs(entry=100.0, exit=exit_price, qty=qty, direction="long")
        expected_stt = exit_price * qty * 0.00025
        assert c["stt"] == pytest.approx(expected_stt, rel=1e-6)

    def test_stt_applied_on_sell_side_only_short(self):
        # For short: sell side is entry
        qty = 100
        entry_price = 100.0
        c = _costs(entry=entry_price, exit=99.0, qty=qty, direction="short")
        expected_stt = entry_price * qty * 0.00025
        assert c["stt"] == pytest.approx(expected_stt, rel=1e-6)

    def test_stt_positive_for_any_valid_trade(self):
        c = _costs(entry=500.0, exit=495.0, qty=50, direction="long")
        assert c["stt"] > 0


class TestTotalCost:
    def test_total_cost_positive_for_any_valid_trade(self):
        for direction in ("long", "short"):
            c = _costs(entry=100.0, exit=101.0, qty=10, direction=direction)
            assert c["total"] > 0, f"Expected positive total cost for {direction}"

    def test_total_cost_positive_when_trade_is_a_loss(self):
        c = _costs(entry=100.0, exit=99.0, qty=100, direction="long")
        assert c["total"] > 0

    def test_all_components_sum_to_total(self):
        c = _costs(entry=200.0, exit=202.0, qty=50, direction="long")
        component_sum = (
            c["brokerage_entry"] + c["brokerage_exit"]
            + c["stt"]
            + c["exchange_fees_entry"] + c["exchange_fees_exit"]
            + c["gst"]
            + c["stamp_duty"]
            + c["sebi_charges"]
            + c["slippage_entry"] + c["slippage_exit"]
        )
        assert component_sum == pytest.approx(c["total"], rel=1e-9)


class TestSlippage:
    def test_slippage_scales_with_trade_value(self):
        c_small = _costs(entry=100.0, exit=101.0, qty=10, direction="long")
        c_large = _costs(entry=100.0, exit=101.0, qty=1000, direction="long")
        assert c_large["slippage_entry"] > c_small["slippage_entry"]
        assert c_large["slippage_exit"] > c_small["slippage_exit"]

    def test_slippage_proportional_to_quantity(self):
        c1 = _costs(entry=100.0, exit=101.0, qty=1, direction="long")
        c100 = _costs(entry=100.0, exit=101.0, qty=100, direction="long")
        assert c100["slippage_entry"] == pytest.approx(c1["slippage_entry"] * 100, rel=1e-9)
