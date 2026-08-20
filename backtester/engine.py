from __future__ import annotations
import logging
from typing import Dict, Optional

import pandas as pd

from backtester.costs import calculate_trade_costs
from backtester.portfolio import Portfolio
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        target_pct: float,
        stoploss_pct: float,
        time_exit_minute: int,
        max_trades_per_symbol_per_day: int,
    ):
        self.strategy = strategy
        self.portfolio = portfolio
        self.target_pct = target_pct
        self.stoploss_pct = stoploss_pct
        self.time_exit_minute = time_exit_minute  # minutes from midnight
        self.max_trades = max_trades_per_symbol_per_day

    def run(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Bar-by-bar simulation following the spec pseudocode order per candle:
          1. Check exits for open positions (using current candle high/low/close)
          2. Generate signals for free symbols (pending order for next candle)
          3. Execute pending orders from PREVIOUS candle at THIS candle's open

        Signal at candle T → fill at open of candle T+1.
        A position opened at T+1 step 3 has its first exit check at candle T+2 step 1.
        """
        # Pre-group every symbol's data by date so per-day lookup is O(1)
        grouped: Dict[str, Dict] = {}
        all_dates: set = set()
        for symbol, df in data.items():
            df = df.sort_values("datetime").reset_index(drop=True)
            df_by_date = {
                date: grp.reset_index(drop=True)
                for date, grp in df.groupby(df["datetime"].dt.date)
            }
            grouped[symbol] = df_by_date
            all_dates.update(df_by_date.keys())
        all_dates_sorted = sorted(all_dates)

        pending_orders: Dict[str, tuple] = {}

        for date in all_dates_sorted:
            self.portfolio.reset_daily_counts()

            day_data: Dict[str, pd.DataFrame] = {
                symbol: dbd[date]
                for symbol, dbd in grouped.items()
                if date in dbd
            }

            if not day_data:
                continue

            # Convert DataFrames to list-of-dicts once per day for fast row access
            day_records: Dict[str, list] = {
                symbol: df.to_dict("records") for symbol, df in day_data.items()
            }
            max_len = max(len(recs) for recs in day_records.values())

            for i in range(max_len):

                # ── Step 1: Check exits for currently open positions ───────────
                for symbol in list(self.portfolio.open_positions.keys()):
                    recs = day_records.get(symbol)
                    if recs is None or i >= len(recs):
                        continue

                    candle = recs[i]
                    pos = self.portfolio.open_positions[symbol]
                    candle_time = candle["datetime"]
                    candle_minute = candle_time.hour * 60 + candle_time.minute

                    exit_price: Optional[float] = None
                    exit_reason: Optional[str] = None

                    if pos.direction == "long":
                        # Both SL and TP hit same candle (gap) → SL wins (conservative)
                        sl_hit = candle["low"] <= pos.stop_price
                        tp_hit = candle["high"] >= pos.target_price
                        if sl_hit:
                            exit_price = pos.stop_price
                            exit_reason = "stoploss"
                        elif tp_hit:
                            exit_price = pos.target_price
                            exit_reason = "target"
                    else:  # short
                        sl_hit = candle["high"] >= pos.stop_price
                        tp_hit = candle["low"] <= pos.target_price
                        if sl_hit:
                            exit_price = pos.stop_price
                            exit_reason = "stoploss"
                        elif tp_hit:
                            exit_price = pos.target_price
                            exit_reason = "target"

                    if exit_reason is None and candle_minute >= self.time_exit_minute:
                        exit_price = candle["close"]
                        exit_reason = "time_exit"

                    if exit_reason is not None:
                        cost_breakdown = calculate_trade_costs(
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            quantity=pos.quantity,
                            direction=pos.direction,
                        )
                        trade = self.portfolio.close_position(
                            symbol=symbol,
                            exit_time=candle_time,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            costs=cost_breakdown["total"],
                        )
                        self.strategy.on_trade_close(trade)

                # ── Step 2: Generate signals for symbols with no open position ─
                new_pending: Dict[str, tuple] = {}
                for symbol, recs in day_records.items():
                    if i >= len(recs):
                        continue
                    if symbol in self.portfolio.open_positions:
                        continue
                    if symbol in pending_orders:
                        continue
                    if not self.portfolio.can_trade(symbol, date, self.max_trades):
                        continue

                    candle = recs[i]
                    signal = self.strategy.generate_signal(candle)
                    if signal != 0:
                        direction = "long" if signal == 1 else "short"
                        quantity = self.portfolio.calculate_quantity(symbol, candle["close"])
                        if quantity == 0:
                            logger.debug(
                                "Skipping %s signal — capital too small at %.2f",
                                symbol, candle["close"],
                            )
                            continue
                        new_pending[symbol] = (direction, quantity)

                # ── Step 3: Execute pending orders from PREVIOUS candle at open ─
                filled = []
                for symbol, (direction, quantity) in pending_orders.items():
                    recs = day_records.get(symbol)
                    if recs is None or i >= len(recs):
                        continue
                    if symbol in self.portfolio.open_positions:
                        continue

                    candle = recs[i]
                    entry_price = candle["open"]
                    entry_time = candle["datetime"]

                    self.portfolio.open_position(
                        symbol=symbol,
                        direction=direction,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        quantity=quantity,
                        target_pct=self.target_pct,
                        stoploss_pct=self.stoploss_pct,
                    )
                    self.strategy.on_trade_open({
                        "symbol": symbol,
                        "direction": direction,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "quantity": quantity,
                    })
                    filled.append(symbol)

                for sym in filled:
                    del pending_orders[sym]

                # Merge new signals into pending for next candle
                pending_orders.update(new_pending)

            # Discard end-of-day pending orders that couldn't fill (no next candle in data)
            # We allow them to carry over to next day's first candle — realistic for
            # orders placed at day close that fill at next morning's open.
            # Discard only if the symbol has no more data.

        # Close any positions still open at end of all data
        for symbol in list(self.portfolio.open_positions.keys()):
            pos = self.portfolio.open_positions[symbol]
            sym_df = data.get(symbol)
            if sym_df is not None and not sym_df.empty:
                last_row = sym_df.iloc[-1]
                exit_price = last_row["close"]
                exit_time = last_row["datetime"]
            else:
                exit_price = pos.entry_price
                exit_time = pos.entry_time

            cost_breakdown = calculate_trade_costs(
                entry_price=pos.entry_price,
                exit_price=exit_price,
                quantity=pos.quantity,
                direction=pos.direction,
            )
            trade = self.portfolio.close_position(
                symbol=symbol,
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason="end_of_data",
                costs=cost_breakdown["total"],
            )
            self.strategy.on_trade_close(trade)
