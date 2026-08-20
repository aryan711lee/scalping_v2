from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class Trade:
    symbol: str
    direction: str          # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: int
    exit_reason: str        # 'target', 'stoploss', 'time_exit', 'end_of_data'
    gross_pnl: float
    costs: float
    net_pnl: float
    capital_before: float
    capital_after: float


@dataclass
class _OpenPosition:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    quantity: int
    stop_price: float
    target_price: float
    capital_before: float


class Portfolio:
    def __init__(
        self,
        capital: float,
        leverage: float,
        sizing_mode: str,
        position_size_pct: float,
    ):
        self.initial_capital = capital
        self.current_capital = capital
        self.leverage = leverage
        self.sizing_mode = sizing_mode          # 'fixed_pct' or 'full_capital'
        self.position_size_pct = position_size_pct  # e.g. 0.20
        self.open_positions: Dict[str, _OpenPosition] = {}
        self.closed_trades: List[Trade] = []
        self.daily_trade_counts: Dict[Tuple, int] = {}  # (date, symbol) -> int

    def calculate_quantity(self, symbol: str, entry_price: float) -> int:
        if self.sizing_mode == "full_capital":
            buying_power = self.current_capital * self.leverage
        else:
            allocated = self.current_capital * self.position_size_pct
            buying_power = allocated * self.leverage

        qty = math.floor(buying_power / entry_price)
        return max(qty, 1) if buying_power >= entry_price else 0

    def can_trade(self, symbol: str, date: object, max_trades: int) -> bool:
        """Returns True if symbol has not hit the daily trade limit."""
        key = (date, symbol)
        return self.daily_trade_counts.get(key, 0) < max_trades

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_time: datetime,
        entry_price: float,
        quantity: int,
        target_pct: float,
        stoploss_pct: float,
    ) -> None:
        if direction == "long":
            target_price = entry_price * (1 + target_pct)
            stop_price = entry_price * (1 - stoploss_pct)
        else:
            target_price = entry_price * (1 - target_pct)
            stop_price = entry_price * (1 + stoploss_pct)

        self.open_positions[symbol] = _OpenPosition(
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            capital_before=self.current_capital,
        )

        date = entry_time.date()
        key = (date, symbol)
        self.daily_trade_counts[key] = self.daily_trade_counts.get(key, 0) + 1

    def close_position(
        self,
        symbol: str,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        costs: float,
    ) -> Trade:
        pos = self.open_positions.pop(symbol)

        direction_mult = 1 if pos.direction == "long" else -1
        gross_pnl = (exit_price - pos.entry_price) * pos.quantity * direction_mult
        net_pnl = gross_pnl - costs
        capital_after = pos.capital_before + net_pnl
        self.current_capital = capital_after

        trade = Trade(
            symbol=symbol,
            direction=pos.direction,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            quantity=pos.quantity,
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
            capital_before=pos.capital_before,
            capital_after=capital_after,
        )
        self.closed_trades.append(trade)
        return trade

    def reset_daily_counts(self) -> None:
        self.daily_trade_counts = {}
