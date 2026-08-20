from __future__ import annotations
import math
from typing import List, Optional

import pandas as pd

from backtester.portfolio import Trade


def calculate_metrics(trades: List[Trade], initial_capital: float) -> dict:
    """Compute full performance metrics from the list of closed Trade objects."""
    if not trades:
        return _empty_metrics()

    total = len(trades)
    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]

    win_rate = len(winners) / total if total else 0.0
    gross_pnl = sum(t.gross_pnl for t in trades)
    total_costs = sum(t.costs for t in trades)
    net_pnl = sum(t.net_pnl for t in trades)

    avg_winner = sum(t.net_pnl for t in winners) / len(winners) if winners else 0.0
    avg_loser = sum(t.net_pnl for t in losers) / len(losers) if losers else 0.0
    avg_net = net_pnl / total if total else 0.0

    winner_sum = sum(t.net_pnl for t in winners)
    loser_sum = abs(sum(t.net_pnl for t in losers))
    profit_factor: Optional[float] = winner_sum / loser_sum if loser_sum > 0 else None

    expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)

    net_return_pct = (net_pnl / initial_capital) * 100.0

    # Build daily P&L series
    trade_df = pd.DataFrame([{
        "date": t.exit_time.date(),
        "net_pnl": t.net_pnl,
        "capital_before": t.capital_before,
    } for t in trades])

    daily_pnl = trade_df.groupby("date").agg(
        daily_net=("net_pnl", "sum"),
        cap_start=("capital_before", "first"),
    ).reset_index()
    daily_pnl["daily_return"] = daily_pnl["daily_net"] / daily_pnl["cap_start"]

    # Max drawdown
    capital_series = _build_capital_series(trades, initial_capital)
    max_dd_pct, max_dd_duration = _max_drawdown(capital_series)

    # Sharpe ratio (annualized)
    returns = daily_pnl["daily_return"]
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
    else:
        sharpe = 0.0

    # Calmar ratio
    trading_days = len(daily_pnl)
    annualized_return = (net_return_pct / 100.0) * (252 / trading_days) if trading_days else 0.0
    calmar = annualized_return / abs(max_dd_pct / 100.0) if max_dd_pct != 0 else None

    # Exit breakdown
    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Regime breakdown
    regime_stats = _regime_breakdown(trades)

    # Time-of-day breakdown
    tod_stats = _time_of_day_breakdown(trades)

    return {
        "total_trades": total,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate_pct": win_rate * 100.0,
        "gross_pnl": gross_pnl,
        "total_costs": total_costs,
        "net_pnl": net_pnl,
        "net_return_pct": net_return_pct,
        "avg_net_pnl_per_trade": avg_net,
        "avg_winner": avg_winner,
        "avg_loser": avg_loser,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown_pct": max_dd_pct,
        "max_drawdown_duration_days": max_dd_duration,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "exit_breakdown": exit_counts,
        "regime_stats": regime_stats,
        "time_of_day_stats": tod_stats,
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "gross_pnl": 0.0,
        "total_costs": 0.0,
        "net_pnl": 0.0,
        "net_return_pct": 0.0,
        "avg_net_pnl_per_trade": 0.0,
        "avg_winner": 0.0,
        "avg_loser": 0.0,
        "profit_factor": None,
        "expectancy": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_duration_days": 0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": None,
        "exit_breakdown": {},
        "regime_stats": {},
        "time_of_day_stats": {},
    }


def _build_capital_series(trades: List[Trade], initial_capital: float) -> List[float]:
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)
    series = [initial_capital]
    for t in sorted_trades:
        series.append(t.capital_after)
    return series


def _max_drawdown(capital_series: List[float]) -> tuple[float, int]:
    if len(capital_series) < 2:
        return 0.0, 0

    peak = capital_series[0]
    max_dd_pct = 0.0
    max_dd_dur = 0
    dd_start_idx = 0

    for i, val in enumerate(capital_series):
        if val > peak:
            peak = val
            dd_start_idx = i
        elif peak > 0:
            dd = (val - peak) / peak * 100.0
            if dd < max_dd_pct:
                max_dd_pct = dd
                max_dd_dur = i - dd_start_idx

    return max_dd_pct, max_dd_dur


def _regime_breakdown(trades: List[Trade]) -> dict:
    """Win rate by trend regime — uses the trade direction as a proxy."""
    buckets: dict = {}
    for t in trades:
        # direction as a proxy for regime context
        key = t.direction
        if key not in buckets:
            buckets[key] = {"total": 0, "wins": 0}
        buckets[key]["total"] += 1
        if t.net_pnl > 0:
            buckets[key]["wins"] += 1

    result = {}
    for key, v in buckets.items():
        win_rate = v["wins"] / v["total"] * 100.0 if v["total"] else 0.0
        result[key] = {"total": v["total"], "win_rate_pct": win_rate}
    return result


def _time_of_day_breakdown(trades: List[Trade]) -> dict:
    """
    Splits trades by entry time into three sessions:
      morning:   09:15–10:59
      midday:    11:00–13:29
      afternoon: 13:30–15:29
    """
    sessions = {
        "morning": {"total": 0, "wins": 0},
        "midday": {"total": 0, "wins": 0},
        "afternoon": {"total": 0, "wins": 0},
    }

    for t in trades:
        minute = t.entry_time.hour * 60 + t.entry_time.minute
        if minute < 660:        # before 11:00
            key = "morning"
        elif minute < 810:      # before 13:30
            key = "midday"
        else:
            key = "afternoon"

        sessions[key]["total"] += 1
        if t.net_pnl > 0:
            sessions[key]["wins"] += 1

    result = {}
    for key, v in sessions.items():
        win_rate = v["wins"] / v["total"] * 100.0 if v["total"] else 0.0
        result[key] = {"total": v["total"], "win_rate_pct": win_rate}
    return result
