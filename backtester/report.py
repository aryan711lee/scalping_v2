from __future__ import annotations
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

from backtester.portfolio import Trade


def generate_report(
    metrics: dict,
    trades: List[Trade],
    strategy_name: str,
    timeframe: str,
    symbols: List[str],
    date_range: tuple,
    capital: float,
    leverage: float,
    sizing_mode: str,
    position_size_pct: float,
    logs_dir: Path,
) -> None:
    """Print console report and save text + CSV files to logs_dir."""
    report_text = _build_report_text(
        metrics=metrics,
        strategy_name=strategy_name,
        timeframe=timeframe,
        symbols=symbols,
        date_range=date_range,
        capital=capital,
        leverage=leverage,
        sizing_mode=sizing_mode,
        position_size_pct=position_size_pct,
    )

    print(report_text)

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = logs_dir / f"backtest_{strategy_name}_{timeframe}_{ts}.txt"
    report_path.write_text(report_text, encoding="utf-8")

    trades_path = logs_dir / f"trades_{strategy_name}_{timeframe}_{ts}.csv"
    _save_trades_csv(trades, trades_path)

    print(f"\nReport saved -> {report_path}")
    print(f"Trade log  -> {trades_path}")


def _build_report_text(
    metrics: dict,
    strategy_name: str,
    timeframe: str,
    symbols: List[str],
    date_range: tuple,
    capital: float,
    leverage: float,
    sizing_mode: str,
    position_size_pct: float,
) -> str:
    lines = []
    sep = "=" * 64

    lines.append(sep)
    lines.append(f"BACKTEST REPORT — {strategy_name}")
    lines.append(
        f"Symbols: {len(symbols)} | Timeframe: {timeframe} | "
        f"Period: {date_range[0]} to {date_range[1]}"
    )
    if sizing_mode == "fixed_pct":
        sizing_str = f"fixed_pct {int(position_size_pct * 100)}%"
    else:
        sizing_str = "full_capital"
    lines.append(
        f"Capital: Rs.{capital:,.0f} | Leverage: {leverage:.0f}x | Sizing: {sizing_str}"
    )
    lines.append(sep)

    lines.append("")
    lines.append("SUMMARY")
    lines.append(f"  Total trades:          {metrics['total_trades']:,}")
    lines.append(f"  Win rate:              {metrics['win_rate_pct']:.1f}%")
    lines.append(f"  Net P&L:               Rs.{metrics['net_pnl']:,.2f}")
    lines.append(f"  Net return:            {metrics['net_return_pct']:.1f}%")
    lines.append(f"  Max drawdown:          {metrics['max_drawdown_pct']:.1f}%")
    lines.append(f"  Sharpe ratio:          {metrics['sharpe_ratio']:.2f}")
    pf = metrics['profit_factor']
    lines.append(f"  Profit factor:         {pf:.2f}" if pf is not None else "  Profit factor:         N/A")
    lines.append(f"  Expectancy per trade:  Rs.{metrics['expectancy']:.2f}")

    lines.append("")
    lines.append("COSTS SUMMARY")
    lines.append(f"  Total gross P&L:       Rs.{metrics['gross_pnl']:,.2f}")
    lines.append(f"  Total costs paid:      Rs.{metrics['total_costs']:,.2f}")
    if metrics["gross_pnl"] != 0:
        costs_pct = metrics["total_costs"] / abs(metrics["gross_pnl"]) * 100.0
        lines.append(f"  Costs as % of gross:   {costs_pct:.1f}%")
    else:
        lines.append("  Costs as % of gross:   N/A")

    lines.append("")
    lines.append("EXIT BREAKDOWN")
    total = metrics["total_trades"]
    eb = metrics["exit_breakdown"]
    for reason in ["target", "stoploss", "time_exit", "end_of_data"]:
        count = eb.get(reason, 0)
        pct = count / total * 100.0 if total else 0.0
        label = reason.replace("_", " ").title()
        lines.append(f"  {label:<18} {count:>6,}  ({pct:.1f}%)")

    lines.append("")
    lines.append("REGIME BREAKDOWN (by direction)")
    for direction, stats in metrics["regime_stats"].items():
        lines.append(
            f"  {direction:<8}  trades: {stats['total']:>5,}  win rate: {stats['win_rate_pct']:.1f}%"
        )

    lines.append("")
    lines.append("TIME OF DAY BREAKDOWN")
    tod_labels = {
        "morning": "Morning  (09:15–10:59)",
        "midday": "Midday   (11:00–13:29)",
        "afternoon": "Afternoon(13:30–15:29)",
    }
    for key, label in tod_labels.items():
        stats = metrics["time_of_day_stats"].get(key, {"total": 0, "win_rate_pct": 0.0})
        lines.append(
            f"  {label}  trades: {stats['total']:>5,}  win rate: {stats['win_rate_pct']:.1f}%"
        )

    lines.append("")
    lines.append(f"  Max drawdown duration: {metrics['max_drawdown_duration_days']} trades")
    calmar = metrics.get("calmar_ratio")
    lines.append(f"  Calmar ratio:          {calmar:.2f}" if calmar is not None else "  Calmar ratio:          N/A")
    lines.append(sep)

    return "\n".join(lines)


def _save_trades_csv(trades: List[Trade], path: Path) -> None:
    fieldnames = [
        "symbol", "direction", "entry_time", "entry_price",
        "exit_time", "exit_price", "quantity", "exit_reason",
        "gross_pnl", "costs", "net_pnl", "capital_after",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time.isoformat(),
                "entry_price": round(t.entry_price, 4),
                "exit_time": t.exit_time.isoformat(),
                "exit_price": round(t.exit_price, 4),
                "quantity": t.quantity,
                "exit_reason": t.exit_reason,
                "gross_pnl": round(t.gross_pnl, 4),
                "costs": round(t.costs, 4),
                "net_pnl": round(t.net_pnl, 4),
                "capital_after": round(t.capital_after, 4),
            })
