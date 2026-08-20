"""
Run the best ML model through the Phase 3 backtester on the test holdout.

This script is the ONLY place that loads the 2026 test holdout data.
The final model must be fully selected and saved before running this.

Usage:
  python scripts/run_ml_backtest.py
  python scripts/run_ml_backtest.py --avoid-afternoon true
  python scripts/run_ml_backtest.py --avoid-afternoon false
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    WATCHLIST, LOGS_DIR, PAPER_CAPITAL, INTRADAY_LEVERAGE,
    SIZING_MODE, POSITION_SIZE_PCT, TARGET_PCT, STOPLOSS_PCT,
)
from app.logger import setup_logging
from backtester.engine import BacktestEngine
from backtester.metrics import calculate_metrics
from backtester.portfolio import Portfolio
from backtester.report import generate_report
from datasets.builder import build_full_dataset
from models.predictor import ModelPredictor
from strategy.ml_strategy import MLStrategy
from validation.dataset_splitter import DatasetSplitter
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS

setup_logging()
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
TIMEFRAME = "3min"
VARIANT   = "L1"

# Match EXP_001 config exactly for direct comparison
TIME_EXIT_MINUTE    = 15 * 60 + 15   # 15:15
MAX_TRADES_PER_DAY  = 3


# EXP_001 baseline numbers for comparison table
EXP_001 = {
    "total_trades":     2239,
    "win_rate_pct":     16.5,
    "net_pnl":         -170125.0,
    "gross_pnl":         8811.0,
    "total_costs":     178936.0,
    "expectancy":        -75.98,
    "sharpe_ratio":     -19.34,
    "max_drawdown_pct": -99.7,
}


def load_best_model_meta() -> dict:
    meta_path = ARTIFACTS_DIR / "best_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"best_model_meta.json not found at {meta_path}. "
            "Run scripts/train_models.py first."
        )
    with open(meta_path) as f:
        return json.load(f)


def build_backtester_data(test_df: "pd.DataFrame") -> dict:
    """
    Convert the multi-symbol test DataFrame into the format BacktestEngine expects:
    {symbol: DataFrame with 'datetime', 'open', 'high', 'low', 'close', ...}
    """
    import pandas as pd

    result = {}
    for symbol in test_df["symbol"].unique():
        sym_df = test_df[test_df["symbol"] == symbol].copy()
        sym_df = sym_df.reset_index()  # bring datetime back as column
        sym_df = sym_df.rename(columns={"index": "datetime"})
        if "datetime" not in sym_df.columns:
            sym_df["datetime"] = sym_df.index
        sym_df["datetime"] = pd.to_datetime(sym_df["datetime"])
        # Remove tz for engine compatibility (engine uses .date())
        if sym_df["datetime"].dt.tz is not None:
            sym_df["datetime"] = sym_df["datetime"].dt.tz_localize(None)
        result[symbol] = sym_df

    return result


def print_comparison_table(metrics: dict, avoid_afternoon: bool):
    sep = "=" * 72
    tag = "EXP_008a" if avoid_afternoon else "EXP_008b"
    print(f"\n{sep}")
    print(f"EXP_008 vs EXP_001 Comparison ({tag})")
    print(sep)

    def delta(key, fmt="{:.0f}", multiplier=1):
        new = metrics.get(key, 0) * multiplier
        old = EXP_001.get(key, 0) * multiplier
        d = new - old
        sign = "+" if d >= 0 else ""
        return fmt.format(new), f"{sign}{fmt.format(d)}"

    def row(label, exp001_val, exp008_val, change):
        print(f"  {label:<30} {exp001_val:>16} {exp008_val:>16} {change:>12}")

    print(f"  {'Metric':<30} {'EXP_001 (baseline)':>16} {'EXP_008 (ML)':>16} {'Change':>12}")
    print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*12}")

    v, d = delta("total_trades", "{:.0f}")
    row("Total trades", str(EXP_001["total_trades"]), v, d)

    v_wr = metrics.get("win_rate_pct", 0)
    d_wr = v_wr - EXP_001["win_rate_pct"]
    row("Win rate", f"{EXP_001['win_rate_pct']:.1f}%",
        f"{v_wr:.1f}%", f"{'+' if d_wr>=0 else ''}{d_wr:.1f}pp")

    v, d = delta("net_pnl", "{:,.0f}")
    row("Net P&L (Rs.)", f"Rs.{EXP_001['net_pnl']:,.0f}", f"Rs.{v}", f"Rs.{d}")

    v, d = delta("gross_pnl", "{:,.0f}")
    row("Gross P&L (Rs.)", f"Rs.{EXP_001['gross_pnl']:,.0f}", f"Rs.{v}", f"Rs.{d}")

    v, d = delta("total_costs", "{:,.0f}")
    row("Total costs (Rs.)", f"Rs.{EXP_001['total_costs']:,.0f}", f"Rs.{v}", f"Rs.{d}")

    v, d = delta("expectancy", "{:.2f}")
    row("Expectancy/trade (Rs.)", f"Rs.{EXP_001['expectancy']:.2f}", f"Rs.{v}", f"Rs.{d}")

    v, d = delta("sharpe_ratio", "{:.2f}")
    row("Sharpe ratio", f"{EXP_001['sharpe_ratio']:.2f}", v, d)

    v, d = delta("max_drawdown_pct", "{:.1f}")
    row("Max drawdown", f"{EXP_001['max_drawdown_pct']:.1f}%", f"{v}%", f"{d}pp")

    print(sep)

    net_pnl = metrics.get("net_pnl", 0)
    if net_pnl > 0:
        print(f"  RESULT: PROFITABLE — Net P&L Rs.{net_pnl:,.0f}")
    else:
        exp_per_trade = metrics.get("expectancy", 0)
        imp = exp_per_trade - EXP_001["expectancy"]
        print(f"  RESULT: LOSS-MAKING — Net P&L Rs.{net_pnl:,.0f}")
        print(f"  Expectancy improvement over EXP_001: Rs.{imp:.2f}/trade")
    print(sep)


def write_exp_008_log(metrics: dict, avoid_afternoon: bool, meta: dict):
    log_path = ROOT / "EXPERIMENT_LOG.md"
    tag = "EXP_008a" if avoid_afternoon else "EXP_008b"

    lines = [
        f"\n## EXP_008 — Final ML Backtest ({tag})\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n",
        f"**Model:** {meta.get('model_name')} | **Threshold:** {meta.get('optimal_threshold')}\n",
        f"**avoid_afternoon:** {avoid_afternoon}\n",
        f"**Test period:** 2026-01-01 → 2026-08-19\n",
        "\n**Comparison vs EXP_001:**\n",
        "| Metric | EXP_001 (baseline) | EXP_008 (ML) | Change |\n",
        "|--------|-------------------|--------------|--------|\n",
        f"| Total trades | {EXP_001['total_trades']:,} | {metrics.get('total_trades', 0):,} | "
        f"{metrics.get('total_trades', 0) - EXP_001['total_trades']:+,} |\n",
        f"| Win rate | {EXP_001['win_rate_pct']:.1f}% | {metrics.get('win_rate_pct', 0):.1f}% | "
        f"{metrics.get('win_rate_pct', 0) - EXP_001['win_rate_pct']:+.1f}pp |\n",
        f"| Net P&L | Rs.{EXP_001['net_pnl']:,.0f} | Rs.{metrics.get('net_pnl', 0):,.0f} | "
        f"Rs.{metrics.get('net_pnl', 0) - EXP_001['net_pnl']:+,.0f} |\n",
        f"| Gross P&L | Rs.{EXP_001['gross_pnl']:,.0f} | Rs.{metrics.get('gross_pnl', 0):,.0f} | "
        f"Rs.{metrics.get('gross_pnl', 0) - EXP_001['gross_pnl']:+,.0f} |\n",
        f"| Total costs | Rs.{EXP_001['total_costs']:,.0f} | Rs.{metrics.get('total_costs', 0):,.0f} | "
        f"Rs.{metrics.get('total_costs', 0) - EXP_001['total_costs']:+,.0f} |\n",
        f"| Expectancy/trade | Rs.{EXP_001['expectancy']:.2f} | Rs.{metrics.get('expectancy', 0):.2f} | "
        f"Rs.{metrics.get('expectancy', 0) - EXP_001['expectancy']:+.2f} |\n",
        f"| Sharpe ratio | {EXP_001['sharpe_ratio']:.2f} | {metrics.get('sharpe_ratio', 0):.2f} | "
        f"{metrics.get('sharpe_ratio', 0) - EXP_001['sharpe_ratio']:+.2f} |\n",
        f"| Max drawdown | {EXP_001['max_drawdown_pct']:.1f}% | {metrics.get('max_drawdown_pct', 0):.1f}% | "
        f"{metrics.get('max_drawdown_pct', 0) - EXP_001['max_drawdown_pct']:+.1f}pp |\n",
        "\n**Conclusion:** [to be filled after reviewing results]\n",
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Appended %s to EXPERIMENT_LOG.md", tag)


def main():
    parser = argparse.ArgumentParser(description="Run ML backtest on test holdout")
    parser.add_argument("--avoid-afternoon", type=str, default="true",
                        choices=["true", "false"],
                        help="Filter out afternoon session signals")
    args = parser.parse_args()
    avoid_afternoon = args.avoid_afternoon.lower() == "true"

    logger.info("Loading best model metadata...")
    meta = load_best_model_meta()
    logger.info("  Model: %s, Threshold: %s", meta["model_name"], meta.get("optimal_threshold"))

    logger.info("Loading predictor from %s", meta["final_model_path"])
    predictor = ModelPredictor(meta["final_model_path"])

    feature_cols = meta.get("feature_columns") or predictor.feature_columns

    strategy = MLStrategy(
        predictor=predictor,
        feature_columns=feature_cols,
        threshold=meta.get("optimal_threshold", 0.5),
        avoid_afternoon=avoid_afternoon,
    )

    # Load test holdout data — ONLY accessed here, after final model is selected
    logger.info("Loading test holdout data (2026-01-01 → 2026-08-19)...")
    logger.warning("ACCESSING TEST HOLDOUT — this is the final EXP_008 evaluation.")

    full_df = build_full_dataset(WATCHLIST, TIMEFRAME, VARIANT)
    wf_splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds_splitter = DatasetSplitter(wf_splitter)
    test_df = ds_splitter.get_test_split(full_df)
    logger.info("  Test rows: %d", len(test_df))

    # Convert to backtester-compatible format
    engine_data = build_backtester_data(test_df)
    logger.info("  Symbols in test data: %d", len(engine_data))

    # Run backtest
    portfolio = Portfolio(
        capital=PAPER_CAPITAL,
        leverage=INTRADAY_LEVERAGE,
        sizing_mode=SIZING_MODE,
        position_size_pct=POSITION_SIZE_PCT,
    )

    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        target_pct=TARGET_PCT,
        stoploss_pct=STOPLOSS_PCT,
        time_exit_minute=TIME_EXIT_MINUTE,
        max_trades_per_symbol_per_day=MAX_TRADES_PER_DAY,
    )

    tag = "EXP_008a" if avoid_afternoon else "EXP_008b"
    logger.info("Running %s backtest (avoid_afternoon=%s)...", tag, avoid_afternoon)
    engine.run(engine_data)

    trades = portfolio.closed_trades
    metrics = calculate_metrics(trades, PAPER_CAPITAL)

    print_comparison_table(metrics, avoid_afternoon)

    generate_report(
        metrics=metrics,
        trades=trades,
        strategy_name=tag,
        timeframe=TIMEFRAME,
        symbols=WATCHLIST,
        date_range=("2026-01-01", "2026-08-19"),
        capital=PAPER_CAPITAL,
        leverage=INTRADAY_LEVERAGE,
        sizing_mode=SIZING_MODE,
        position_size_pct=POSITION_SIZE_PCT,
        logs_dir=LOGS_DIR,
    )

    write_exp_008_log(metrics, avoid_afternoon, meta)
    logger.info("%s complete.", tag)


if __name__ == "__main__":
    main()
