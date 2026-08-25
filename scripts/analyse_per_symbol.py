"""
EXP_010 Step 1: Per-symbol breakdown of EXP_008b backtest results.

Loads the most recent ML backtest trade log from logs/ and computes per-symbol
metrics to identify the top 5 symbols for the focused-universe experiment.

Usage:
    python scripts/analyse_per_symbol.py
    python scripts/analyse_per_symbol.py --log logs/EXP_008b_trades.csv
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

LOGS_DIR = ROOT / "logs"
MIN_TRADES = 50  # minimum trades per symbol for reliable statistics


def find_trade_log(log_arg: str | None) -> Path:
    """Find the EXP_008b trade log, or the most recent ML trade CSV."""
    if log_arg:
        p = Path(log_arg)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
        raise FileNotFoundError(f"Trade log not found: {p}")

    # Try known EXP_008b filenames first
    candidates = sorted(LOGS_DIR.glob("*EXP_008b*trades*.csv")) + \
                 sorted(LOGS_DIR.glob("*ML*trades*.csv")) + \
                 sorted(LOGS_DIR.glob("*trades*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No trade CSV found in {LOGS_DIR}. "
            "Run scripts/run_ml_backtest.py first."
        )

    # Use most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def compute_per_symbol_metrics(df) -> list[dict]:
    """Compute per-symbol backtest metrics from a trade DataFrame."""
    import pandas as pd
    import numpy as np

    results = []
    for symbol, grp in df.groupby("symbol"):
        n = len(grp)
        wins = grp[grp["pnl"] > 0]
        win_rate = len(wins) / n if n > 0 else 0.0

        gross_pnl = grp["gross_pnl"].sum() if "gross_pnl" in grp.columns else grp["pnl"].sum()
        costs = grp["costs"].sum() if "costs" in grp.columns else 0.0
        net_pnl = grp["pnl"].sum()
        expectancy = net_pnl / n if n > 0 else 0.0

        results.append({
            "symbol": symbol,
            "trades": n,
            "win_rate_pct": round(win_rate * 100, 1),
            "net_pnl": round(net_pnl, 0),
            "gross_pnl": round(gross_pnl, 0),
            "costs_paid": round(costs, 0),
            "expectancy_per_trade": round(expectancy, 2),
        })

    return sorted(results, key=lambda r: r["expectancy_per_trade"], reverse=True)


def print_table(rows: list[dict], title: str) -> None:
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    fmt = (
        f"  {'Symbol':<22} {'Trades':>7} {'Win%':>6} "
        f"{'Net P&L':>10} {'Gross P&L':>10} {'Costs':>10} {'Exp/trade':>10}"
    )
    print(fmt)
    print("-" * 90)
    for r in rows:
        flag = " (*)" if r["trades"] < MIN_TRADES else ""
        print(
            f"  {r['symbol']:<22} {r['trades']:>7} {r['win_rate_pct']:>5.1f}% "
            f"  Rs.{r['net_pnl']:>7,.0f} Rs.{r['gross_pnl']:>7,.0f} "
            f"Rs.{r['costs_paid']:>7,.0f} Rs.{r['expectancy_per_trade']:>8.2f}{flag}"
        )
    print(sep)
    print(f"  (*) fewer than {MIN_TRADES} trades — excluded from top-5 selection")
    print(sep)


def select_top5(rows: list[dict]) -> list[str]:
    """
    Select top 5 symbols by expectancy_per_trade, requiring >= MIN_TRADES.
    Falls back to top 3 if fewer than 5 meet the criteria.
    """
    eligible = [r for r in rows if r["trades"] >= MIN_TRADES]
    top = eligible[:5] if len(eligible) >= 5 else eligible[:3]
    return [r["symbol"] for r in top]


def main():
    parser = argparse.ArgumentParser(description="Per-symbol EXP_008b analysis for EXP_010")
    parser.add_argument("--log", default=None, help="Path to trade log CSV")
    args = parser.parse_args()

    try:
        log_path = find_trade_log(args.log)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info("Loading trade log: %s", log_path)

    import pandas as pd
    trades_df = pd.read_csv(log_path)
    logger.info("  Loaded %d trades", len(trades_df))

    # Normalise column names (different report versions may use different names)
    trades_df.columns = [c.lower().strip() for c in trades_df.columns]
    if "pnl" not in trades_df.columns and "net_pnl" in trades_df.columns:
        trades_df["pnl"] = trades_df["net_pnl"]
    if "gross_pnl" not in trades_df.columns:
        trades_df["gross_pnl"] = trades_df.get("pnl", 0) + trades_df.get("costs", 0)

    if "symbol" not in trades_df.columns:
        logger.error("Trade log has no 'symbol' column. Columns: %s", list(trades_df.columns))
        sys.exit(1)

    rows = compute_per_symbol_metrics(trades_df)
    print_table(rows, "EXP_010: Per-symbol breakdown (ranked by expectancy/trade)")

    top5 = select_top5(rows)
    print(f"\nSelected top symbols for EXP_010 focused universe (>={MIN_TRADES} trades):")
    for i, sym in enumerate(top5, 1):
        print(f"  {i}. {sym}")

    if top5:
        symbols_arg = ",".join(top5)
        print(f"\nTo retrain on focused universe:")
        print(
            f"  python scripts/train_models.py --model xgboost --variant L1 "
            f"--symbols \"{symbols_arg}\""
        )

    return top5


if __name__ == "__main__":
    main()
