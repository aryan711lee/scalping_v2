"""
EXP_014: Best combination backtest on the 2026 test holdout.

Combines the best findings from EXP_009–013:
  - Best label variant (from EXP_009 vs EXP_006 comparison)
  - Best symbol universe (from EXP_010)
  - New features if EXP_011/012 success criteria were met (auto-detected from feature count)
  - ConfidenceFilter if EXP_013 was viable

This script reads best_model_meta.json and confidence_filter_meta.json from
models/artifacts/ to determine the final configuration, then runs the full
Phase 3 backtester on the 2026 holdout and writes the comparison table.

Usage:
    python scripts/run_exp014_backtest.py
    python scripts/run_exp014_backtest.py --no-confidence-filter
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
from models.confidence_filter import ConfidenceFilter
from models.predictor import ModelPredictor
from strategy.ml_strategy import MLStrategy
from validation.dataset_splitter import DatasetSplitter
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS

setup_logging()
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
TIMEFRAME = "3min"
TIME_EXIT_MINUTE = 15 * 60 + 15  # 15:15
MAX_TRADES_PER_DAY = 3

# Comparison baselines
EXP_001 = {
    "total_trades": 2239, "win_rate_pct": 16.5, "net_pnl": -170125.0,
    "gross_pnl": 8811.0, "total_costs": 178936.0, "expectancy": -75.98,
    "sharpe_ratio": -19.34, "max_drawdown_pct": -99.7,
}
EXP_008b = {
    "total_trades": 1906, "win_rate_pct": 18.5, "net_pnl": -133145.0,
    "gross_pnl": 12014.0, "total_costs": 145158.0, "expectancy": -69.86,
    "sharpe_ratio": -17.51, "max_drawdown_pct": -99.7,
}


def load_meta(path: Path, name: str) -> dict | None:
    if not path.exists():
        logger.warning("%s not found at %s — skipping", name, path)
        return None
    with open(path) as f:
        return json.load(f)


def build_backtester_data(test_df) -> dict:
    import pandas as pd
    result = {}
    for symbol in test_df["symbol"].unique():
        sym_df = test_df[test_df["symbol"] == symbol].copy().reset_index()
        sym_df = sym_df.rename(columns={"index": "datetime"})
        if "datetime" not in sym_df.columns:
            sym_df["datetime"] = sym_df.index
        sym_df["datetime"] = pd.to_datetime(sym_df["datetime"])
        if sym_df["datetime"].dt.tz is not None:
            sym_df["datetime"] = sym_df["datetime"].dt.tz_localize(None)
        result[symbol] = sym_df
    return result


def print_comparison_table(metrics: dict, model_meta: dict, use_filter: bool) -> None:
    sep = "=" * 76
    print(f"\n{sep}")
    print("EXP_014 — Best Combination Backtest vs EXP_001 and EXP_008b")
    print(sep)
    h = f"  {'Metric':<30} {'EXP_001':>12} {'EXP_008b':>12} {'EXP_014':>12} {'vs 008b':>10}"
    print(h)
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    def row(label, key, fmt="{:.0f}", mult=1):
        v001 = EXP_001.get(key, 0) * mult
        v008 = EXP_008b.get(key, 0) * mult
        v014 = metrics.get(key, 0) * mult
        d = v014 - v008
        sign = "+" if d >= 0 else ""
        print(f"  {label:<30} {fmt.format(v001):>12} {fmt.format(v008):>12} "
              f"{fmt.format(v014):>12} {sign+fmt.format(d):>10}")

    row("Total trades", "total_trades")
    print(f"  {'Win rate':<30} {EXP_001['win_rate_pct']:>11.1f}% "
          f"{EXP_008b['win_rate_pct']:>11.1f}% "
          f"{metrics.get('win_rate_pct',0):>11.1f}% "
          f"{metrics.get('win_rate_pct',0)-EXP_008b['win_rate_pct']:>+9.1f}pp")
    row("Net P&L (Rs.)", "net_pnl", "{:,.0f}")
    row("Gross P&L (Rs.)", "gross_pnl", "{:,.0f}")
    row("Total costs (Rs.)", "total_costs", "{:,.0f}")
    row("Expectancy/trade (Rs.)", "expectancy", "{:.2f}")
    row("Sharpe ratio", "sharpe_ratio", "{:.2f}")
    row("Max drawdown (%)", "max_drawdown_pct", "{:.1f}")
    print(sep)

    variant = model_meta.get("variant", "L1")
    n_sym = len(model_meta.get("symbols", WATCHLIST))
    print(f"  Configuration: variant={variant}, symbols={n_sym}, "
          f"confidence_filter={'yes' if use_filter else 'no'}")

    net = metrics.get("net_pnl", 0)
    exp = metrics.get("expectancy", 0)
    costs = metrics.get("total_costs", 0)
    gross = metrics.get("gross_pnl", 0) or 1
    costs_pct = costs / gross * 100 if gross != 0 else float("inf")

    print(f"\n  Costs/gross: {costs_pct:.0f}%")

    success = (net > 0) or (exp > -30) or (costs_pct < 500)
    if net > 0:
        print(f"  RESULT: PROFITABLE — Net P&L Rs.{net:,.0f}")
    elif exp > EXP_008b["expectancy"]:
        print(f"  RESULT: IMPROVED — Expectancy Rs.{exp:.2f}/trade "
              f"(vs Rs.{EXP_008b['expectancy']:.2f} in EXP_008b)")
    else:
        print(f"  RESULT: LOSS-MAKING — Net P&L Rs.{net:,.0f}, Expectancy Rs.{exp:.2f}/trade")

    print(f"\n  Success criteria met: {'YES' if success else 'NO'}")
    if not success:
        print("  None of: net_pnl > 0, expectancy > Rs.-30, costs/gross < 500% were achieved.")
        print("  Proceed to Phase 7 paper trading with the best available model.")
    print(sep)


def write_exp014_log(metrics: dict, model_meta: dict, use_filter: bool) -> None:
    log_path = ROOT / "EXPERIMENT_LOG.md"
    variant = model_meta.get("variant", "L1")
    n_sym = len(model_meta.get("symbols", WATCHLIST))
    costs = metrics.get("total_costs", 0)
    gross = metrics.get("gross_pnl", 0) or 1
    costs_pct = costs / gross * 100 if gross != 0 else float("inf")

    def delta(key, fmt="{:.0f}", ref=EXP_008b):
        v = metrics.get(key, 0)
        d = v - ref.get(key, 0)
        sign = "+" if d >= 0 else ""
        return fmt.format(v), sign + fmt.format(d)

    lines = [
        "\n## EXP_014 — Best Combination Backtest\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n",
        f"**Configuration:** variant={variant}, symbols={n_sym}, "
        f"confidence_filter={'yes' if use_filter else 'no'}\n",
        f"**Test period:** 2026-01-01 → 2026-08-19\n",
        "\n**Comparison table:**\n",
        "| Metric | EXP_001 (baseline) | EXP_008b (Phase 6 ML) | EXP_014 (best combo) | Change vs 008b |\n",
        "|--------|-------------------|----------------------|---------------------|----------------|\n",
    ]

    def mrow(label, key, fmt="{:.0f}"):
        v001 = fmt.format(EXP_001.get(key, 0))
        v008 = fmt.format(EXP_008b.get(key, 0))
        v014, d = delta(key, fmt)
        lines.append(f"| {label} | {v001} | {v008} | {v014} | {d} |\n")

    mrow("Total trades", "total_trades")
    win014 = metrics.get("win_rate_pct", 0)
    dwin = win014 - EXP_008b["win_rate_pct"]
    lines.append(f"| Win rate | {EXP_001['win_rate_pct']:.1f}% | {EXP_008b['win_rate_pct']:.1f}% "
                 f"| {win014:.1f}% | {'+' if dwin>=0 else ''}{dwin:.1f}pp |\n")
    mrow("Net P&L (Rs.)", "net_pnl", "{:,.0f}")
    mrow("Gross P&L (Rs.)", "gross_pnl", "{:,.0f}")
    mrow("Total costs (Rs.)", "total_costs", "{:,.0f}")
    mrow("Expectancy/trade (Rs.)", "expectancy", "{:.2f}")
    mrow("Sharpe ratio", "sharpe_ratio", "{:.2f}")

    net = metrics.get("net_pnl", 0)
    exp_t = metrics.get("expectancy", 0)
    success = (net > 0) or (exp_t > -30) or (costs_pct < 500)

    lines += [
        f"\n**Costs/gross:** {costs_pct:.0f}%\n",
        f"\n**Success criteria:** {'PASSED' if success else 'NONE MET'}\n",
        "- Net P&L > 0: " + ("YES" if net > 0 else "no") + "\n",
        f"- Expectancy > Rs.-30: " + ("YES" if exp_t > -30 else "no") + "\n",
        f"- Costs/gross < 500%: " + ("YES" if costs_pct < 500 else "no") + "\n",
        "\n**Conclusion:** [to be filled after reviewing results]\n",
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Appended EXP_014 to EXPERIMENT_LOG.md")


def main():
    parser = argparse.ArgumentParser(description="EXP_014: best combination backtest")
    parser.add_argument("--no-confidence-filter", action="store_true",
                        help="Skip confidence filter even if sweep was done")
    args = parser.parse_args()

    model_meta = load_meta(ARTIFACTS_DIR / "best_model_meta.json", "best_model_meta")
    if not model_meta:
        logger.error("best_model_meta.json not found. Run scripts/train_models.py first.")
        sys.exit(1)

    cf_meta = None if args.no_confidence_filter else load_meta(
        ARTIFACTS_DIR / "confidence_filter_meta.json", "confidence_filter_meta"
    )
    use_filter = cf_meta is not None and cf_meta.get("viable", False)

    variant = model_meta.get("variant", "L1")
    symbols = model_meta.get("symbols", WATCHLIST)
    threshold = model_meta.get("optimal_threshold", 0.5)

    logger.info("EXP_014 configuration:")
    logger.info("  variant=%s, symbols=%d, threshold=%.2f, confidence_filter=%s",
                variant, len(symbols), threshold, use_filter)

    logger.info("Loading predictor from %s", model_meta["final_model_path"])
    base_predictor = ModelPredictor(model_meta["final_model_path"])
    feature_cols = model_meta.get("feature_columns") or base_predictor.feature_columns

    # Build strategy with or without confidence filter
    if use_filter:
        logger.info("Applying ConfidenceFilter: top_pct=%.2f, window=%d, min_conf=%.2f",
                    cf_meta["top_pct"], cf_meta["window_candles"], cf_meta["min_confidence"])
        filt = ConfidenceFilter(
            base_predictor=base_predictor,
            top_pct=cf_meta["top_pct"],
            window_candles=cf_meta["window_candles"],
            min_confidence=cf_meta["min_confidence"],
        )
        # Wrap in MLStrategy-compatible interface via adapter
        class _FilteredPredictor:
            def __init__(self, f, fc):
                self._f = f
                self._fc = fc
            @property
            def feature_columns(self):
                return self._fc
            def predict(self, X, threshold=0.5):
                return self._f.predict(X, threshold=threshold)
            def predict_proba(self, X):
                return self._f.base_predictor.predict_proba(X)

        predictor = _FilteredPredictor(filt, feature_cols)
    else:
        predictor = base_predictor

    strategy = MLStrategy(
        predictor=predictor,
        feature_columns=feature_cols,
        threshold=threshold,
        avoid_afternoon=False,
    )

    logger.info("Loading test holdout data (2026-01-01 → 2026-08-19)...")
    logger.warning("ACCESSING TEST HOLDOUT — EXP_014 final evaluation.")

    full_df = build_full_dataset(symbols, TIMEFRAME, variant)
    wf_splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds_splitter = DatasetSplitter(wf_splitter)
    test_df = ds_splitter.get_test_split(full_df)
    logger.info("  Test rows: %d", len(test_df))

    engine_data = build_backtester_data(test_df)
    logger.info("  Symbols in test data: %d", len(engine_data))

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

    logger.info("Running EXP_014 backtest...")
    engine.run(engine_data)

    trades = portfolio.closed_trades
    metrics = calculate_metrics(trades, PAPER_CAPITAL)

    print_comparison_table(metrics, model_meta, use_filter)

    generate_report(
        metrics=metrics,
        trades=trades,
        strategy_name="EXP_014",
        timeframe=TIMEFRAME,
        symbols=symbols,
        date_range=("2026-01-01", "2026-08-19"),
        capital=PAPER_CAPITAL,
        leverage=INTRADAY_LEVERAGE,
        sizing_mode=SIZING_MODE,
        position_size_pct=POSITION_SIZE_PCT,
        logs_dir=LOGS_DIR,
    )

    # Save best final model for Phase 7
    import shutil
    src = Path(model_meta["final_model_path"])
    dst = ARTIFACTS_DIR / "best_final_model.joblib"
    if src.exists() and src != dst:
        shutil.copy2(src, dst)
        logger.info("Saved best final model to %s", dst)

    write_exp014_log(metrics, model_meta, use_filter)
    logger.info("EXP_014 complete.")


if __name__ == "__main__":
    main()
