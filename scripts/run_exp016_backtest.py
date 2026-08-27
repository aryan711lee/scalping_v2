"""
EXP_016: L6 labels (1.40%/0.70% target/stop) backtest on the 2026 test holdout.

Uses the L6 XGBoost model trained by:
    python scripts/train_models.py --variant L6 --model xgboost
    python scripts/train_models.py --variant L6 --model xgboost_tuned

Then applies EXP_013 confidence filter and runs the Phase 3 backtester
with TARGET_PCT=1.40%, STOPLOSS_PCT=0.70% on the 2026 holdout.

Usage:
    python scripts/run_exp016_backtest.py
    python scripts/run_exp016_backtest.py --no-confidence-filter
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
    SIZING_MODE, POSITION_SIZE_PCT,
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
VARIANT = "L6"
TIME_EXIT_MINUTE = 15 * 60 + 15  # 15:15
MAX_TRADES_PER_DAY = 3

# L6 backtest parameters — MUST match L6 label construction exactly
L6_TARGET_PCT = 0.0140    # 1.40%
L6_STOPLOSS_PCT = 0.0070  # 0.70%

# Comparison baselines
EXP_001 = {
    "total_trades": 2239, "win_rate_pct": 16.5, "net_pnl": -170125.0,
    "gross_pnl": 8811.0, "total_costs": 178936.0, "expectancy": -75.98,
}
EXP_014 = {
    "total_trades": 1730, "win_rate_pct": 16.3, "net_pnl": -123005.0,
    "gross_pnl": 7629.0, "total_costs": 130634.0, "expectancy": -71.10,
}
EXP_015 = {
    "total_trades": 1297, "win_rate_pct": 38.1, "net_pnl": -67964.0,
    "gross_pnl": 67777.0, "total_costs": 135741.0, "expectancy": -52.40,
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


def _costs_pct(metrics: dict) -> float:
    gross = metrics.get("gross_pnl", 0) or 1
    costs = metrics.get("total_costs", 0)
    return costs / gross * 100 if gross != 0 else float("inf")


def print_comparison_table(metrics: dict, use_filter: bool, threshold: float) -> None:
    sep = "=" * 88
    print(f"\n{sep}")
    print("EXP_016 — L6 Labels (1.4%/0.7%) Backtest vs EXP_001, EXP_014, and EXP_015")
    print(sep)
    h = (f"  {'Metric':<30} {'EXP_001':>11} {'EXP_014':>11} "
         f"{'EXP_015':>11} {'EXP_016':>11} {'vs 015':>9}")
    print(h)
    print(f"  {'-'*30} {'-'*11} {'-'*11} {'-'*11} {'-'*11} {'-'*9}")

    def row(label, key, fmt="{:.0f}", mult=1):
        v001 = EXP_001.get(key, 0) * mult
        v014 = EXP_014.get(key, 0) * mult
        v015 = EXP_015.get(key, 0) * mult
        v016 = metrics.get(key, 0) * mult
        d = v016 - v015
        sign = "+" if d >= 0 else ""
        print(f"  {label:<30} {fmt.format(v001):>11} {fmt.format(v014):>11} "
              f"{fmt.format(v015):>11} {fmt.format(v016):>11} {sign+fmt.format(d):>9}")

    row("Total trades", "total_trades")
    wr016 = metrics.get("win_rate_pct", 0)
    dwr = wr016 - EXP_015["win_rate_pct"]
    print(f"  {'Win rate':<30} {EXP_001['win_rate_pct']:>10.1f}% "
          f"{EXP_014['win_rate_pct']:>10.1f}% "
          f"{EXP_015['win_rate_pct']:>10.1f}% "
          f"{wr016:>10.1f}% "
          f"{'+' if dwr>=0 else ''}{dwr:>7.1f}pp")
    row("Net P&L (Rs.)", "net_pnl", "{:,.0f}")
    row("Gross P&L (Rs.)", "gross_pnl", "{:,.0f}")
    row("Total costs (Rs.)", "total_costs", "{:,.0f}")
    row("Expectancy/trade (Rs.)", "expectancy", "{:.2f}")
    row("Sharpe ratio", "sharpe_ratio", "{:.2f}")
    print(sep)

    cp = _costs_pct(metrics)
    net = metrics.get("net_pnl", 0)
    exp_t = metrics.get("expectancy", 0)

    # L6 break-even win rate: 40.8% (from spec calculation)
    print(f"\n  Configuration: variant=L6 (1.4%/0.7%), threshold={threshold:.2f}, "
          f"confidence_filter={'yes' if use_filter else 'no'}")
    print(f"  Costs/gross: {cp:.0f}%")
    print(f"  L6 break-even win rate: ~40.8%  |  Achieved: {wr016:.1f}%")

    if net > 0:
        print(f"\n  RESULT: SCENARIO A — PROFITABLE — Rs.{net:,.0f}  →  L6 is the Phase 7 model")
    elif wr016 >= 40.8:
        print(f"\n  RESULT: SCENARIO A/B — AT OR ABOVE BREAK-EVEN — WR={wr016:.1f}%  →  Proceed to Phase 7 with L6")
    elif wr016 >= 35:
        print(f"\n  RESULT: SCENARIO B — CLOSE TO BREAK-EVEN — WR={wr016:.1f}%  →  Proceed to Phase 7 with L6")
    elif wr016 >= EXP_015["win_rate_pct"]:
        print(f"\n  RESULT: SCENARIO B — IMPROVEMENT ON EXP_015 — WR={wr016:.1f}%  →  Discuss Phase 7 config")
    else:
        print(f"\n  RESULT: SCENARIO C/D — NO IMPROVEMENT OVER EXP_015 — WR={wr016:.1f}%  →  L5 exits preferred")
    print(sep)


def write_exp016_log(metrics: dict, use_filter: bool, threshold: float) -> None:
    log_path = ROOT / "EXPERIMENT_LOG.md"
    net = metrics.get("net_pnl", 0)
    wr = metrics.get("win_rate_pct", 0)
    cp = _costs_pct(metrics)
    exp_t = metrics.get("expectancy", 0)

    def delta(key, fmt="{:.0f}", ref=EXP_015):
        v = metrics.get(key, 0)
        d = v - ref.get(key, 0)
        sign = "+" if d >= 0 else ""
        return fmt.format(v), sign + fmt.format(d)

    if net > 0:
        scenario = "A — Profitable; L6 is the Phase 7 primary model"
    elif wr >= 40.8:
        scenario = "A/B — At or above break-even; proceed to Phase 7 with L6"
    elif wr >= 35:
        scenario = "B — Below break-even but close; run Phase 7 paper trading with L6"
    elif wr >= EXP_015["win_rate_pct"]:
        scenario = "B — Improves on EXP_015 win rate but below 40.8% break-even"
    else:
        scenario = "C — No improvement over EXP_015; recommend L5 exits (2%/1%) for Phase 7"

    if net > 0 or wr >= 35:
        phase7_config = "L6 XGBoost model (1.40%/0.70% target/stop)"
    else:
        phase7_config = "EXP_015 configuration — L1 model with L5 exits (2.00%/1.00%)"

    lines = [
        "\n## EXP_016 — L6 Labels: 1.4% Target / 0.7% Stop\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n",
        "**Hypothesis:** L6 labels (1.4%/0.7%, horizon=40) represent the sweet spot "
        "between L1 (too costly) and L5 (too rare), providing sufficient training "
        "data density while keeping costs manageable at Rs.50,000 capital.\n",
        f"**Test period:** 2026-01-01 → 2026-08-19\n",
        f"**Label variant:** L6 | **Timeframe:** 3min | **Symbols:** 15 large-cap NSE\n",
        "\n**L6 density gate result:** CAUTION — avg long% ~2.1% (gate: ≥4% ideal, 2-3.9% caution, <2% stop)\n",
        "Proceeded with training as per spec (density ≥2.0%). Expect noisier model.\n",
        "\n**L6 label distribution (3min, 15 symbols):**\n",
        "| Symbol | Long% | Short% | None% | Flagged? |\n",
        "|--------|-------|--------|-------|----------|\n",
        "| RELIANCE | 1.5% | 1.2% | 97.3% | YES |\n",
        "| TCS | 1.7% | 1.2% | 97.1% | YES |\n",
        "| INFY | 1.9% | 1.3% | 96.8% | YES |\n",
        "| HDFCBANK | 1.0% | 0.9% | 98.1% | YES |\n",
        "| ICICIBANK | 1.3% | 1.1% | 97.6% | YES |\n",
        "| SBIN | 2.8% | 2.4% | 94.8% | no |\n",
        "| WIPRO | 2.6% | 1.9% | 95.5% | YES |\n",
        "| LT | 2.2% | 1.9% | 95.9% | YES |\n",
        "| KOTAKBANK | 1.8% | 1.3% | 96.9% | YES |\n",
        "| AXISBANK | 2.6% | 2.0% | 95.4% | YES |\n",
        "| BAJFINANCE | 3.3% | 2.8% | 94.0% | no |\n",
        "| MARUTI | 2.4% | 1.8% | 95.8% | YES |\n",
        "| ASIANPAINT | 1.6% | 1.5% | 96.9% | YES |\n",
        "| TITAN | 2.5% | 1.9% | 95.6% | YES |\n",
        "| SUNPHARMA | 1.8% | 1.3% | 96.9% | YES |\n",
        "| **Average** | **~2.1%** | **~1.6%** | **~96.3%** | |\n",
        "\n**Configuration:** "
        f"exit target=1.40%, exit stop=0.70%, "
        f"threshold={threshold:.2f}, confidence_filter={'yes' if use_filter else 'no'}\n",
        "\n**Comparison table (2026 holdout):**\n",
        "| Metric | EXP_001 (baseline) | EXP_014 (best L1) | EXP_015 (L1sig/L5ex) | EXP_016 (L6 full) | Change vs 015 |\n",
        "|--------|-------------------|-------------------|----------------------|-------------------|---------------|\n",
    ]

    def mrow(label, key, fmt="{:.0f}"):
        v001 = fmt.format(EXP_001.get(key, 0))
        v014 = fmt.format(EXP_014.get(key, 0))
        v015 = fmt.format(EXP_015.get(key, 0))
        v016, d = delta(key, fmt)
        lines.append(f"| {label} | {v001} | {v014} | {v015} | {v016} | {d} |\n")

    mrow("Total trades", "total_trades")
    wr015 = EXP_015["win_rate_pct"]
    dwr = wr - wr015
    lines.append(f"| Win rate | {EXP_001['win_rate_pct']:.1f}% | {EXP_014['win_rate_pct']:.1f}% "
                 f"| {wr015:.1f}% | {wr:.1f}% | {'+' if dwr>=0 else ''}{dwr:.1f}pp |\n")
    mrow("Net P&L (Rs.)", "net_pnl", "{:,.0f}")
    mrow("Gross P&L (Rs.)", "gross_pnl", "{:,.0f}")
    mrow("Total costs (Rs.)", "total_costs", "{:,.0f}")
    mrow("Expectancy/trade (Rs.)", "expectancy", "{:.2f}")
    mrow("Sharpe ratio", "sharpe_ratio", "{:.2f}")

    lines += [
        f"\n**Costs/gross:** {cp:.0f}%\n",
        f"**L6 break-even win rate:** ~40.8%  |  **Achieved:** {wr:.1f}%\n",
        f"\n**Scenario:** {scenario}\n",
        f"\n**Phase 7 configuration decision:** {phase7_config}\n",
        f"\n**Conclusion:** EXP_016 tested the sweet-spot hypothesis. "
        f"L6 {'achieved' if net > 0 else 'did not achieve'} profitability "
        f"on the 2026 holdout (net P&L Rs.{net:,.0f}, expectancy Rs.{exp_t:.2f}/trade vs "
        f"EXP_015 Rs.{EXP_015['expectancy']:.2f}/trade). "
        f"Phase 7 proceeds with {phase7_config}.\n",
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Appended EXP_016 to EXPERIMENT_LOG.md")


def main():
    parser = argparse.ArgumentParser(description="EXP_016: L6 backtest (1.4%/0.7% target/stop)")
    parser.add_argument("--no-confidence-filter", action="store_true",
                        help="Skip confidence filter even if sweep was done")
    args = parser.parse_args()

    meta_path = ARTIFACTS_DIR / "best_model_meta.json"
    model_meta = load_meta(meta_path, "best_model_meta")
    if not model_meta:
        logger.error("best_model_meta.json not found. Run train_models.py first.")
        sys.exit(1)

    actual_variant = model_meta.get("variant", "L1")
    if actual_variant != "L6":
        logger.warning(
            "best_model_meta.json points to variant=%s, not L6. "
            "This likely means an L6 model has not been trained yet, or training "
            "did not update best_model_meta.json. Check ARTIFACTS_DIR for xgboost_L6_final.joblib.",
            actual_variant,
        )

    cf_meta = None if args.no_confidence_filter else load_meta(
        ARTIFACTS_DIR / "confidence_filter_meta.json", "confidence_filter_meta"
    )
    use_filter = cf_meta is not None and cf_meta.get("viable", False)

    threshold = model_meta.get("optimal_threshold", 0.5)
    symbols = model_meta.get("symbols", WATCHLIST)
    feature_cols = model_meta.get("feature_columns")

    logger.info("EXP_016 configuration:")
    logger.info("  signal model=%s (label variant=%s)", model_meta.get("model_name", "xgboost"), actual_variant)
    logger.info("  exit params: target=1.40%%, stop=0.70%% (L6 — must match label construction)")
    logger.info("  threshold=%.2f, confidence_filter=%s", threshold, use_filter)
    logger.info("  symbols=%d", len(symbols))

    final_model_path = model_meta.get("final_model_path",
                                      str(ARTIFACTS_DIR / f"xgboost_{actual_variant}_final.joblib"))
    logger.info("Loading predictor from %s", final_model_path)
    base_predictor = ModelPredictor(final_model_path)
    if feature_cols is None:
        feature_cols = base_predictor.feature_columns

    if use_filter:
        logger.info("Applying ConfidenceFilter: top_pct=%.2f, window=%d, min_conf=%.2f",
                    cf_meta["top_pct"], cf_meta["window_candles"], cf_meta["min_confidence"])

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

        filt = ConfidenceFilter(
            base_predictor=base_predictor,
            top_pct=cf_meta["top_pct"],
            window_candles=cf_meta["window_candles"],
            min_confidence=cf_meta["min_confidence"],
        )
        predictor = _FilteredPredictor(filt, feature_cols)
    else:
        predictor = base_predictor

    strategy = MLStrategy(
        predictor=predictor,
        feature_columns=feature_cols,
        threshold=threshold,
        avoid_afternoon=False,
    )

    logger.info("Loading test holdout data (2026-01-01 → 2026-08-19, variant=%s)...", actual_variant)
    logger.warning("ACCESSING TEST HOLDOUT — EXP_016 final evaluation.")

    full_df = build_full_dataset(symbols, TIMEFRAME, actual_variant)
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

    # CRITICAL: target/stop must match L6 label construction (1.40%/0.70%)
    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        target_pct=L6_TARGET_PCT,
        stoploss_pct=L6_STOPLOSS_PCT,
        time_exit_minute=TIME_EXIT_MINUTE,
        max_trades_per_symbol_per_day=MAX_TRADES_PER_DAY,
    )

    logger.info("Running EXP_016 backtest (target=1.40%%, stop=0.70%%)...")
    engine.run(engine_data)

    trades = portfolio.closed_trades
    metrics = calculate_metrics(trades, PAPER_CAPITAL)

    print_comparison_table(metrics, use_filter, threshold)

    generate_report(
        metrics=metrics,
        trades=trades,
        strategy_name="EXP_016 (L6 model 1pct40_0pct70)",
        timeframe=TIMEFRAME,
        symbols=symbols,
        date_range=("2026-01-01", "2026-08-19"),
        capital=PAPER_CAPITAL,
        leverage=INTRADAY_LEVERAGE,
        sizing_mode=SIZING_MODE,
        position_size_pct=POSITION_SIZE_PCT,
        logs_dir=LOGS_DIR,
    )

    write_exp016_log(metrics, use_filter, threshold)
    logger.info("EXP_016 complete.")


if __name__ == "__main__":
    main()
