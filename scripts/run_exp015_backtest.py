"""
EXP_015: L5 labels (2%/1% target/stop) backtest on the 2026 test holdout.

Uses the L5 XGBoost model trained by:
    python scripts/train_models.py --variant L5 --model xgboost
    python scripts/train_models.py --variant L5 --model xgboost_tuned

Then applies EXP_013 confidence filter and runs the Phase 3 backtester
with TARGET_PCT=2.00%, STOPLOSS_PCT=1.00% on the 2026 holdout.

Usage:
    python scripts/run_exp015_backtest.py
    python scripts/run_exp015_backtest.py --no-confidence-filter
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
VARIANT = "L5"
TIME_EXIT_MINUTE = 15 * 60 + 15  # 15:15
MAX_TRADES_PER_DAY = 3

# L5 backtest parameters — MUST match L5 label construction exactly
L5_TARGET_PCT = 0.0200    # 2.00%
L5_STOPLOSS_PCT = 0.0100  # 1.00%

# Comparison baselines
EXP_001 = {
    "total_trades": 2239, "win_rate_pct": 16.5, "net_pnl": -170125.0,
    "gross_pnl": 8811.0, "total_costs": 178936.0, "expectancy": -75.98,
}
EXP_014 = {
    "total_trades": 1730, "win_rate_pct": 16.3, "net_pnl": -123005.0,
    "gross_pnl": 7629.0, "total_costs": 130634.0, "expectancy": -71.10,
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
    sep = "=" * 80
    print(f"\n{sep}")
    print("EXP_015 — L5 Labels (2%/1%) Backtest vs EXP_001 and EXP_014")
    print(sep)
    h = (f"  {'Metric':<30} {'EXP_001':>12} {'EXP_014':>12} "
         f"{'EXP_015':>12} {'vs 014':>10}")
    print(h)
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    def row(label, key, fmt="{:.0f}", mult=1):
        v001 = EXP_001.get(key, 0) * mult
        v014 = EXP_014.get(key, 0) * mult
        v015 = metrics.get(key, 0) * mult
        d = v015 - v014
        sign = "+" if d >= 0 else ""
        print(f"  {label:<30} {fmt.format(v001):>12} {fmt.format(v014):>12} "
              f"{fmt.format(v015):>12} {sign+fmt.format(d):>10}")

    row("Total trades", "total_trades")
    wr015 = metrics.get("win_rate_pct", 0)
    dwr = wr015 - EXP_014["win_rate_pct"]
    print(f"  {'Win rate':<30} {EXP_001['win_rate_pct']:>11.1f}% "
          f"{EXP_014['win_rate_pct']:>11.1f}% "
          f"{wr015:>11.1f}% "
          f"{'+' if dwr>=0 else ''}{dwr:>8.1f}pp")
    row("Net P&L (Rs.)", "net_pnl", "{:,.0f}")
    row("Gross P&L (Rs.)", "gross_pnl", "{:,.0f}")
    row("Total costs (Rs.)", "total_costs", "{:,.0f}")
    row("Expectancy/trade (Rs.)", "expectancy", "{:.2f}")
    row("Sharpe ratio", "sharpe_ratio", "{:.2f}")
    print(sep)

    cp = _costs_pct(metrics)
    net = metrics.get("net_pnl", 0)
    exp_t = metrics.get("expectancy", 0)

    # Break-even win rate for L5: cost_per_rt / (gross_winner + gross_loser)
    # gross_winner = 2%×leverage×capital×size; gross_loser = 1%×same
    # break-even ≈ 38.5% (from spec)
    print(f"\n  Configuration: variant=L5 (2%/1%), threshold={threshold:.2f}, "
          f"confidence_filter={'yes' if use_filter else 'no'}")
    print(f"  Costs/gross: {cp:.0f}%")
    print(f"  L5 break-even win rate: ~38.5%  |  Achieved: {wr015:.1f}%")

    if net > 0:
        print(f"\n  RESULT: SCENARIO A — PROFITABLE — Rs.{net:,.0f}  →  Proceed to Phase 7 with L5")
    elif wr015 >= 25:
        print(f"\n  RESULT: SCENARIO B — LOSS-REDUCING — WR={wr015:.1f}%, "
              f"Expectancy Rs.{exp_t:.2f}/trade  →  Proceed to Phase 7")
    else:
        print(f"\n  RESULT: SCENARIO C/D — NO IMPROVEMENT — WR={wr015:.1f}%  →  Discuss before Phase 7")
    print(sep)


def write_exp015_log(metrics: dict, use_filter: bool, threshold: float,
                     label_dist: dict | None, signal_variant: str = "L5") -> None:
    log_path = ROOT / "EXPERIMENT_LOG.md"
    net = metrics.get("net_pnl", 0)
    wr = metrics.get("win_rate_pct", 0)
    cp = _costs_pct(metrics)
    exp_t = metrics.get("expectancy", 0)

    def delta(key, fmt="{:.0f}", ref=EXP_014):
        v = metrics.get(key, 0)
        d = v - ref.get(key, 0)
        sign = "+" if d >= 0 else ""
        return fmt.format(v), sign + fmt.format(d)

    if net > 0:
        scenario = "A — Profitable; proceed to Phase 7 with L5 configuration"
    elif wr >= 25:
        scenario = "B — Loss-reducing; proceed to Phase 7, document minimum viable capital"
    elif wr >= 10:
        scenario = "C — No improvement over L1; proceed to Phase 7 with EXP_014 config"
    else:
        scenario = "D — Label density or win rate too low; skip L5, proceed to Phase 7 with EXP_014"

    if net > 0:
        phase7_config = "L5 (2%/1% target/stop)"
    else:
        phase7_config = "EXP_014 (L1 XGBoost + confidence filter)"

    model_note = (
        f"signal_model={signal_variant} (L5 model not trained — Scenario D label density), "
        if signal_variant != "L5"
        else f"signal_model=L5, "
    )
    lines = [
        "\n## EXP_015 — L5 Exit Params (2%/1% Target/Stop) Backtest\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n",
        f"**Note:** L5 labels had insufficient density for training (Scenario D). "
        f"This backtest uses the {signal_variant} signal model with L5 exit parameters "
        f"(target=2.00%, stop=1.00%) to answer: does the existing signal selector "
        f"capture 2% moves when we hold longer?\n",
        f"**Test period:** 2026-01-01 → 2026-08-19\n",
        f"**Configuration:** {model_note}"
        f"exit target=2.00%, exit stop=1.00%, "
        f"threshold={threshold:.2f}, confidence_filter={'yes' if use_filter else 'no'}\n",
    ]

    if label_dist:
        lines += [
            "\n**L5 label distribution (3min, 15 symbols):**\n",
            f"- Long (+1): {label_dist.get('long_pct', 'N/A')}\n",
            f"- Short (-1): {label_dist.get('short_pct', 'N/A')}\n",
            f"- None (0): {label_dist.get('none_pct', 'N/A')}\n",
        ]

    lines += [
        "\n**Comparison table (2026 holdout):**\n",
        "| Metric | EXP_001 (baseline) | EXP_014 (best L1) | EXP_015 (L5 2%/1%) | Change vs 014 |\n",
        "|--------|-------------------|-------------------|---------------------|---------------|\n",
    ]

    def mrow(label, key, fmt="{:.0f}"):
        v001 = fmt.format(EXP_001.get(key, 0))
        v014 = fmt.format(EXP_014.get(key, 0))
        v015, d = delta(key, fmt)
        lines.append(f"| {label} | {v001} | {v014} | {v015} | {d} |\n")

    mrow("Total trades", "total_trades")
    wr014 = EXP_014["win_rate_pct"]
    dwr = wr - wr014
    lines.append(f"| Win rate | {EXP_001['win_rate_pct']:.1f}% | {wr014:.1f}% "
                 f"| {wr:.1f}% | {'+' if dwr>=0 else ''}{dwr:.1f}pp |\n")
    mrow("Net P&L (Rs.)", "net_pnl", "{:,.0f}")
    mrow("Gross P&L (Rs.)", "gross_pnl", "{:,.0f}")
    mrow("Total costs (Rs.)", "total_costs", "{:,.0f}")
    mrow("Expectancy/trade (Rs.)", "expectancy", "{:.2f}")
    mrow("Sharpe ratio", "sharpe_ratio", "{:.2f}")

    lines += [
        f"\n**Costs/gross:** {cp:.0f}%\n",
        f"**L5 break-even win rate:** ~38.5%  |  **Achieved:** {wr:.1f}%\n",
        f"\n**Scenario:** {scenario}\n",
        f"\n**Phase 7 configuration decision:** {phase7_config}\n",
        "\n**Conclusion:** EXP_015 determined which target/stop width is viable "
        "at Rs.50,000 capital. "
        f"L5 {'improved' if net > 0 or wr >= 25 else 'did not improve'} on EXP_014 "
        f"(expectancy Rs.{exp_t:.2f}/trade vs Rs.{EXP_014['expectancy']:.2f}/trade).\n",
    ]

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Appended EXP_015 to EXPERIMENT_LOG.md")


def main():
    parser = argparse.ArgumentParser(description="EXP_015: L5 backtest (2%/1% target/stop)")
    parser.add_argument("--no-confidence-filter", action="store_true",
                        help="Skip confidence filter even if sweep was done")
    args = parser.parse_args()

    # Load model metadata — prefer L5 model; fall back to best available (L1 from EXP_014)
    # EXP_015 Scenario D: no L5 model trained (label density too low).
    # Running with L1 model + L5 exit params answers: "does the L1 signal selector
    # capture 2% moves if we hold longer?" This is still a valid experiment.
    l5_meta_path = ARTIFACTS_DIR / "xgboost_L5_final.joblib"
    meta_path = ARTIFACTS_DIR / "best_model_meta.json"
    model_meta = load_meta(meta_path, "best_model_meta")
    if not model_meta:
        logger.error("best_model_meta.json not found. Run train_models.py first.")
        sys.exit(1)

    actual_variant = model_meta.get("variant", "L1")
    if actual_variant != "L5":
        logger.warning(
            "No L5 model found (Scenario D — label density too low). "
            "Using %s model with L5 exit params (target=2%%, stop=1%%). "
            "This tests whether the existing signal selector captures 2%% moves.",
            actual_variant,
        )

    cf_meta = None if args.no_confidence_filter else load_meta(
        ARTIFACTS_DIR / "confidence_filter_meta.json", "confidence_filter_meta"
    )
    use_filter = cf_meta is not None and cf_meta.get("viable", False)

    threshold = model_meta.get("optimal_threshold", 0.5)
    symbols = model_meta.get("symbols", WATCHLIST)
    feature_cols = model_meta.get("feature_columns")

    logger.info("EXP_015 configuration:")
    logger.info("  signal model=%s (label variant=%s)", model_meta.get("model_name", "xgboost"), actual_variant)
    logger.info("  exit params: target=2.00%%, stop=1.00%% (L5 — overrides model training params)")
    logger.info("  threshold=%.2f, confidence_filter=%s", threshold, use_filter)
    logger.info("  symbols=%d", len(symbols))

    # Load predictor — use best available model (L5 if trained, else L1 from EXP_014)
    final_model_path = model_meta.get("final_model_path",
                                      str(ARTIFACTS_DIR / f"xgboost_{actual_variant}_final.joblib"))
    logger.info("Loading L5 predictor from %s", final_model_path)
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

    # Load 2026 test holdout — use the model's own label variant for the dataset
    logger.info("Loading test holdout data (2026-01-01 → 2026-08-19, variant=%s)...", actual_variant)
    logger.warning("ACCESSING TEST HOLDOUT — EXP_015 final evaluation.")

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

    # CRITICAL: target/stop must match L5 label construction (2%/1%), not .env defaults
    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        target_pct=L5_TARGET_PCT,
        stoploss_pct=L5_STOPLOSS_PCT,
        time_exit_minute=TIME_EXIT_MINUTE,
        max_trades_per_symbol_per_day=MAX_TRADES_PER_DAY,
    )

    logger.info("Running EXP_015 backtest (target=2.00%%, stop=1.00%%)...")
    engine.run(engine_data)

    trades = portfolio.closed_trades
    metrics = calculate_metrics(trades, PAPER_CAPITAL)

    print_comparison_table(metrics, use_filter, threshold)

    strategy_label = (
        f"EXP_015 (L5 exits 2pct_1pct, {actual_variant} signals)"
        if actual_variant != "L5"
        else "EXP_015 (L5 model 2pct_1pct)"
    )
    generate_report(
        metrics=metrics,
        trades=trades,
        strategy_name=strategy_label,
        timeframe=TIMEFRAME,
        symbols=symbols,
        date_range=("2026-01-01", "2026-08-19"),
        capital=PAPER_CAPITAL,
        leverage=INTRADAY_LEVERAGE,
        sizing_mode=SIZING_MODE,
        position_size_pct=POSITION_SIZE_PCT,
        logs_dir=LOGS_DIR,
    )

    write_exp015_log(metrics, use_filter, threshold, label_dist=None,
                     signal_variant=actual_variant)
    logger.info("EXP_015 complete.")


if __name__ == "__main__":
    main()
