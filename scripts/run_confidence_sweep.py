"""
EXP_013: Confidence-weighted signal filtering — parameter sweep.

Loads the trained XGBoost fold models and sweeps ConfidenceFilter hyperparameters
to find the combination that maximises combined_precision at signal_rate >= 1.5%.

Usage:
    python scripts/run_confidence_sweep.py
    python scripts/run_confidence_sweep.py --variant L1
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import WATCHLIST
from app.logger import setup_logging
from datasets.builder import build_full_dataset
from models.confidence_filter import ConfidenceFilter
from models.predictor import ModelPredictor
from validation.dataset_splitter import DatasetSplitter
from validation.evaluator import evaluate_predictions, aggregate_fold_results
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS

setup_logging()
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
TIMEFRAME = "3min"
FOLDS = [1, 2, 3]

# EXP_013 sweep space
TOP_PCTS = [0.10, 0.15, 0.20, 0.25, 0.30]
WINDOW_CANDLES = [50, 100, 200]
MIN_CONFIDENCES = [0.35, 0.40, 0.45]


def run_sweep(variant: str, symbols: list) -> list[dict]:
    logger.info("Loading dataset (%d symbols, %s, %s)...", len(symbols), TIMEFRAME, variant)
    df = build_full_dataset(symbols, TIMEFRAME, variant)

    wf_splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds_splitter = DatasetSplitter(wf_splitter)
    feature_cols = ds_splitter.get_feature_columns(df)

    model_name = "xgboost"
    all_sweep_rows = []

    for tp in TOP_PCTS:
        for wc in WINDOW_CANDLES:
            for mc in MIN_CONFIDENCES:
                fold_metrics = []
                for fold_id in FOLDS:
                    fold_path = ARTIFACTS_DIR / f"{model_name}_{variant}_fold{fold_id}.joblib"
                    if not fold_path.exists():
                        logger.warning("Missing fold model: %s", fold_path)
                        continue

                    base_predictor = ModelPredictor(str(fold_path))
                    filt = ConfidenceFilter(
                        base_predictor=base_predictor,
                        top_pct=tp,
                        window_candles=wc,
                        min_confidence=mc,
                    )

                    _, val_df = ds_splitter.split(df, fold_id=fold_id)
                    y_pred = filt.predict(val_df[feature_cols])
                    y_true = val_df["label"].values

                    metrics = evaluate_predictions(
                        y_pred=y_pred, y_true=y_true,
                        fold_id=fold_id, model_name="confidence_filter",
                        variant=variant,
                    )
                    fold_metrics.append(metrics)

                if not fold_metrics:
                    continue

                agg = aggregate_fold_results(fold_metrics)
                cp = agg.get("combined_precision", {})
                sr = agg.get("signal_rate", {})
                cp_mean = (cp.get("mean") or 0.0)
                sr_mean = (sr.get("mean") or 0.0)
                viable = sr_mean >= 0.015

                row = {
                    "top_pct": tp,
                    "window_candles": wc,
                    "min_confidence": mc,
                    "combined_precision": cp_mean,
                    "signal_rate": sr_mean,
                    "viable": viable,
                    "fold_metrics": fold_metrics,
                    "agg": agg,
                }
                all_sweep_rows.append(row)
                logger.info(
                    "  tp=%.2f wc=%d mc=%.2f → cp=%.1f%% sr=%.1f%% viable=%s",
                    tp, wc, mc, cp_mean * 100, sr_mean * 100, viable,
                )

    return all_sweep_rows


def print_sweep_table(rows: list[dict]) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("EXP_013: Confidence Filter Sweep Results")
    print(f"{'top_pct':>8} {'window':>8} {'min_conf':>10} {'combined_prec':>15} {'signal_rate':>13} {'viable':>8}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: (x["viable"], x["combined_precision"]), reverse=True)[:30]:
        print(
            f"{r['top_pct']:>8.2f} {r['window_candles']:>8} {r['min_confidence']:>10.2f} "
            f"{r['combined_precision']*100:>14.1f}% {r['signal_rate']*100:>12.1f}% "
            f"{'yes' if r['viable'] else 'no':>8}"
        )
    print(sep)


def write_exp013_log(best: dict, variant: str, sweep_rows: list[dict]) -> None:
    log_path = ROOT / "EXPERIMENT_LOG.md"
    cp = best["combined_precision"] * 100
    sr = best["signal_rate"] * 100

    lines = [
        f"\n## EXP_013 — Confidence-Weighted Signal Filtering\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n",
        f"**Variant:** {variant} | **Timeframe:** 3min | **Base model:** XGBoost\n",
        f"\n**Best configuration:**\n",
        f"- top_pct: {best['top_pct']:.2f}\n",
        f"- window_candles: {best['window_candles']}\n",
        f"- min_confidence: {best['min_confidence']:.2f}\n",
        f"- combined_precision: {cp:.1f}%\n",
        f"- signal_rate: {sr:.1f}%\n",
        f"- viable (signal_rate >= 1.5%): {'yes' if best['viable'] else 'no'}\n",
        f"\n**Sweep table (top 15 by combined_precision, viable only):**\n",
        "| top_pct | window | min_conf | combined_prec | signal_rate | viable |\n",
        "|---------|--------|----------|---------------|-------------|--------|\n",
    ]

    viable_rows = [r for r in sweep_rows if r["viable"]]
    for r in sorted(viable_rows, key=lambda x: x["combined_precision"], reverse=True)[:15]:
        lines.append(
            f"| {r['top_pct']:.2f} | {r['window_candles']} | {r['min_confidence']:.2f} | "
            f"{r['combined_precision']*100:.1f}% | {r['signal_rate']*100:.1f}% | yes |\n"
        )

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Appended EXP_013 to EXPERIMENT_LOG.md")


def main():
    parser = argparse.ArgumentParser(description="EXP_013: confidence filter sweep")
    parser.add_argument("--variant", default="L1", choices=["L1", "L2", "L3", "L4"])
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated symbols (default: full WATCHLIST)")
    args = parser.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",")]
        if args.symbols else WATCHLIST
    )

    sweep_rows = run_sweep(args.variant, symbols)

    if not sweep_rows:
        logger.error("No sweep results — check that XGBoost fold models exist.")
        sys.exit(1)

    print_sweep_table(sweep_rows)

    viable = [r for r in sweep_rows if r["viable"]]
    if viable:
        best = max(viable, key=lambda r: r["combined_precision"])
        logger.info(
            "EXP_013 best: top_pct=%.2f wc=%d mc=%.2f → cp=%.1f%% sr=%.1f%%",
            best["top_pct"], best["window_candles"], best["min_confidence"],
            best["combined_precision"] * 100, best["signal_rate"] * 100,
        )
    else:
        logger.warning("No viable configuration found (signal_rate >= 1.5%). "
                       "Using best by combined_precision regardless.")
        best = max(sweep_rows, key=lambda r: r["combined_precision"])

    # Save best filter config for EXP_014
    meta_path = ARTIFACTS_DIR / "confidence_filter_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "top_pct": best["top_pct"],
            "window_candles": best["window_candles"],
            "min_confidence": best["min_confidence"],
            "combined_precision": best["combined_precision"],
            "signal_rate": best["signal_rate"],
            "viable": best["viable"],
            "variant": args.variant,
        }, f, indent=2)
    logger.info("Saved confidence filter config to %s", meta_path)

    write_exp013_log(best, args.variant, sweep_rows)
    logger.info("EXP_013 complete.")


if __name__ == "__main__":
    main()
