"""
Train ML models through walk-forward folds.

Usage:
  python scripts/train_models.py                     # trains all 4 models
  python scripts/train_models.py --model logistic
  python scripts/train_models.py --model rf
  python scripts/train_models.py --model xgboost
  python scripts/train_models.py --model xgboost_tuned

Order: LR → RF → XGBoost → XGBoost+threshold tuning (EXP_004–007).
Final model is trained on all non-test fold data and saved to models/artifacts/.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import WATCHLIST, LOGS_DIR
from app.logger import setup_logging
from datasets.builder import build_full_dataset
from models.trainer import ModelTrainer
from models.predictor import ModelPredictor
from models.selector import ModelSelector
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS
from validation.dataset_splitter import DatasetSplitter
from validation.evaluator import evaluate_predictions, aggregate_fold_results

setup_logging()
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "3min"
VARIANT   = "L1"
FOLDS     = [1, 2, 3]

# Threshold sweep for EXP_007
THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def load_dataset(splitter: DatasetSplitter, symbols: list, variant: str):
    logger.info("Loading full dataset (%d symbols, 3min, %s)...", len(symbols), variant)
    df = build_full_dataset(symbols, TIMEFRAME, variant)
    logger.info("  Total rows: %d", len(df))
    feature_cols = splitter.get_feature_columns(df)
    logger.info("  Feature columns: %d", len(feature_cols))
    return df, feature_cols


def train_and_evaluate(model, model_name: str, df: pd.DataFrame,
                       feature_cols: list, splitter: DatasetSplitter,
                       exp_id: str, variant: str = VARIANT) -> dict:
    """
    Train model through all folds, evaluate on val sets.
    Returns experiment dict with fold_results, agg, and final model path.
    """
    logger.info("=" * 60)
    logger.info("Training %s (%s)", model_name, exp_id)

    # Compute class weights from the full training set of fold 1
    # (the smallest training set — conservative estimate)
    train_fold1, _ = splitter.split(df, fold_id=1)
    class_weights = splitter.get_class_weights(train_fold1)
    logger.info("  Class weights: %s", class_weights)

    trainer = ModelTrainer(
        model=model,
        model_name=model_name,
        variant=variant,
        feature_columns=feature_cols,
        class_weights=class_weights,
    )

    fold_results = []
    for fold_id in FOLDS:
        train_df, val_df = splitter.split(df, fold_id=fold_id)
        logger.info("  Fold %d: train=%d, val=%d", fold_id, len(train_df), len(val_df))

        fold_result = trainer.train_fold(train_df, fold_id, val_df=val_df)
        trainer.save_fold_model(fold_result)

        # Evaluate on validation set
        fitted = fold_result["model"]
        scaler  = fold_result.get("scaler")

        import joblib
        from pathlib import Path as P
        payload = {
            "model": fitted,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "model_name": model_name,
            "variant": variant,
            "is_xgb": "XGB" in type(fitted).__name__,
        }
        tmp_path = ARTIFACTS_DIR / f"_tmp_{model_name}_{fold_id}.joblib"
        joblib.dump(payload, tmp_path)

        predictor = ModelPredictor(str(tmp_path))
        y_pred = predictor.predict(val_df[feature_cols])
        y_true = val_df["label"].values

        metrics = evaluate_predictions(y_pred=y_pred, y_true=y_true,
                                       fold_id=fold_id, model_name=model_name,
                                       variant=variant)
        metrics["train_time_seconds"] = fold_result["train_time_seconds"]
        metrics["train_rows"] = fold_result["train_rows"]

        if "feature_importances" in fold_result:
            metrics["feature_importances"] = fold_result["feature_importances"]

        fold_results.append(metrics)
        logger.info(
            "  Fold %d val: combined_precision=%.1f%%, signal_rate=%.1f%%",
            fold_id,
            (metrics["combined_precision"] or 0) * 100,
            metrics["signal_rate"] * 100,
        )

        tmp_path.unlink(missing_ok=True)

    agg = aggregate_fold_results(fold_results)

    cp = agg.get("combined_precision", {})
    sr = agg.get("signal_rate", {})
    cp_mean = cp.get("mean", 0) or 0
    cp_std  = cp.get("std",  0) or 0
    sr_mean = sr.get("mean", 0) or 0
    consistency = (cp_std / cp_mean * 100) if cp_mean > 0 else float("nan")

    logger.info(
        "%s aggregate: combined_precision=%.1f%%±%.1f%%, signal_rate=%.1f%%, "
        "consistency=%.1f%%",
        exp_id, cp_mean * 100, cp_std * 100, sr_mean * 100, consistency,
    )

    # Train final model on all non-test data (combined folds 1+2+3 train sets)
    # fold 3 train covers 2023-01-02 → 2025-06-30 — the union of all fold trains
    full_train_df, _ = splitter.split(df, fold_id=3)
    logger.info("  Training final model on %d rows (fold 3 train = max non-test)", len(full_train_df))
    trainer.train_final(full_train_df)

    return {
        "experiment_id": exp_id,
        "model_name": model_name,
        "variant": variant,
        "fold_results": fold_results,
        "agg": agg,
        "final_model_path": str(ARTIFACTS_DIR / f"{model_name}_{variant}_final.joblib"),
    }


def threshold_sweep(exp_xgb: dict, df: pd.DataFrame, feature_cols: list,
                    splitter: DatasetSplitter, min_signal_rate: float = 0.02) -> dict:
    """
    EXP_007: sweep thresholds on the best XGBoost model's fold models.
    Returns the optimal threshold and full sweep table.
    """
    logger.info("=" * 60)
    logger.info("EXP_007: Threshold sweep on XGBoost")

    model_name = exp_xgb["model_name"]
    sweep_rows = []

    for t in THRESHOLDS:
        fold_metrics = []
        for fold_id in FOLDS:
            fold_path = ARTIFACTS_DIR / f"{model_name}_{exp_xgb.get('variant', VARIANT)}_fold{fold_id}.joblib"
            if not fold_path.exists():
                logger.warning("Missing fold model: %s", fold_path)
                continue

            predictor = ModelPredictor(str(fold_path))
            _, val_df = splitter.split(df, fold_id=fold_id)

            y_pred = predictor.predict(val_df[feature_cols], threshold=t)
            y_true = val_df["label"].values
            metrics = evaluate_predictions(y_pred=y_pred, y_true=y_true,
                                           fold_id=fold_id,
                                           model_name=model_name,
                                           variant=exp_xgb.get("variant", VARIANT))
            fold_metrics.append(metrics)

        agg = aggregate_fold_results(fold_metrics)
        cp = agg.get("combined_precision", {})
        sr = agg.get("signal_rate", {})
        cp_mean = cp.get("mean", 0) or 0
        cp_std  = cp.get("std",  0) or 0
        sr_mean = sr.get("mean", 0) or 0
        fold_std = cp.get("std", 0) or 0
        viable = cp_mean > 0 and sr_mean >= min_signal_rate

        sweep_rows.append({
            "threshold": t,
            "combined_precision": cp_mean,
            "signal_rate": sr_mean,
            "fold_std": fold_std,
            "viable": viable,
            "fold_metrics": fold_metrics,
            "agg": agg,
        })

        logger.info(
            "  t=%.2f  combined_precision=%.1f%%  signal_rate=%.1f%%  viable=%s",
            t, cp_mean * 100, sr_mean * 100, viable,
        )

    # Print sweep table
    print("\nThreshold Sweep Results (EXP_007):")
    print(f"{'Threshold':>10} {'combined_precision':>20} {'signal_rate':>12} {'fold_std':>10} {'viable':>8}")
    for row in sweep_rows:
        print(
            f"{row['threshold']:>10.2f} "
            f"{row['combined_precision']*100:>19.1f}% "
            f"{row['signal_rate']*100:>11.1f}% "
            f"{row['fold_std']*100:>9.1f}% "
            f"{'yes' if row['viable'] else 'no':>8}"
        )

    # Select: maximise combined_precision subject to min_signal_rate constraint
    viable = [r for r in sweep_rows if r["viable"]]
    if viable:
        best = max(viable, key=lambda r: r["combined_precision"])
    else:
        logger.warning("No threshold achieves signal_rate >= %.1f%%. Selecting best available.",
                       min_signal_rate * 100)
        best = max(sweep_rows, key=lambda r: r["combined_precision"])

    logger.info(
        "EXP_007 selected threshold=%.2f (combined_precision=%.1f%%, signal_rate=%.1f%%)",
        best["threshold"], best["combined_precision"] * 100, best["signal_rate"] * 100,
    )

    return {
        "experiment_id": "EXP_007",
        "model_name": model_name,
        "variant": exp_xgb.get("variant", VARIANT),
        "optimal_threshold": best["threshold"],
        "sweep_rows": sweep_rows,
        "fold_results": best["fold_metrics"],
        "agg": best["agg"],
        "final_model_path": exp_xgb["final_model_path"],
    }


def print_ranking(experiments: list):
    selector = ModelSelector(experiments)
    ranked = selector.rank_models()
    print("\nModel Ranking (EXP_004–007):")
    print(ranked[[
        "experiment_id", "model_name", "combined_precision",
        "signal_rate", "consistency", "composite_score"
    ]].to_string(index=False))
    return selector


def _make_logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=0.1,
        random_state=42,
    )


def _make_rf(class_weights: dict):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        max_features="sqrt",
        class_weight=class_weights,
        n_jobs=-1,
        random_state=42,
    )


def _make_xgb():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    # early_stopping_rounds removed: mlogloss plateau != precision plateau.
    # The model achieves "good" mlogloss by predicting no-trade (65% class) early,
    # then stops before learning directional signals. Full 500 rounds are needed.
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ML models for scalping system")
    parser.add_argument(
        "--model",
        choices=["logistic", "rf", "xgboost", "xgboost_tuned", "all"],
        default="all",
        help="Which model to train",
    )
    parser.add_argument(
        "--variant",
        choices=["L1", "L2", "L3", "L4", "L5"],
        default="L1",
        help="Label variant to train on (default: L1)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated list of symbols to use (default: full WATCHLIST)",
    )
    args = parser.parse_args()

    active_variant = args.variant
    active_symbols = (
        [s.strip() for s in args.symbols.split(",")]
        if args.symbols
        else WATCHLIST
    )

    wf_splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds_splitter = DatasetSplitter(wf_splitter)

    df, feature_cols = load_dataset(ds_splitter, active_symbols, active_variant)

    # Compute class weights from fold 1 train set
    train_fold1, _ = ds_splitter.split(df, fold_id=1)
    class_weights = ds_splitter.get_class_weights(train_fold1)

    experiments = []
    exp_xgb = None

    run_all = args.model == "all"

    # Dynamic experiment IDs based on variant / symbols
    _is_default_config = (active_variant == "L1" and active_symbols == WATCHLIST)
    _exp_prefix = "" if _is_default_config else f"[{active_variant},{len(active_symbols)}sym]"

    # EXP_004: Logistic Regression
    if run_all or args.model == "logistic":
        exp = train_and_evaluate(
            model=_make_logistic(),
            model_name="logistic",
            df=df,
            feature_cols=feature_cols,
            splitter=ds_splitter,
            exp_id="EXP_004",
            variant=active_variant,
        )
        experiments.append(exp)
        _write_exp_log(exp, "EXP_004", "Logistic Regression")

    # EXP_005: Random Forest
    if run_all or args.model == "rf":
        exp = train_and_evaluate(
            model=_make_rf(class_weights),
            model_name="rf",
            df=df,
            feature_cols=feature_cols,
            splitter=ds_splitter,
            exp_id="EXP_005",
            variant=active_variant,
        )
        experiments.append(exp)
        _write_exp_log(exp, "EXP_005", "Random Forest")

    # EXP_006: XGBoost
    if run_all or args.model == "xgboost":
        exp = train_and_evaluate(
            model=_make_xgb(),
            model_name="xgboost",
            df=df,
            feature_cols=feature_cols,
            splitter=ds_splitter,
            exp_id="EXP_006",
            variant=active_variant,
        )
        experiments.append(exp)
        exp_xgb = exp
        _write_exp_log(exp, "EXP_006", "XGBoost", include_importances=True)

    # EXP_007: XGBoost threshold tuning
    if run_all or args.model == "xgboost_tuned":
        if exp_xgb is None:
            # Load previously trained XGBoost fold models
            exp_xgb = {
                "model_name": "xgboost",
                "variant": active_variant,
                "final_model_path": str(ARTIFACTS_DIR / f"xgboost_{active_variant}_final.joblib"),
            }
        # L5 has fewer signals by design — lower minimum acceptable signal rate
        min_sr = 0.010 if active_variant == "L5" else 0.020
        exp_tuned = threshold_sweep(exp_xgb, df, feature_cols, ds_splitter,
                                    min_signal_rate=min_sr)
        experiments.append(exp_tuned)
        _write_exp_log(exp_tuned, "EXP_007", "XGBoost + Threshold Tuning")

    if experiments:
        selector = print_ranking(experiments)
        best = selector.select_best()
        logger.info("Best experiment: %s", best.get("experiment_id"))
        logger.info("Final model path: %s", best.get("final_model_path"))

        # Save best experiment metadata for run_ml_backtest.py
        meta = {
            "experiment_id": best.get("experiment_id"),
            "model_name": best.get("model_name"),
            "variant": best.get("variant", active_variant),
            "final_model_path": best.get("final_model_path"),
            "optimal_threshold": best.get("optimal_threshold", 0.5),
            "feature_columns": feature_cols,
            "symbols": active_symbols,
        }
        meta_path = ARTIFACTS_DIR / "best_model_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved best model metadata to %s", meta_path)

    print(f"\nTraining complete (variant={active_variant}, symbols={len(active_symbols)}). "
          "Check EXPERIMENT_LOG.md for full results.")
    print("Next step: python scripts/run_ml_backtest.py")


def _write_exp_log(exp: dict, exp_id: str, title: str, include_importances: bool = False):
    """Append experiment entry to EXPERIMENT_LOG.md."""
    log_path = ROOT / "EXPERIMENT_LOG.md"

    agg = exp.get("agg", {})
    cp = agg.get("combined_precision", {})
    sr = agg.get("signal_rate", {})
    cp_mean = (cp.get("mean") or 0) * 100
    cp_std  = (cp.get("std")  or 0) * 100
    sr_mean = (sr.get("mean") or 0) * 100
    sr_std  = (sr.get("std")  or 0) * 100
    consistency = (cp.get("std", 0) / cp.get("mean", 1) * 100) if cp.get("mean") else float("nan")

    fold_results = exp.get("fold_results", [])

    lines = [
        f"\n## {exp_id} — {title}\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n",
        f"**Model:** {title}\n",
        f"**Variant:** {exp.get('variant', 'L1')} | **Timeframe:** 3min\n",
        "\n**Per-fold results:**\n",
        "| Fold | combined_precision | signal_rate | train_rows |\n",
        "|------|-------------------|-------------|------------|\n",
    ]

    for fr in fold_results:
        cp_val = (fr.get("combined_precision") or 0) * 100
        sr_val = fr.get("signal_rate", 0) * 100
        rows   = fr.get("train_rows", "?")
        rows_str = f"{rows:,}" if isinstance(rows, int) else str(rows)
        lines.append(f"| {fr.get('fold_id')} | {cp_val:.1f}% | {sr_val:.1f}% | {rows_str} |\n")

    lines += [
        "\n**Aggregate:**\n",
        f"- combined_precision: {cp_mean:.1f}% ± {cp_std:.1f}%\n",
        f"- signal_rate: {sr_mean:.1f}% ± {sr_std:.1f}%\n",
        f"- consistency (std/mean): {consistency:.1f}%\n",
    ]

    if exp_id == "EXP_007":
        lines.append("\n**Threshold sweep:**\n")
        lines.append("| threshold | combined_precision | signal_rate | fold_std | viable |\n")
        lines.append("|-----------|-------------------|-------------|----------|--------|\n")
        for row in exp.get("sweep_rows", []):
            lines.append(
                f"| {row['threshold']:.2f} | {row['combined_precision']*100:.1f}% | "
                f"{row['signal_rate']*100:.1f}% | {row['fold_std']*100:.1f}% | "
                f"{'yes' if row['viable'] else 'no'} |\n"
            )
        lines.append(f"\n**Selected threshold:** {exp.get('optimal_threshold', '?')}\n")

    if include_importances and fold_results:
        # Use fold 3 importances if available
        imp_source = next((fr for fr in fold_results if fr.get("fold_id") == 3), fold_results[-1])
        if "feature_importances" in imp_source:
            sorted_imp = sorted(imp_source["feature_importances"].items(),
                                key=lambda x: x[1], reverse=True)
            lines.append("\n**Feature importances (Fold 3, top 20):**\n")
            for feat, imp in sorted_imp[:20]:
                lines.append(f"- {feat}: {imp:.4f}\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Appended %s to EXPERIMENT_LOG.md", exp_id)


if __name__ == "__main__":
    main()
