"""
Phase 5 build script — walk-forward fold construction and baseline evaluation.

Steps:
1. Instantiate WalkForwardSplitter with hardcoded fold dates
2. Load full dataset (all 15 symbols, 3min, L1)
3. For each fold:
   a. Split into train/validate
   b. Run leakage_checker (mandatory — stops on any violation)
   c. Compute class distribution and weights
   d. Evaluate baseline strategy through fold evaluator
   e. Save fold index Parquets to logs/
4. Access test holdout only to count rows and compute class distribution
5. Print fold summary table and evaluation report
6. Save fold_report.txt and fold_metadata.json to logs/
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- path setup ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import WATCHLIST, LOGS_DIR
from datasets.builder import build_full_dataset
from strategy.baseline import BaselineStrategy
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS
from validation.dataset_splitter import DatasetSplitter
from validation.evaluator import evaluate_predictions, aggregate_fold_results, format_evaluation_report
from validation.leakage_checker import check_fold_leakage, check_test_isolation, check_feature_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TIMEFRAME = "3min"
VARIANT   = "L1"


def _class_dist(series: pd.Series) -> dict:
    vc = series.value_counts(normalize=True)
    return {
        str(k): round(float(vc.get(k, 0.0)), 4)
        for k in [-1, 0, 1]
    }


def _baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """Run baseline strategy rules on each row and return signal array."""
    strat = BaselineStrategy()
    preds = []
    for _, row in df.iterrows():
        preds.append(strat.generate_signal(row))
    return np.array(preds)


def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 5 — Walk-Forward Fold Construction")
    logger.info("=" * 60)

    # 1. Splitter
    splitter = WalkForwardSplitter(FOLD_DEFINITIONS)
    logger.info("\n" + splitter.summary())

    # 2. Load full dataset
    logger.info(f"\nLoading full dataset: {len(WATCHLIST)} symbols, {TIMEFRAME}, {VARIANT} ...")
    full_df = build_full_dataset(WATCHLIST, TIMEFRAME, VARIANT)
    logger.info(f"Full dataset: {len(full_df):,} rows, {full_df.index.min()} -> {full_df.index.max()}")

    # 3. Feature leakage check on full dataset
    dataset_splitter = DatasetSplitter(splitter)
    suspicious_cols = check_feature_leakage(full_df)
    if suspicious_cols:
        logger.error(f"FEATURE LEAKAGE DETECTED: {suspicious_cols}")
        sys.exit(1)
    logger.info("Feature leakage check: CLEAN")

    feature_cols = dataset_splitter.get_feature_columns(full_df)
    logger.info(f"Feature columns: {len(feature_cols)} (6 raw price-level features dropped)")

    # Accumulate train+val data for test isolation check later
    all_non_test_dfs = []

    fold_metadata_list = []
    all_fold_results   = []  # baseline predictions per fold

    # Fold summary table rows
    table_rows = []

    # 4. Per-fold processing
    for fold_id in [1, 2, 3]:
        fold_def = splitter.get_fold(fold_id)
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold_id}: Train {fold_def['train_start']}->{fold_def['train_end']} | "
                    f"Val {fold_def['validate_start']}->{fold_def['validate_end']}")
        logger.info("="*50)

        train_df, val_df = dataset_splitter.split(full_df, fold_id)
        logger.info(f"  train rows: {len(train_df):,}  |  val rows: {len(val_df):,}")

        # 4b. Leakage check — mandatory
        try:
            check_fold_leakage(train_df, val_df, fold_id)
            logger.info(f"  Fold {fold_id} leakage check: CLEAN")
        except ValueError as exc:
            logger.error(f"LEAKAGE DETECTED in fold {fold_id}: {exc}")
            sys.exit(1)

        all_non_test_dfs.append(train_df)
        all_non_test_dfs.append(val_df)

        # 4c. Class distribution
        train_dist = _class_dist(train_df["label"])
        val_dist   = _class_dist(val_df["label"])
        logger.info(f"  Train dist — long: {train_dist['1']*100:.1f}%  short: {train_dist['-1']*100:.1f}%  none: {train_dist['0']*100:.1f}%")
        logger.info(f"  Val   dist — long: {val_dist['1']*100:.1f}%  short: {val_dist['-1']*100:.1f}%  none: {val_dist['0']*100:.1f}%")

        # 4d. Class weights
        weights = dataset_splitter.get_class_weights(train_df)
        logger.info(f"  Class weights — long: {weights[1]:.3f}  short: {weights[-1]:.3f}  none: {weights[0]:.3f}")

        # 4e. Baseline evaluation on validation set
        logger.info(f"  Running baseline strategy on val set ({len(val_df):,} rows) ...")
        y_true = val_df["label"].values
        y_pred = _baseline_predict(val_df)

        fold_result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            fold_id=fold_id,
            model_name="Baseline_EXP001",
            variant=VARIANT,
        )
        all_fold_results.append(fold_result)

        cp = fold_result["combined_precision"]
        sr = fold_result["signal_rate"]
        logger.info(f"  Baseline combined_precision: {cp*100:.1f}%  signal_rate: {sr*100:.1f}%")

        # Save fold index Parquets
        train_idx_df = pd.DataFrame({"datetime": train_df.index})
        val_idx_df   = pd.DataFrame({"datetime": val_df.index})
        train_idx_path = LOGS_DIR / f"fold_{fold_id}_train_index.parquet"
        val_idx_path   = LOGS_DIR / f"fold_{fold_id}_val_index.parquet"
        train_idx_df.to_parquet(train_idx_path, index=False)
        val_idx_df.to_parquet(val_idx_path, index=False)
        logger.info(f"  Saved fold index parquets to logs/")

        # Metadata for this fold
        fold_meta = {
            "fold_id":            fold_id,
            "train_start":        fold_def["train_start"],
            "train_end":          fold_def["train_end"],
            "validate_start":     fold_def["validate_start"],
            "validate_end":       fold_def["validate_end"],
            "train_rows":         int(len(train_df)),
            "validate_rows":      int(len(val_df)),
            "train_class_dist":   train_dist,
            "validate_class_dist": val_dist,
            "class_weights":      {str(k): round(v, 4) for k, v in weights.items()},
        }
        fold_metadata_list.append(fold_meta)

        # Table row
        table_rows.append({
            "fold_id":      fold_id,
            "train_period": f"{fold_def['train_start']} → {fold_def['train_end']}",
            "train_rows":   len(train_df),
            "val_period":   f"{fold_def['validate_start']} → {fold_def['validate_end']}",
            "val_rows":     len(val_df),
            "long_pct":     val_dist["1"] * 100,
            "short_pct":    val_dist["-1"] * 100,
            "none_pct":     val_dist["0"] * 100,
        })

    # 5. Test holdout — ONLY row count + class distribution
    logger.info("\n" + "="*50)
    logger.info("TEST HOLDOUT (loading for metadata only — no model sees these labels)")
    logger.info("="*50)
    test_df = dataset_splitter.get_test_split(full_df)
    logger.info(f"  Test rows: {len(test_df):,}")

    # Verify test isolation
    combined_non_test = pd.concat(all_non_test_dfs, axis=0)
    try:
        check_test_isolation(combined_non_test, test_df)
        logger.info("  Test isolation check: CLEAN")
    except ValueError as exc:
        logger.error(f"TEST ISOLATION VIOLATED: {exc}")
        sys.exit(1)

    test_dist = _class_dist(test_df["label"])
    logger.info(f"  Test dist — long: {test_dist['1']*100:.1f}%  short: {test_dist['-1']*100:.1f}%  none: {test_dist['0']*100:.1f}%")

    # Save test index Parquet
    test_idx_df = pd.DataFrame({"datetime": test_df.index})
    test_idx_path = LOGS_DIR / "test_index.parquet"
    test_idx_df.to_parquet(test_idx_path, index=False)
    logger.info("  Saved test_index.parquet to logs/")

    tp = splitter.get_test_period()
    table_rows.append({
        "fold_id":      "TEST",
        "train_period": "[HOLDOUT]",
        "train_rows":   None,
        "val_period":   f"{tp['test_start']} → {tp['test_end']}",
        "val_rows":     len(test_df),
        "long_pct":     test_dist["1"] * 100,
        "short_pct":    test_dist["-1"] * 100,
        "none_pct":     test_dist["0"] * 100,
    })

    # 6. Fold summary table
    logger.info("\n")
    logger.info("FOLD SUMMARY TABLE")
    header = f"{'Fold':<6}  {'Train Period':<27}  {'Train Rows':>11}  {'Val Period':<27}  {'Val Rows':>10}  {'Long%':>6}  {'Short%':>7}  {'None%':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    summary_lines = [header, "-" * len(header)]
    for r in table_rows:
        tr_str = f"{r['train_rows']:>11,}" if r["train_rows"] is not None else f"{'—':>11s}"
        row_str = (
            f"{str(r['fold_id']):<6}  {r['train_period']:<27}  {tr_str}  "
            f"{r['val_period']:<27}  {r['val_rows']:>10,}  "
            f"{r['long_pct']:>6.1f}%  {r['short_pct']:>7.1f}%  {r['none_pct']:>7.1f}%"
        )
        logger.info(row_str)
        summary_lines.append(row_str)

    # 7. Baseline aggregate evaluation
    agg = aggregate_fold_results(all_fold_results)
    report_str = format_evaluation_report(
        fold_results=all_fold_results,
        agg=agg,
        model_name="Baseline_EXP001",
        variant=VARIANT,
        timeframe=TIMEFRAME,
    )
    logger.info("\n" + report_str)

    # 8. Save fold_report.txt
    fold_report_path = LOGS_DIR / "fold_report.txt"
    with open(fold_report_path, "w", encoding="utf-8") as f:
        f.write("PHASE 5 — WALK-FORWARD FOLD REPORT\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"Timeframe: {TIMEFRAME}  |  Variant: {VARIANT}\n\n")
        f.write(splitter.summary() + "\n\n")
        f.write("FOLD SUMMARY TABLE\n")
        f.write("\n".join(summary_lines) + "\n\n")
        f.write(report_str + "\n")
    logger.info(f"\nFold report saved: {fold_report_path}")

    # 9. Save fold_metadata.json
    metadata = {
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "primary_timeframe": TIMEFRAME,
        "primary_variant":   VARIANT,
        "symbols":           WATCHLIST,
        "feature_columns":   feature_cols,
        "dropped_features":  ["ema_9", "ema_21", "ema_50", "bb_upper", "bb_lower", "vwap"],
        "folds":             fold_metadata_list,
        "test": {
            "test_start":      tp["test_start"],
            "test_end":        tp["test_end"],
            "test_rows":       int(len(test_df)),
            "test_class_dist": test_dist,
        },
        "baseline_fold_results": [
            {
                "fold_id":            r["fold_id"],
                "combined_precision": r["combined_precision"],
                "signal_rate":        r["signal_rate"],
                "long_precision":     r["long_precision"],
                "short_precision":    r["short_precision"],
                "macro_f1":           r["macro_f1"],
            }
            for r in all_fold_results
        ],
    }
    meta_path = LOGS_DIR / "fold_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Fold metadata saved: {meta_path}")

    logger.info("\nPhase 5 complete. Report back to Claude Chat with:")
    logger.info("  1. logs/fold_report.txt")
    logger.info("  2. EXP_003 entry from EXPERIMENT_LOG.md")
    logger.info("  3. Baseline per-fold combined_precision values above")


if __name__ == "__main__":
    main()
