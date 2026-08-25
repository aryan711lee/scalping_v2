"""
Smoke test: one symbol, one fold, 10,000 rows max.
Verifies the full pipeline (LR → RF → XGBoost → threshold sweep)
runs without errors before committing to a full 15-symbol training run.

Usage:
  python scripts/smoke_test_training.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.logger import setup_logging
from datasets.builder import build_dataset
from models.trainer import ModelTrainer
from models.predictor import ModelPredictor
from models.selector import ModelSelector
from validation.splitter import WalkForwardSplitter, FOLD_DEFINITIONS
from validation.dataset_splitter import DatasetSplitter
from validation.evaluator import evaluate_predictions, aggregate_fold_results

setup_logging()
logger = logging.getLogger(__name__)

SMOKE_SYMBOL   = "NSE:RELIANCE-EQ"
TIMEFRAME      = "3min"
VARIANT        = "L1"
FOLD_ID        = 1
MAX_ROWS       = 10_000
ARTIFACTS_DIR  = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _cap(df, n):
    if len(df) > n:
        logger.info("  Capping to %d rows (from %d)", n, len(df))
        return df.iloc[:n].copy()
    return df


def make_logistic():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(solver="lbfgs", max_iter=500, C=0.1, random_state=42)


def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=50,          # smoke: reduced from 300
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=100,         # smoke: reduced from 500
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def run_one(model, model_name, train_df, val_df, feature_cols, class_weights):
    trainer = ModelTrainer(
        model=model,
        model_name=f"smoke_{model_name}",
        variant=VARIANT,
        feature_columns=feature_cols,
        class_weights=class_weights,
    )

    t0 = time.perf_counter()
    result = trainer.train_fold(train_df, fold_id=FOLD_ID, val_df=val_df)
    elapsed = time.perf_counter() - t0

    # Save fold model and load via predictor to test round-trip
    path = trainer.save_fold_model(result)
    predictor = ModelPredictor(path)

    y_pred  = predictor.predict(val_df[feature_cols], threshold=0.5)
    y_proba = predictor.predict_proba(val_df[feature_cols])
    y_true  = val_df["label"].values

    assert set(y_pred).issubset({-1, 0, 1}), f"Bad label values: {set(y_pred)}"
    assert y_proba.shape == (len(val_df), 3), f"Bad proba shape: {y_proba.shape}"
    np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-5)

    metrics = evaluate_predictions(
        y_pred=y_pred, y_true=y_true,
        fold_id=FOLD_ID, model_name=model_name, variant=VARIANT,
    )

    cp = (metrics["combined_precision"] or 0) * 100
    sr = metrics["signal_rate"] * 100
    print(
        f"  {model_name:<12} | train={len(train_df):>5,} val={len(val_df):>5,} "
        f"| combined_prec={cp:5.1f}%  signal_rate={sr:5.1f}%  "
        f"| time={elapsed:.1f}s"
    )

    return {
        "experiment_id": f"SMOKE_{model_name.upper()}",
        "model_name": model_name,
        "variant": VARIANT,
        "fold_results": [metrics],
        "agg": aggregate_fold_results([metrics]),
    }


def main():
    print("=" * 70)
    print(f"SMOKE TEST — {SMOKE_SYMBOL} | fold {FOLD_ID} | max {MAX_ROWS:,} rows")
    print("=" * 70)

    logger.info("Loading dataset for %s...", SMOKE_SYMBOL)
    df = build_dataset(SMOKE_SYMBOL, TIMEFRAME, VARIANT)
    logger.info("  Loaded %d rows", len(df))

    wf   = WalkForwardSplitter(FOLD_DEFINITIONS)
    ds   = DatasetSplitter(wf)
    feat = ds.get_feature_columns(df)
    print(f"  Features: {len(feat)}")

    train_df, val_df = ds.split(df, fold_id=FOLD_ID)
    train_df = _cap(train_df, MAX_ROWS)
    val_df   = _cap(val_df,   MAX_ROWS // 4)

    class_weights = ds.get_class_weights(train_df)
    print(f"  Class weights: { {k: round(v,3) for k,v in class_weights.items()} }")
    print()

    experiments = []

    print("Running Logistic Regression...")
    experiments.append(run_one(make_logistic(), "logistic", train_df, val_df, feat, class_weights))

    print("Running Random Forest (n=50, smoke)...")
    experiments.append(run_one(make_rf(), "rf", train_df, val_df, feat, class_weights))

    print("Running XGBoost (n=100, smoke)...")
    experiments.append(run_one(make_xgb(), "xgboost", train_df, val_df, feat, class_weights))

    # Quick threshold sweep on XGBoost fold model
    print("\nThreshold sweep on XGBoost (smoke):")
    xgb_path = ARTIFACTS_DIR / f"smoke_xgboost_{VARIANT}_fold{FOLD_ID}.joblib"
    xgb_pred = ModelPredictor(str(xgb_path))
    for t in [0.40, 0.50, 0.60]:
        y_pred = xgb_pred.predict(val_df[feat], threshold=t)
        sig_rate = ((y_pred != 0).sum() / len(y_pred)) * 100
        m = evaluate_predictions(y_pred, val_df["label"].values,
                                 fold_id=FOLD_ID, model_name="xgb_sweep", variant=VARIANT)
        cp = (m["combined_precision"] or 0) * 100
        print(f"  t={t:.2f} → combined_prec={cp:.1f}%  signal_rate={sig_rate:.1f}%")

    # Selector
    print("\nModel ranking:")
    selector = ModelSelector(experiments)
    ranked = selector.rank_models()
    print(ranked[["experiment_id", "combined_precision", "signal_rate", "composite_score"]].to_string(index=False))

    best = selector.select_best()
    print(f"\nBest model: {best['experiment_id']}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED — pipeline runs end-to-end without errors.")
    print("Ready for full training: python scripts/train_models.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
