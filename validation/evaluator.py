"""
Model evaluator — scores predictions consistently across all folds.

combined_precision is the key metric: when the model fires a directional signal,
how often is it correct? This maps directly to live-trading win rate.
"""

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score,
)


# Average candles per session on 3min timeframe: (375 min / 3) - opening/closing filter
# 375 min session, minus first 30 min and last 15 min = 330 min / 3 = 110 candles
_AVG_CANDLES_PER_SESSION = 110


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_id: int | str,
    model_name: str,
    variant: str,
) -> dict:
    """
    Computes all classification and trading-relevant metrics for one fold.
    Returns a dict keyed by metric name.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = [-1, 0, 1]

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    prec_short, prec_none, prec_long   = prec
    rec_short,  rec_none,  rec_long    = rec
    f1_short,   f1_none,   f1_long     = f1

    macro_f1    = f1_score(y_true, y_pred, labels=classes, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Trading metrics
    total = len(y_pred)
    long_signals  = (y_pred == 1).sum()
    short_signals = (y_pred == -1).sum()
    signal_count  = long_signals + short_signals
    signal_rate   = signal_count / total if total > 0 else 0.0

    # correct directional calls
    correct_long  = ((y_pred == 1)  & (y_true == 1)).sum()
    correct_short = ((y_pred == -1) & (y_true == -1)).sum()

    if signal_count > 0:
        combined_precision = (correct_long + correct_short) / signal_count
    else:
        combined_precision = None

    expected_trades_per_day = signal_rate * _AVG_CANDLES_PER_SESSION

    return {
        "fold_id":               fold_id,
        "model_name":            model_name,
        "variant":               variant,
        # per-class
        "precision_long":        float(prec_long),
        "recall_long":           float(rec_long),
        "f1_long":               float(f1_long),
        "precision_short":       float(prec_short),
        "recall_short":          float(rec_short),
        "f1_short":              float(f1_short),
        "precision_notrade":     float(prec_none),
        "recall_notrade":        float(rec_none),
        "macro_f1":              float(macro_f1),
        "weighted_f1":           float(weighted_f1),
        "confusion_matrix":      cm.tolist(),
        # trading metrics
        "signal_rate":           float(signal_rate),
        "long_precision":        float(prec_long),
        "short_precision":       float(prec_short),
        "combined_precision":    float(combined_precision) if combined_precision is not None else None,
        "expected_trades_per_day": float(expected_trades_per_day),
        "n_total":               int(total),
        "n_long_signals":        int(long_signals),
        "n_short_signals":       int(short_signals),
    }


def aggregate_fold_results(fold_results: list[dict]) -> dict:
    """
    Aggregates metrics across folds. For each numeric metric returns
    mean, std, min, max. High std/mean ratio = unstable model.
    """
    if not fold_results:
        return {}

    # Gather numeric scalar metrics (exclude non-numeric and per-fold metadata)
    skip = {"fold_id", "model_name", "variant", "confusion_matrix",
            "n_total", "n_long_signals", "n_short_signals"}

    numeric_keys = [
        k for k, v in fold_results[0].items()
        if k not in skip and v is not None and isinstance(v, (int, float))
    ]

    agg = {}
    for key in numeric_keys:
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        agg[key] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=0)),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "values": [float(v) for v in arr],
        }

    return agg


def format_evaluation_report(
    fold_results: list[dict],
    agg: dict,
    model_name: str,
    variant: str,
    timeframe: str,
) -> str:
    """Renders the evaluation report string matching the spec format."""
    lines = []
    sep = "=" * 60
    lines.append(sep)
    lines.append("MODEL EVALUATION REPORT")
    lines.append(f"Model: {model_name} | Variant: {variant} | Timeframe: {timeframe}")
    lines.append(sep)
    lines.append("")

    # Fold-by-fold table
    lines.append("FOLD-BY-FOLD RESULTS")
    header_metrics = [
        "combined_precision", "long_precision", "short_precision",
        "signal_rate", "macro_f1",
    ]
    fold_ids = [r["fold_id"] for r in fold_results]
    header = f"{'':25s}" + "".join(f"{'Fold ' + str(fid):12s}" for fid in fold_ids) + f"{'Mean ± Std':>18s}"
    lines.append(header)

    for m in header_metrics:
        vals = [r.get(m) for r in fold_results]
        row = f"{m:25s}"
        for v in vals:
            if v is None:
                row += f"{'N/A':12s}"
            elif m in ("combined_precision", "long_precision", "short_precision", "signal_rate"):
                row += f"{v * 100:>10.1f}%  "
            else:
                row += f"{v:>12.4f}"
        if m in agg:
            a = agg[m]
            if m in ("combined_precision", "long_precision", "short_precision", "signal_rate"):
                row += f"{a['mean']*100:>8.1f}% ± {a['std']*100:.1f}%"
            else:
                row += f"{a['mean']:>8.4f} ± {a['std']:.4f}"
        lines.append(row)

    lines.append("")
    lines.append("AGGREGATE SUMMARY")

    def _pct(key):
        a = agg.get(key, {})
        return f"{a.get('mean', 0)*100:.1f}% ± {a.get('std', 0)*100:.1f}%"

    cp = agg.get("combined_precision", {})
    sr = agg.get("signal_rate", {})
    et = agg.get("expected_trades_per_day", {})
    mean_cp = cp.get("mean", 0) or 0
    std_cp  = cp.get("std",  0) or 0
    mean_sr = sr.get("mean", 0) or 0
    consistency = (std_cp / mean_cp * 100) if mean_cp else float("nan")

    lines.append(f"  combined_precision:     {_pct('combined_precision')}   [target: >40%]")
    lines.append(f"  signal_rate:             {_pct('signal_rate')}   [lower = fewer, costlier trades]")
    lines.append(f"  estimated trades/day:    ~{et.get('mean', 0):.1f}          [per symbol]")
    lines.append(f"  consistency (std/mean):  {consistency:.1f}%           [<10% is acceptable]")

    # Confusion matrix from middle fold (fold 2 if exists)
    if len(fold_results) >= 2:
        rep_fold = fold_results[1]
    else:
        rep_fold = fold_results[0]
    cm = rep_fold.get("confusion_matrix", [])
    if cm:
        lines.append("")
        lines.append(f"CONFUSION MATRIX (Fold {rep_fold['fold_id']} — representative)")
        lines.append(f"{'':14s}{'Pred -1':>10s}{'Pred 0':>10s}{'Pred +1':>10s}")
        labels = ["True -1:", "True  0:", "True +1:"]
        for i, label in enumerate(labels):
            row_vals = cm[i] if i < len(cm) else [0, 0, 0]
            lines.append(f"  {label:12s}" + "".join(f"{v:>10,d}" for v in row_vals))

    lines.append("")
    lines.append("BASELINE COMPARISON (EXP_001)")
    lines.append("  EXP_001 signal rate:       ~100% (fires on all rule-matching candles)")
    lines.append("  EXP_001 win rate:           16.5%")
    lines.append(f"  This model signal rate:      {mean_sr*100:.1f}%")
    lines.append(f"  This model combined prec:   {mean_cp*100:.1f}%")
    verdict = "IMPROVEMENT over baseline" if mean_cp > 0.165 else "NOT AN IMPROVEMENT"
    lines.append(f"  Verdict: {verdict}")
    lines.append(sep)

    return "\n".join(lines)
