"""
Model selector — ranks experiments across folds and selects the best one.

Composite score:
  score = combined_precision_mean
          - penalty_for_instability
          - penalty_for_low_signal_rate

  penalty_for_instability     = max(0, std/mean - 0.10) * 20
  penalty_for_low_signal_rate = max(0, 0.02 - signal_rate_mean) * 100
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class ModelSelector:
    def __init__(self, experiment_results: list):
        """
        experiment_results: list of dicts, one per experiment.
        Each dict must contain:
          - 'experiment_id': str (e.g. 'EXP_004')
          - 'model_name': str
          - 'agg': dict from aggregate_fold_results()
            with keys 'combined_precision' and 'signal_rate', each having
            sub-keys 'mean' and 'std'
          - 'fold_results': list of per-fold metric dicts
        """
        self._experiments = experiment_results

    def rank_models(self) -> pd.DataFrame:
        """
        Returns DataFrame sorted by composite score descending.
        """
        rows = []
        for exp in self._experiments:
            score, breakdown = self._score(exp)
            agg = exp.get("agg", {})
            cp = agg.get("combined_precision", {})
            sr = agg.get("signal_rate", {})
            rows.append({
                "experiment_id":       exp.get("experiment_id", "?"),
                "model_name":          exp.get("model_name", "?"),
                "combined_precision":  cp.get("mean", 0.0),
                "cp_std":              cp.get("std", 0.0),
                "signal_rate":         sr.get("mean", 0.0),
                "consistency":         breakdown["consistency"],
                "penalty_instability": breakdown["penalty_instability"],
                "penalty_signal_rate": breakdown["penalty_signal_rate"],
                "composite_score":     score,
            })
        df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df

    def select_best(self) -> dict:
        """
        Returns the experiment dict with the highest composite score.
        Logs the selection rationale.
        """
        if not self._experiments:
            raise ValueError("No experiments provided to ModelSelector")

        ranked = [(self._score(exp)[0], exp) for exp in self._experiments]
        ranked.sort(key=lambda x: x[0], reverse=True)
        best_score, best_exp = ranked[0]

        agg = best_exp.get("agg", {})
        cp = agg.get("combined_precision", {})
        sr = agg.get("signal_rate", {})

        logger.info(
            "Selected best model: %s (score=%.4f, combined_precision=%.1f%%, "
            "signal_rate=%.1f%%)",
            best_exp.get("experiment_id", "?"),
            best_score,
            cp.get("mean", 0) * 100,
            sr.get("mean", 0) * 100,
        )
        return best_exp

    def _score(self, exp: dict) -> tuple:
        agg = exp.get("agg", {})
        cp = agg.get("combined_precision", {})
        sr = agg.get("signal_rate", {})

        cp_mean = cp.get("mean", 0.0) or 0.0
        cp_std  = cp.get("std",  0.0) or 0.0
        sr_mean = sr.get("mean", 0.0) or 0.0

        consistency = (cp_std / cp_mean) if cp_mean > 0 else float("inf")

        penalty_instability = max(0.0, consistency - 0.10) * 20.0
        penalty_signal_rate = max(0.0, 0.02 - sr_mean) * 100.0

        score = cp_mean - penalty_instability - penalty_signal_rate

        return score, {
            "consistency": consistency,
            "penalty_instability": penalty_instability,
            "penalty_signal_rate": penalty_signal_rate,
        }
