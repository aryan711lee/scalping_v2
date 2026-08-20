"""
Walk-forward fold definitions for temporal validation.

Fold dates are hardcoded and must not change mid-research —
changing them invalidates all cross-experiment comparisons.
"""

from datetime import datetime


FOLD_DEFINITIONS = [
    {
        "fold_id": 1,
        "train_start":    "2023-01-02",
        "train_end":      "2024-06-30",
        "validate_start": "2024-07-01",
        "validate_end":   "2024-12-31",
    },
    {
        "fold_id": 2,
        "train_start":    "2023-01-02",
        "train_end":      "2024-12-31",
        "validate_start": "2025-01-01",
        "validate_end":   "2025-06-30",
    },
    {
        "fold_id": 3,
        "train_start":    "2023-01-02",
        "train_end":      "2025-06-30",
        "validate_start": "2025-07-01",
        "validate_end":   "2025-12-31",
    },
]

TEST_PERIOD = {
    "test_start": "2026-01-01",
    "test_end":   "2026-08-19",
}


def _months_between(start_str: str, end_str: str) -> int:
    s = datetime.strptime(start_str, "%Y-%m-%d")
    e = datetime.strptime(end_str, "%Y-%m-%d")
    return (e.year - s.year) * 12 + (e.month - s.month)


class WalkForwardSplitter:
    def __init__(self, fold_definitions: list[dict]):
        self._folds = {f["fold_id"]: f for f in fold_definitions}

    def get_fold(self, fold_id: int) -> dict:
        if fold_id not in self._folds:
            raise KeyError(f"fold_id {fold_id} not found. Available: {list(self._folds)}")
        return self._folds[fold_id]

    def get_test_period(self) -> dict:
        return dict(TEST_PERIOD)

    def get_all_folds(self) -> list[dict]:
        return [self._folds[k] for k in sorted(self._folds)]

    def summary(self) -> str:
        lines = []
        for f in self.get_all_folds():
            train_m = _months_between(f["train_start"], f["train_end"])
            val_m   = _months_between(f["validate_start"], f["validate_end"])
            lines.append(
                f"Fold {f['fold_id']}: "
                f"Train {f['train_start']}->{f['train_end']} ({train_m}m) | "
                f"Val {f['validate_start']}->{f['validate_end']} ({val_m}m)"
            )
        tp = TEST_PERIOD
        test_m = _months_between(tp["test_start"], tp["test_end"])
        lines.append(
            f"Test:   {tp['test_start']}->{tp['test_end']} ({test_m}m) "
            "[HOLDOUT - do not touch until Phase 6 final eval]"
        )
        return "\n".join(lines)
