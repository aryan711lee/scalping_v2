"""
Leakage checker — mandatory before saving any fold results.

Silent data leakage is the most dangerous failure mode in ML research:
a model that looks great in validation but uses future information will fail
completely in live trading with no warning during development.
"""

import pandas as pd


# Column name substrings that indicate forward-looking features
_FORWARD_LOOKING_SUBSTRINGS = [
    "future", "forward", "next", "ahead", "target_price", "exit_price"
]


def check_fold_leakage(
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    fold_id: int,
) -> bool:
    """
    Verifies no temporal leakage between train and validation sets.

    Checks:
    1. No datetime in train_df >= validate_start
    2. No datetime in validate_df <= train_end
    3. No shared datetime values between the two sets
    4. Gap between max(train) and min(val) is at least 1 calendar day

    Returns True if clean. Raises ValueError if leakage is detected.
    """
    if train_df.empty or validate_df.empty:
        raise ValueError(
            f"Fold {fold_id}: empty train or validate DataFrame — cannot check leakage."
        )

    train_max = train_df.index.max()
    val_min   = validate_df.index.min()
    val_max   = validate_df.index.max()
    train_min = train_df.index.min()

    # Check 1: no train row is >= validate_start
    if train_max >= val_min:
        raise ValueError(
            f"Fold {fold_id} leakage: train_max ({train_max}) >= val_min ({val_min}). "
            "Training data overlaps validation period."
        )

    # Check 2: no validate row is <= train_end
    if val_max <= train_max:
        raise ValueError(
            f"Fold {fold_id} leakage: val_max ({val_max}) <= train_max ({train_max}). "
            "Validation data is entirely within training period."
        )

    # Check 3: no shared datetime values
    overlap = train_df.index.intersection(validate_df.index)
    if len(overlap) > 0:
        raise ValueError(
            f"Fold {fold_id} leakage: {len(overlap)} shared datetime values between "
            f"train and validate sets. First overlap: {overlap[0]}"
        )

    # Check 4: gap >= 1 trading day (timestamps are in IST, compare dates)
    train_max_date = train_max.date()
    val_min_date   = val_min.date()
    if val_min_date <= train_max_date:
        raise ValueError(
            f"Fold {fold_id} leakage: val_min date ({val_min_date}) is not strictly "
            f"after train_max date ({train_max_date}). Boundary overlap detected."
        )

    return True


def check_test_isolation(
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """
    Verifies the test holdout period is completely isolated from all training data.
    full_train_df should include all folds' train + validation data combined.
    Raises ValueError if any test datetime appears in full_train_df.
    """
    if test_df.empty:
        raise ValueError("test_df is empty — cannot verify test isolation.")

    overlap = full_train_df.index.intersection(test_df.index)
    if len(overlap) > 0:
        raise ValueError(
            f"Test isolation violated: {len(overlap)} test datetimes appear in "
            f"full_train_df. First: {overlap[0]}. "
            "The 2026 holdout has been contaminated."
        )

    # Additional date-level check: no train row has a date >= test_start
    test_min_date = test_df.index.min().date()
    train_max_date = full_train_df.index.max().date()
    if train_max_date >= test_min_date:
        raise ValueError(
            f"Test isolation violated: full_train_df has rows up to {train_max_date}, "
            f"but test period starts {test_min_date}."
        )

    return True


def check_feature_leakage(df: pd.DataFrame) -> list[str]:
    """
    Flags column names that suggest forward-looking features.
    Returns list of suspicious column names (should be empty for a clean dataset).
    """
    suspicious = []
    for col in df.columns:
        col_lower = col.lower()
        if any(s in col_lower for s in _FORWARD_LOOKING_SUBSTRINGS):
            suspicious.append(col)
    return suspicious
