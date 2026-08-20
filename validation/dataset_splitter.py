"""
Applies walk-forward fold definitions to a labeled feature DataFrame.
"""

import logging

import pandas as pd

from validation.splitter import WalkForwardSplitter

logger = logging.getLogger(__name__)

# Raw price-level features dropped before ML training (Phase 4 decision)
_DROPPED_FEATURES = {"ema_9", "ema_21", "ema_50", "bb_upper", "bb_lower", "vwap"}

# Columns that are metadata, not features
_NON_FEATURE_COLS = {"label", "symbol"}


class DatasetSplitter:
    def __init__(self, splitter: WalkForwardSplitter):
        self.splitter = splitter

    def split(
        self,
        df: pd.DataFrame,
        fold_id: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (train_df, validate_df) for the given fold, both in chronological order.
        No shuffling — shuffling is the ML trainer's responsibility.
        """
        fold = self.splitter.get_fold(fold_id)

        train_start = pd.Timestamp(fold["train_start"]).tz_localize("Asia/Kolkata")
        train_end   = pd.Timestamp(fold["train_end"]).tz_localize("Asia/Kolkata") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        val_start   = pd.Timestamp(fold["validate_start"]).tz_localize("Asia/Kolkata")
        val_end     = pd.Timestamp(fold["validate_end"]).tz_localize("Asia/Kolkata") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        train_df = df.loc[(df.index >= train_start) & (df.index <= train_end)].copy()
        val_df   = df.loc[(df.index >= val_start)   & (df.index <= val_end)].copy()

        logger.debug(
            f"Fold {fold_id}: train={len(train_df)} rows, val={len(val_df)} rows"
        )
        return train_df, val_df

    def get_test_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns test holdout rows. Logs a warning on every call — this data
        must only be accessed for final Phase 6 evaluation.
        """
        logger.warning(
            "WARNING: Accessing test holdout data. This should only happen "
            "during final Phase 6 evaluation."
        )
        tp = self.splitter.get_test_period()
        test_start = pd.Timestamp(tp["test_start"]).tz_localize("Asia/Kolkata")
        test_end   = pd.Timestamp(tp["test_end"]).tz_localize("Asia/Kolkata") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return df.loc[(df.index >= test_start) & (df.index <= test_end)].copy()

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Returns numeric feature column names, excluding label, symbol, the 6
        raw price-level features from Phase 4, and any non-numeric columns
        (e.g. trend_regime / vol_regime raw string labels — use their _enc versions).
        """
        exclude = _NON_FEATURE_COLS | _DROPPED_FEATURES
        return [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    def get_class_weights(self, train_df: pd.DataFrame) -> dict:
        """
        Balanced class weights: weight_c = total / (n_classes * count_c).
        Returns {-1: float, 0: float, 1: float}.
        """
        counts = train_df["label"].value_counts()
        total = len(train_df)
        n_classes = 3
        weights = {}
        for cls in [-1, 0, 1]:
            count = counts.get(cls, 1)
            weights[cls] = total / (n_classes * count)
        return weights
