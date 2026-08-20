"""
MLStrategy — wraps a trained ModelPredictor and plugs into the Phase 3 backtester.

Afternoon session filter: EXP_001 showed 8.8% win rate after 13:30 vs ~19% morning/midday.
When avoid_afternoon=True, any candle with is_closing_30min==1 returns 0.
This is configurable to allow A/B comparison in EXP_008.
"""

import logging
from typing import Union

import numpy as np
import pandas as pd

from models.predictor import ModelPredictor
from strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

# 13:30 = 810 minutes from midnight
_AFTERNOON_START_MINUTE = 810


class MLStrategy(BaseStrategy):
    def __init__(
        self,
        predictor: ModelPredictor,
        feature_columns: list,
        threshold: float,
        avoid_afternoon: bool = True,
    ):
        self.predictor = predictor
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.avoid_afternoon = avoid_afternoon

    def generate_signal(self, row: Union[pd.Series, dict]) -> int:
        """
        Generates a trading signal for a single candle row.

        Returns +1 (long), -1 (short), or 0 (no trade).
        If avoid_afternoon=True and is_closing_30min==1, returns 0.
        """
        if self.avoid_afternoon:
            # is_closing_30min is the feature flag; also check raw time as fallback
            if row.get("is_closing_30min", 0) == 1:
                return 0
            # fallback via datetime if present
            dt = row.get("datetime")
            if dt is not None:
                minute = dt.hour * 60 + dt.minute
                if minute >= _AFTERNOON_START_MINUTE:
                    return 0

        # Build a one-row DataFrame for the predictor
        row_data = {col: [row.get(col, np.nan)] for col in self.feature_columns}
        X = pd.DataFrame(row_data)

        preds = self.predictor.predict(X, threshold=self.threshold)
        return int(preds[0])
