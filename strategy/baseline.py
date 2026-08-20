import pandas as pd
from strategy.base import BaseStrategy


class BaselineStrategy(BaseStrategy):
    """
    Model Zero — EMA crossover + RSI + VWAP + volume filter rule set.
    No ML, no parameter optimization. Used as the benchmark floor for Phase 6 ML.
    """

    def generate_signal(self, row: pd.Series) -> int:
        """
        Returns +1 (long), -1 (short), or 0 (no trade).
        All conditions in each direction must be satisfied simultaneously.
        """
        # Common blockers — checked before directional logic
        in_opening = row.get("is_opening_30min", 1) == 1
        high_vol = row.get("vol_regime_enc", 1) == 1
        if in_opening or high_vol:
            return 0

        volume_ratio = row.get("volume_ratio", 0.0)
        rsi = row.get("rsi_14", 50.0)
        nifty_trend = row.get("nifty_trend", 0)

        # Long conditions
        if (
            row.get("ema9_above_ema21", 0) == 1
            and row.get("ema21_above_ema50", 0) == 1
            and 40 <= rsi <= 65
            and row.get("vwap_above", 0) == 1
            and volume_ratio >= 1.2
            and nifty_trend >= 0
        ):
            return 1

        # Short conditions
        if (
            row.get("ema9_above_ema21", 1) == 0
            and row.get("ema21_above_ema50", 1) == 0
            and 35 <= rsi <= 60
            and row.get("vwap_above", 1) == 0
            and volume_ratio >= 1.2
            and nifty_trend <= 0
        ):
            return -1

        return 0
