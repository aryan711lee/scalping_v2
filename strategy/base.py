from abc import ABC, abstractmethod
from typing import Union
import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, row: Union[pd.Series, dict]) -> int:
        """
        Return +1 (long), -1 (short), or 0 (no trade).
        row may be a pd.Series (for tests/manual use) or a plain dict (engine hot path).
        Both support .get(key, default) so implementations must use .get() only.
        """
        pass

    def on_trade_open(self, trade_info: dict) -> None:
        """Optional hook called when a trade opens."""
        pass

    def on_trade_close(self, trade: object) -> None:
        """Optional hook called when a trade closes."""
        pass
