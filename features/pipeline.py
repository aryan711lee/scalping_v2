import logging
from typing import Optional

import pandas as pd

from app.config import PROCESSED_DIR
from features.market_context import add_nifty_context, add_time_features
from features.momentum import add_momentum
from features.momentum_quality import add_momentum_quality
from features.multi_timeframe import add_5min_context
from features.price_action import add_price_action
from features.regime import add_regime
from features.trend import add_trend
from features.volatility import add_volatility
from features.volume import add_volume

logger = logging.getLogger(__name__)

# Columns present in the cleaned parquet that are not feature columns.
_BASE_COLS = {"datetime", "open", "high", "low", "close", "volume"}


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def build_features(
    symbol: str,
    timeframe: str,
    nifty_df: pd.DataFrame,
    df_5min: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load cleaned candles for symbol+timeframe, compute all features,
    join pre-computed NIFTY context features, drop NaN warmup rows,
    return complete feature DataFrame.

    nifty_df must be the output of compute_nifty_features() — i.e. a DataFrame
    with columns [datetime, nifty_return_1, nifty_return_3, nifty_return_5,
    nifty_rsi, nifty_trend].

    df_5min: optional 5-min feature DataFrame (output of build_features for 5min).
    When provided and timeframe == '3min', 5-min context columns are added.
    """
    safe = _safe_symbol(symbol)
    path = PROCESSED_DIR / f"{safe}_{timeframe}_clean.parquet"
    df = pd.read_parquet(path)
    logger.info(f"{symbol} {timeframe}: loaded {len(df)} rows from {path.name}")

    df = add_price_action(df)
    df = add_trend(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume(df)
    df = add_time_features(df)
    df = add_nifty_context(df, nifty_df)
    df = add_regime(df)
    df = add_momentum_quality(df)

    # Multi-timeframe context: only for 3-min when 5-min features are available
    if timeframe == "3min" and df_5min is not None:
        df = add_5min_context(df, df_5min)
        logger.info(f"{symbol} {timeframe}: added 5-min context columns")

    feature_cols = [c for c in df.columns if c not in _BASE_COLS]
    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    logger.info(
        f"{symbol} {timeframe}: dropped {dropped} NaN warmup rows → {len(df)} rows, "
        f"{len(feature_cols)} feature columns"
    )

    return df.reset_index(drop=True)
