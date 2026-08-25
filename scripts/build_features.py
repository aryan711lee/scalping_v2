"""
Build feature Parquet files for all symbols × timeframes.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --symbol NSE:RELIANCE-EQ
    python scripts/build_features.py --timeframe 3min
    python scripts/build_features.py --symbol NSE:RELIANCE-EQ --timeframe 3min
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.logger  # noqa: F401 — side-effect: configures root logger
from app.config import FEATURES_DIR, TIMEFRAMES, WATCHLIST
from data.ingestion.nifty import fetch_nifty, load_nifty
from features.market_context import compute_nifty_features
from features.pipeline import _safe_symbol, build_features

logger = logging.getLogger(__name__)

_BASE_COLS = {"datetime", "open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, symbol: str, tf: str) -> list[str]:
    """Return a list of error strings (empty = passed)."""
    errors: list[str] = []
    warnings_: list[str] = []

    if df.isnull().values.any():
        errors.append("NaN values remain after dropna")

    numeric = df.select_dtypes(include="number")
    if np.isinf(numeric.values).any():
        errors.append("Infinite values present")

    if "price_return_1" in df.columns:
        if (df["price_return_1"].abs() >= 0.20).any():
            warnings_.append("price_return_1 has values ≥ 20% — check for bad ticks")

    if "rsi_14" in df.columns:
        bad = df[(df["rsi_14"] < 0) | (df["rsi_14"] > 100)]
        if len(bad):
            errors.append(f"rsi_14 out of [0, 100]: {len(bad)} rows")

    if "volume_ratio" in df.columns and (df["volume_ratio"] < 0).any():
        errors.append("volume_ratio has negative values")

    if "bb_position" in df.columns:
        bad = df[(df["bb_position"] < -0.5) | (df["bb_position"] > 1.5)]
        if len(bad):
            warnings_.append(f"bb_position outside [-0.5, 1.5]: {len(bad)} rows")

    if "close_position" in df.columns:
        bad = df[(df["close_position"] < 0) | (df["close_position"] > 1)]
        if len(bad):
            warnings_.append(f"close_position outside [0, 1]: {len(bad)} rows")

    for col in ("nifty_trend", "trend_regime_enc", "vol_regime_enc"):
        if col in df.columns:
            bad = df[~df[col].isin([-1, 0, 1])]
            if len(bad):
                errors.append(f"{col} has values outside {{-1, 0, 1}}: {len(bad)} rows")

    for w in warnings_:
        logger.warning(f"  WARN  {symbol} {tf}: {w}")
    for e in errors:
        logger.error(f"  ERROR {symbol} {tf}: {e}")

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature Parquet files.")
    parser.add_argument("--symbol", default=None, help="Single symbol, e.g. NSE:RELIANCE-EQ")
    parser.add_argument("--timeframe", default=None, help="Single timeframe: 1min / 3min / 5min")
    parser.add_argument("--no-5min-context", action="store_true",
                        help="Skip 5-min multi-timeframe context for 3-min features (EXP_011 mode)")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else WATCHLIST
    timeframes = [args.timeframe] if args.timeframe else list(TIMEFRAMES.keys())

    # Step 1 — ensure NIFTY data is on disk
    logger.info("=== Step 1: NIFTY data ===")
    fetch_nifty()

    # Step 2 — create output directory
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 3 — build features per symbol × timeframe
    # Build in order: 5min first, then 3min (needs 5min context), then 1min.
    # If a single timeframe was requested, just use that order.
    logger.info("=== Step 2: Building features ===")
    summary: list[dict] = []

    _TF_ORDER = ["5min", "3min", "1min"]
    ordered_timeframes = [tf for tf in _TF_ORDER if tf in timeframes] + \
                         [tf for tf in timeframes if tf not in _TF_ORDER]

    # Cache 5-min feature DataFrames keyed by symbol for passing to 3-min builder
    _5min_cache: dict[str, pd.DataFrame] = {}

    for tf in ordered_timeframes:
        nifty_raw = load_nifty(tf)
        nifty_feat = compute_nifty_features(nifty_raw)

        for symbol in symbols:
            safe = _safe_symbol(symbol)
            out_path = FEATURES_DIR / f"{safe}_{tf}_features.parquet"

            # For 3-min, pass the 5-min feature DataFrame if available (unless disabled)
            df_5min = (
                _5min_cache.get(symbol)
                if tf == "3min" and not args.no_5min_context
                else None
            )

            try:
                df = build_features(symbol, tf, nifty_feat, df_5min=df_5min)

                errors = _validate(df, symbol, tf)
                if errors:
                    raise ValueError(f"Validation failed: {errors}")

                df.to_parquet(out_path, index=False)

                # Cache 5-min features for the 3-min pass
                if tf == "5min":
                    _5min_cache[symbol] = df

                feature_cols = [c for c in df.columns if c not in _BASE_COLS]
                date_range = f"{df['datetime'].min().date()} – {df['datetime'].max().date()}"
                summary.append(
                    dict(
                        symbol=symbol,
                        timeframe=tf,
                        rows=len(df),
                        features=len(feature_cols),
                        date_range=date_range,
                    )
                )
                logger.info(f"  OK  {safe}_{tf}_features.parquet  ({len(df)} rows, {len(feature_cols)} features)")

            except Exception as exc:
                logger.error(f"  FAILED {symbol} {tf}: {exc}")
                summary.append(dict(symbol=symbol, timeframe=tf, rows=0, features=0, date_range="ERROR"))

    # Step 4 — summary table
    print("\n" + "=" * 80)
    print(f"{'Symbol':<24} {'Timeframe':<10} {'Rows':>8} {'Features':>9}  Date Range")
    print("-" * 80)
    for r in summary:
        print(
            f"{r['symbol']:<24} {r['timeframe']:<10} {r['rows']:>8} {r['features']:>9}  {r['date_range']}"
        )
    print("=" * 80)

    n_failed = sum(1 for r in summary if r["rows"] == 0)
    if n_failed:
        logger.error(f"{n_failed} symbol/timeframe combination(s) failed.")
        sys.exit(1)
    else:
        logger.info(f"All {len(summary)} feature file(s) built successfully.")


if __name__ == "__main__":
    main()
