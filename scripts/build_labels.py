"""
Build all label Parquet files (15 symbols x 3 timeframes x 4 variants = 180 files).

Usage:
  python scripts/build_labels.py
  python scripts/build_labels.py --variant L1
  python scripts/build_labels.py --symbol NSE:RELIANCE-EQ
  python scripts/build_labels.py --timeframe 3min
  python scripts/build_labels.py --variant L1 --timeframe 3min
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.logger  # noqa: F401 — configures root logger
from app.config import FEATURES_DIR, LOGS_DIR, TIMEFRAMES, WATCHLIST
from labels.constructor import construct_labels
from labels.validator import (
    cross_variant_check,
    save_report,
    validate_labels,
)
from labels.variants import LABEL_VARIANTS

logger = logging.getLogger(__name__)

LABELS_DIR = Path(__file__).resolve().parent.parent / "data_storage" / "labels"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace(":", "_").replace("-", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label Parquet files.")
    parser.add_argument("--variant",   default=None, help="Single variant: L1/L2/L3/L4")
    parser.add_argument("--symbol",    default=None, help="Single symbol e.g. NSE:RELIANCE-EQ")
    parser.add_argument("--timeframe", default=None, help="Single timeframe: 1min/3min/5min")
    args = parser.parse_args()

    symbols   = [args.symbol]   if args.symbol   else WATCHLIST
    timeframes = [args.timeframe] if args.timeframe else list(TIMEFRAMES.keys())
    variants  = [args.variant]  if args.variant  else list(LABEL_VARIANTS.keys())

    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    total_combinations = len(symbols) * len(timeframes) * len(variants)
    logger.info(
        f"Building labels: {len(symbols)} symbols x {len(timeframes)} timeframes "
        f"x {len(variants)} variants = {total_combinations} combinations"
    )

    all_results = []
    summary_rows = []
    n_failed = 0

    for tf in timeframes:
        for symbol in symbols:
            safe = _safe_symbol(symbol)
            feat_path = FEATURES_DIR / f"{safe}_{tf}_features.parquet"

            if not feat_path.exists():
                logger.error(f"Feature file missing: {feat_path} — skipping {symbol} {tf}")
                n_failed += len(variants)
                continue

            try:
                feat_df = pd.read_parquet(feat_path)
                feat_df["datetime"] = pd.to_datetime(feat_df["datetime"])
                logger.info(f"Loaded {symbol} {tf}: {len(feat_df)} rows")
            except Exception as exc:
                logger.error(f"Failed to load {feat_path}: {exc}")
                n_failed += len(variants)
                continue

            for variant in variants:
                vcfg = LABEL_VARIANTS[variant]
                t0 = time.time()

                try:
                    labels = construct_labels(
                        feat_df,
                        target_pct=vcfg["target_pct"],
                        stop_pct=vcfg["stop_pct"],
                        horizon=vcfg["horizon"],
                    )

                    # Save label Parquet
                    out_path = LABELS_DIR / f"{safe}_{tf}_{variant}_labels.parquet"
                    label_df = pd.DataFrame(
                        {
                            "datetime": feat_df["datetime"],
                            "label":    labels.values,
                            "variant":  variant,
                        }
                    )
                    label_df["label"] = label_df["label"].astype("int8")
                    label_df.to_parquet(out_path, index=False)

                    elapsed = time.time() - t0
                    n_long  = int((labels == 1).sum())
                    n_short = int((labels == -1).sum())
                    n_none  = int((labels == 0).sum())
                    total   = len(labels)

                    logger.info(
                        f"  {variant} {safe} {tf}: "
                        f"+1={n_long} ({n_long/total*100:.1f}%) "
                        f"-1={n_short} ({n_short/total*100:.1f}%) "
                        f"0={n_none} ({n_none/total*100:.1f}%)  "
                        f"[{elapsed:.1f}s]"
                    )

                    # Validate
                    result = validate_labels(
                        feat_df, labels,
                        symbol=symbol, timeframe=tf, variant=variant,
                        target_pct=vcfg["target_pct"],
                        stop_pct=vcfg["stop_pct"],
                        horizon=vcfg["horizon"],
                    )
                    all_results.append(result)

                    summary_rows.append({
                        "symbol":    symbol,
                        "timeframe": tf,
                        "variant":   variant,
                        "total":     total,
                        "long_pct":  n_long  / total * 100,
                        "short_pct": n_short / total * 100,
                        "none_pct":  n_none  / total * 100,
                        "flagged":   result["flagged"],
                        "elapsed_s": elapsed,
                    })

                except Exception as exc:
                    logger.error(f"  FAILED {variant} {symbol} {tf}: {exc}", exc_info=True)
                    n_failed += 1

    # Cross-variant consistency check
    cross_warnings = cross_variant_check(all_results)
    if cross_warnings:
        for w in cross_warnings:
            logger.warning(f"Cross-variant: {w}")

    # Save validation report
    save_report(all_results, cross_warnings, LOGS_DIR)

    # Final summary
    print("\n" + "=" * 90)
    print(
        f"{'Symbol':<26} {'TF':<6} {'Variant':<8} "
        f"{'Long%':>7} {'Short%':>7} {'None%':>7}  {'Time':>6}  Flagged"
    )
    print("-" * 90)
    for r in summary_rows:
        flag_str = "YES ***" if r["flagged"] else "no"
        print(
            f"{r['symbol']:<26} {r['timeframe']:<6} {r['variant']:<8} "
            f"{r['long_pct']:>6.1f}% {r['short_pct']:>6.1f}% {r['none_pct']:>6.1f}%  "
            f"{r['elapsed_s']:>5.1f}s  {flag_str}"
        )
    print("=" * 90)

    n_flagged = sum(1 for r in summary_rows if r["flagged"])
    n_built = len(summary_rows)
    logger.info(f"Built {n_built} label files, {n_flagged} flagged, {n_failed} failed.")

    if n_failed:
        logger.error(f"{n_failed} combination(s) failed — check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
