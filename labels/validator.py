"""
Label quality checks and distribution reporting.

Checks run after construction:
  1. Distribution check — flag if long/short < 2% or > 40%, or none < 30%
  2. Leakage sanity check — spot-verify 100 random +1 and -1 labels
  3. Cross-variant consistency — L2 should differ from L1 in distribution
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from labels.variants import LABEL_VARIANTS

logger = logging.getLogger(__name__)

# Thresholds
_MIN_TRADEABLE_PCT = 2.0
_MAX_TRADEABLE_PCT = 40.0
_MIN_NONE_PCT = 30.0
_SPOT_CHECK_N = 100


def _distribution(labels: pd.Series) -> dict:
    total = len(labels)
    count_long  = int((labels == 1).sum())
    count_short = int((labels == -1).sum())
    count_none  = int((labels == 0).sum())
    return {
        "total":       total,
        "count_long":  count_long,
        "count_short": count_short,
        "count_none":  count_none,
        "long_pct":    count_long  / total * 100 if total else 0,
        "short_pct":   count_short / total * 100 if total else 0,
        "none_pct":    count_none  / total * 100 if total else 0,
    }


def _flag_distribution(dist: dict) -> list[str]:
    flags = []
    if dist["long_pct"] < _MIN_TRADEABLE_PCT:
        flags.append(f"long_pct={dist['long_pct']:.1f}% < {_MIN_TRADEABLE_PCT}%")
    if dist["long_pct"] > _MAX_TRADEABLE_PCT:
        flags.append(f"long_pct={dist['long_pct']:.1f}% > {_MAX_TRADEABLE_PCT}%")
    if dist["short_pct"] < _MIN_TRADEABLE_PCT:
        flags.append(f"short_pct={dist['short_pct']:.1f}% < {_MIN_TRADEABLE_PCT}%")
    if dist["short_pct"] > _MAX_TRADEABLE_PCT:
        flags.append(f"short_pct={dist['short_pct']:.1f}% > {_MAX_TRADEABLE_PCT}%")
    if dist["none_pct"] < _MIN_NONE_PCT:
        flags.append(f"none_pct={dist['none_pct']:.1f}% < {_MIN_NONE_PCT}%")
    return flags


def leakage_check(
    df: pd.DataFrame,
    labels: pd.Series,
    target_pct: float,
    stop_pct: float,
    horizon: int,
    n_samples: int = _SPOT_CHECK_N,
) -> list[str]:
    """
    Spot-check n_samples random +1 and n_samples random -1 labels.
    Returns list of error strings (empty = passed).
    """
    errors = []
    dt = pd.to_datetime(df["datetime"])
    dates = dt.dt.date.to_numpy()
    opens  = df["open"].to_numpy(dtype=np.float64)
    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)

    def _verify_long(i: int) -> str | None:
        """Return error string if label[i]==+1 is wrong, else None."""
        entry_idx = i + 1
        if entry_idx >= len(df):
            return f"idx={i}: no next candle for entry"
        entry_price = opens[entry_idx]
        target = entry_price * (1 + target_pct)
        stop   = entry_price * (1 - stop_pct)
        current_date = dates[i]

        for j in range(i + 1, min(len(df), i + 1 + horizon)):
            if dates[j] != current_date:
                break
            h, lo = highs[j], lows[j]
            if lo <= stop:
                return f"idx={i}: stop hit at j={j} before target"
            if h >= target:
                return None  # target reached, no stop before it
        return f"idx={i}: target never reached within horizon"

    def _verify_short(i: int) -> str | None:
        entry_idx = i + 1
        if entry_idx >= len(df):
            return f"idx={i}: no next candle for entry"
        entry_price = opens[entry_idx]
        target = entry_price * (1 - target_pct)
        stop   = entry_price * (1 + stop_pct)
        current_date = dates[i]

        for j in range(i + 1, min(len(df), i + 1 + horizon)):
            if dates[j] != current_date:
                break
            h, lo = highs[j], lows[j]
            if h >= stop:
                return f"idx={i}: stop hit at j={j} before target"
            if lo <= target:
                return None
        return f"idx={i}: target never reached within horizon"

    long_idxs  = labels.index[labels == 1].tolist()
    short_idxs = labels.index[labels == -1].tolist()

    sample_long  = random.sample(long_idxs,  min(n_samples, len(long_idxs)))
    sample_short = random.sample(short_idxs, min(n_samples, len(short_idxs)))

    for i in sample_long:
        err = _verify_long(i)
        if err:
            errors.append(f"LONG leakage: {err}")

    for i in sample_short:
        err = _verify_short(i)
        if err:
            errors.append(f"SHORT leakage: {err}")

    return errors


def validate_labels(
    df: pd.DataFrame,
    labels: pd.Series,
    symbol: str,
    timeframe: str,
    variant: str,
    target_pct: float,
    stop_pct: float,
    horizon: int,
) -> dict:
    """
    Run all validation checks. Returns a result dict with keys:
      symbol, timeframe, variant, dist (dict), flags (list), leakage_errors (list)
    """
    dist = _distribution(labels)
    flags = _flag_distribution(dist)
    leakage_errors = leakage_check(df, labels, target_pct, stop_pct, horizon)

    if leakage_errors:
        logger.warning(
            f"{symbol} {timeframe} {variant}: {len(leakage_errors)} leakage errors: "
            f"{leakage_errors[:3]}"
        )

    return {
        "symbol":         symbol,
        "timeframe":      timeframe,
        "variant":        variant,
        "dist":           dist,
        "flags":          flags,
        "leakage_errors": leakage_errors,
        "flagged":        bool(flags or leakage_errors),
    }


def cross_variant_check(results: list[dict]) -> list[str]:
    """
    Check that L1 and L2 don't have identical distributions (indicates a bug).
    Returns list of warning strings.
    """
    warnings = []
    by_key: dict[tuple, dict] = {}
    for r in results:
        key = (r["symbol"], r["timeframe"], r["variant"])
        by_key[key] = r

    all_symbols_tfs = set((r["symbol"], r["timeframe"]) for r in results)
    for sym, tf in all_symbols_tfs:
        l1 = by_key.get((sym, tf, "L1"))
        l2 = by_key.get((sym, tf, "L2"))
        if l1 and l2:
            d1 = l1["dist"]
            d2 = l2["dist"]
            if (
                abs(d1["long_pct"]  - d2["long_pct"])  < 0.01
                and abs(d1["short_pct"] - d2["short_pct"]) < 0.01
            ):
                warnings.append(
                    f"{sym} {tf}: L1 and L2 have identical distributions — possible bug"
                )
    return warnings


def print_summary_table(results: list[dict]) -> str:
    """Return formatted summary table string and print it."""
    header = (
        f"{'Variant':<8} {'Symbol':<26} {'TF':<6} "
        f"{'Long%':>7} {'Short%':>7} {'None%':>7}  Flagged"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        dist = r["dist"]
        flagged = "YES ***" if r["flagged"] else "no"
        lines.append(
            f"{r['variant']:<8} {r['symbol']:<26} {r['timeframe']:<6} "
            f"{dist['long_pct']:>6.1f}% {dist['short_pct']:>6.1f}% "
            f"{dist['none_pct']:>6.1f}%  {flagged}"
        )

    table = "\n".join(lines)
    print(table)
    return table


def save_report(results: list[dict], cross_warnings: list[str], logs_dir: Path) -> None:
    """Save full validation report to logs/label_validation_report.txt."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "label_validation_report.txt"

    table = print_summary_table(results)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("LABEL VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(table + "\n\n")

        if cross_warnings:
            f.write("CROSS-VARIANT WARNINGS:\n")
            for w in cross_warnings:
                f.write(f"  {w}\n")
            f.write("\n")

        flagged = [r for r in results if r["flagged"]]
        if flagged:
            f.write(f"FLAGGED ({len(flagged)} combinations):\n")
            for r in flagged:
                f.write(f"  {r['variant']} {r['symbol']} {r['timeframe']}\n")
                for flag in r["flags"]:
                    f.write(f"    flag: {flag}\n")
                for err in r["leakage_errors"][:5]:
                    f.write(f"    leakage: {err}\n")
        else:
            f.write("No flags raised — all distributions within expected ranges.\n")

    logger.info(f"Validation report saved to {out_path}")
