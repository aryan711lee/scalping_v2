import logging
import warnings
from pathlib import Path

import pandas as pd
import pytz

import app.logger  # noqa: F401
from app.config import LOGS_DIR, MARKET_OPEN, MARKET_CLOSE, TIMEFRAMES

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

COMPLETENESS_WARN = 0.95
COMPLETENESS_CRITICAL = 0.80

# Trading minutes per session
_OPEN_MINUTES = sum(
    divmod(int(t.split(":")[0]) * 60 + int(t.split(":")[1]), 1)[0]
    for t in [MARKET_CLOSE, MARKET_OPEN]
)
SESSION_MINUTES = (
    (int(MARKET_CLOSE.split(":")[0]) * 60 + int(MARKET_CLOSE.split(":")[1]))
    - (int(MARKET_OPEN.split(":")[0]) * 60 + int(MARKET_OPEN.split(":")[1]))
)  # = 375 minutes


def _expected_candles_per_day(interval_minutes: int) -> int:
    return SESSION_MINUTES // interval_minutes


def _count_trading_days(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return df["datetime"].dt.date.nunique()


def validate_symbol(
    df: pd.DataFrame,
    symbol: str,
    timeframe_label: str,
    interval_minutes: int,
) -> dict:
    trading_days = _count_trading_days(df)
    expected_per_day = _expected_candles_per_day(interval_minutes)
    expected_total = trading_days * expected_per_day
    actual_total = len(df)
    completeness = actual_total / expected_total if expected_total else 0.0

    zero_vol = int((df["volume"] == 0).sum()) if not df.empty else 0
    ohlc_anomalies = 0
    if not df.empty:
        ohlc_anomalies = int((
            (df["high"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
        ).sum())

    # Gap detection — intraday only (skip overnight gaps)
    if not df.empty and len(df) > 1:
        expected_delta = pd.Timedelta(minutes=interval_minutes)
        dt = df["datetime"]
        same_day = dt.dt.date.values[1:] == dt.dt.date.values[:-1]
        deltas = dt.diff().iloc[1:]
        gaps = int(((deltas > expected_delta) & same_day).sum())
    else:
        gaps = 0

    date_min = df["datetime"].min() if not df.empty else None
    date_max = df["datetime"].max() if not df.empty else None
    vol_min = float(df["volume"].min()) if not df.empty else 0
    vol_max = float(df["volume"].max()) if not df.empty else 0
    vol_mean = float(df["volume"].mean()) if not df.empty else 0
    price_min = float(df["close"].min()) if not df.empty else 0
    price_max = float(df["close"].max()) if not df.empty else 0

    result = {
        "symbol": symbol,
        "timeframe": timeframe_label,
        "actual_candles": actual_total,
        "expected_candles": expected_total,
        "completeness": round(completeness, 4),
        "date_from": str(date_min),
        "date_to": str(date_max),
        "gaps": gaps,
        "zero_volume_candles": zero_vol,
        "ohlc_anomalies_remaining": ohlc_anomalies,
        "vol_min": round(vol_min, 0),
        "vol_max": round(vol_max, 0),
        "vol_mean": round(vol_mean, 2),
        "price_min": round(price_min, 2),
        "price_max": round(price_max, 2),
        "flagged": completeness < COMPLETENESS_WARN,
        "critical": completeness < COMPLETENESS_CRITICAL,
    }

    if completeness < COMPLETENESS_CRITICAL:
        warnings.warn(
            f"CRITICAL: {symbol} [{timeframe_label}] completeness={completeness:.1%} (< {COMPLETENESS_CRITICAL:.0%})",
            stacklevel=2,
        )
    elif completeness < COMPLETENESS_WARN:
        warnings.warn(
            f"WARNING: {symbol} [{timeframe_label}] completeness={completeness:.1%} (< {COMPLETENESS_WARN:.0%})",
            stacklevel=2,
        )

    return result


def run_validation_report(
    data: dict[tuple[str, str], pd.DataFrame],
    interval_map: dict[str, int] | None = None,
) -> list[dict]:
    """
    Run validation on all (symbol, timeframe) DataFrames.
    Prints a summary table and saves the report to logs/.
    """
    if interval_map is None:
        interval_map = {k: int(v) for k, v in TIMEFRAMES.items()}

    rows = []
    for (symbol, tf_label), df in data.items():
        interval = interval_map.get(tf_label, 1)
        result = validate_symbol(df, symbol, tf_label, interval)
        rows.append(result)

    _print_summary(rows)
    _save_report(rows)
    return rows


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        print("No data to validate.")
        return

    header = f"{'Symbol':<25} {'TF':<6} {'Actual':>8} {'Expected':>9} {'Complete':>9} {'Gaps':>6} {'ZeroVol':>8} {'Flag'}"
    print("\n" + "=" * len(header))
    print("DATA VALIDATION REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    critical_symbols = []
    for r in rows:
        flag = ""
        if r["critical"]:
            flag = "CRITICAL"
            critical_symbols.append(r["symbol"])
        elif r["flagged"]:
            flag = "WARN"
        print(
            f"{r['symbol']:<25} {r['timeframe']:<6} {r['actual_candles']:>8,} {r['expected_candles']:>9,} "
            f"{r['completeness']:>8.1%} {r['gaps']:>6} {r['zero_volume_candles']:>8} {flag}"
        )

    print("=" * len(header))
    if critical_symbols:
        print(f"CRITICAL symbols (completeness < {COMPLETENESS_CRITICAL:.0%}): {', '.join(set(critical_symbols))}")
    flagged_count = sum(1 for r in rows if r["flagged"] and not r["critical"])
    if flagged_count:
        print(f"Flagged symbols (completeness < {COMPLETENESS_WARN:.0%}): {flagged_count}")
    print()


def _save_report(rows: list[dict]) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / "data_validation_report.txt"

    lines = ["DATA VALIDATION REPORT", "=" * 80, ""]
    for r in rows:
        lines.append(f"Symbol:             {r['symbol']}")
        lines.append(f"Timeframe:          {r['timeframe']}")
        lines.append(f"Actual candles:     {r['actual_candles']:,}")
        lines.append(f"Expected candles:   {r['expected_candles']:,}")
        lines.append(f"Completeness:       {r['completeness']:.2%}")
        lines.append(f"Date range:         {r['date_from']} – {r['date_to']}")
        lines.append(f"Gaps detected:      {r['gaps']}")
        lines.append(f"Zero-volume candles:{r['zero_volume_candles']}")
        lines.append(f"OHLC anomalies:     {r['ohlc_anomalies_remaining']}")
        lines.append(f"Volume (min/max/mean): {r['vol_min']:.0f} / {r['vol_max']:.0f} / {r['vol_mean']:.1f}")
        lines.append(f"Price range (close): {r['price_min']:.2f} – {r['price_max']:.2f}")
        lines.append(f"Flagged:            {'YES' if r['flagged'] else 'no'}")
        lines.append(f"Critical:           {'YES' if r['critical'] else 'no'}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Validation report saved → %s", report_path)
