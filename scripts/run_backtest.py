"""
Run a backtest against historical feature files.

Usage:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --strategy baseline
  python scripts/run_backtest.py --timeframe 3min
  python scripts/run_backtest.py --symbol NSE:RELIANCE-EQ
  python scripts/run_backtest.py --from 2024-01-01 --to 2024-12-31
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from app import config
from backtester.engine import BacktestEngine
from backtester.metrics import calculate_metrics
from backtester.portfolio import Portfolio
from backtester.report import generate_report
from strategy.baseline import BaselineStrategy


STRATEGY_MAP = {
    "baseline": BaselineStrategy,
}


def symbol_to_filename_prefix(symbol: str, timeframe: str) -> str:
    """Convert 'NSE:RELIANCE-EQ' + '3min' → 'NSE_RELIANCE_EQ_3min_features'."""
    safe = symbol.replace(":", "_").replace("-", "_")
    return f"{safe}_{timeframe}_features"


def load_feature_files(
    symbols: list[str],
    timeframe: str,
    features_dir: Path,
    date_from: str | None,
    date_to: str | None,
) -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in symbols:
        prefix = symbol_to_filename_prefix(symbol, timeframe)
        path = features_dir / f"{prefix}.parquet"
        if not path.exists():
            print(f"  [WARN] Feature file not found: {path}")
            continue
        df = pd.read_parquet(path)

        # Ensure datetime column is properly typed
        if "datetime" not in df.columns:
            print(f"  [WARN] No 'datetime' column in {path}")
            continue

        df["datetime"] = pd.to_datetime(df["datetime"], utc=False)

        if date_from:
            df = df[df["datetime"].dt.date >= pd.to_datetime(date_from).date()]
        if date_to:
            df = df[df["datetime"].dt.date <= pd.to_datetime(date_to).date()]

        df = df.reset_index(drop=True)
        if df.empty:
            print(f"  [WARN] No data in range for {symbol}")
            continue

        data[symbol] = df
        print(f"  Loaded {symbol}: {len(df):,} rows")

    return data


def main():
    parser = argparse.ArgumentParser(description="Run backtest on historical feature files")
    parser.add_argument("--strategy", default="baseline", choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--timeframe", default=config.PRIMARY_TIMEFRAME)
    parser.add_argument("--symbol", default=None, help="Single symbol (e.g. NSE:RELIANCE-EQ)")
    parser.add_argument("--from", dest="date_from", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else config.WATCHLIST
    timeframe = args.timeframe

    print(f"\nBacktest: {args.strategy} | {timeframe} | {len(symbols)} symbol(s)")
    print(f"Capital: Rs.{config.PAPER_CAPITAL:,.0f} | Leverage: {config.INTRADAY_LEVERAGE:.0f}x")
    print(f"Sizing: {config.SIZING_MODE} | Target: {config.TARGET_PCT*100:.2f}% | Stop: {config.STOPLOSS_PCT*100:.2f}%\n")

    # Load data
    data = load_feature_files(
        symbols=symbols,
        timeframe=timeframe,
        features_dir=config.FEATURES_DIR,
        date_from=args.date_from,
        date_to=args.date_to,
    )

    if not data:
        print("No feature data found. Aborting.")
        sys.exit(1)

    # Determine actual date range
    all_dates = []
    for df in data.values():
        all_dates.append(df["datetime"].min())
        all_dates.append(df["datetime"].max())
    date_range = (
        min(all_dates).strftime("%Y-%m-%d"),
        max(all_dates).strftime("%Y-%m-%d"),
    )

    # Set up portfolio
    portfolio = Portfolio(
        capital=config.PAPER_CAPITAL,
        leverage=config.INTRADAY_LEVERAGE,
        sizing_mode=config.SIZING_MODE,
        position_size_pct=config.POSITION_SIZE_PCT,
    )

    # Set up strategy
    strategy_cls = STRATEGY_MAP[args.strategy]
    strategy = strategy_cls()
    strategy_name = f"{args.strategy.title()} Strategy (Model Zero)"

    # Run engine
    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        target_pct=config.TARGET_PCT,
        stoploss_pct=config.STOPLOSS_PCT,
        time_exit_minute=config.TIME_EXIT_MINUTE,
        max_trades_per_symbol_per_day=config.MAX_TRADES_PER_SYMBOL_PER_DAY,
    )
    print("Running backtest...")
    engine.run(data)

    # Calculate metrics
    metrics = calculate_metrics(portfolio.closed_trades, config.PAPER_CAPITAL)

    # Generate report
    generate_report(
        metrics=metrics,
        trades=portfolio.closed_trades,
        strategy_name=strategy_name,
        timeframe=timeframe,
        symbols=list(data.keys()),
        date_range=date_range,
        capital=config.PAPER_CAPITAL,
        leverage=config.INTRADAY_LEVERAGE,
        sizing_mode=config.SIZING_MODE,
        position_size_pct=config.POSITION_SIZE_PCT,
        logs_dir=config.LOGS_DIR,
    )


if __name__ == "__main__":
    main()
