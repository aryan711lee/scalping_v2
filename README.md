# Scalping V2 — Personal Quantitative Research System

Intraday scalping research system for Indian equity markets (NSE), powered by Fyers API.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in your Fyers credentials
```

Required variables:

| Variable | Description |
|---|---|
| `FYERS_APP_ID` | Your Fyers app client ID |
| `FYERS_SECRET_KEY` | Your Fyers secret key |
| `FYERS_REDIRECT_URI` | OAuth redirect URI registered in Fyers |
| `FYERS_ACCESS_TOKEN` | Active access token (refresh daily) |
| `DATABASE_URL` | SQLite path (default: `sqlite:///data_storage/scalping_v2.db`) |
| `HISTORICAL_DATA_START` | Backfill start date (e.g. `2023-01-01`) |

### 3. Download historical data

```bash
python scripts/download_historical.py
```

Options:

```bash
# Single symbol
python scripts/download_historical.py --symbol NSE:RELIANCE-EQ

# Custom date range
python scripts/download_historical.py --from 2024-01-01 --to 2024-06-30
```

The script:
1. Downloads 1-min, 3-min, and 5-min candles for all 15 watchlist symbols
2. Saves raw CSVs to `data_storage/raw/`
3. Runs the cleaning pipeline (dedup, OHLC checks, gap fill)
4. Saves cleaned Parquet files to `data_storage/processed/`
5. Resamples 1-min data to 3-min and 5-min
6. Stores all candles to the SQLite database
7. Prints a data quality validation report

### 4. Run tests

```bash
pytest tests/ -v --cov=data --cov=database --cov=app
```

## Project Structure

```
scalping_v2/
├── app/            # Config and logging
├── data/           # Ingestion, cleaning, resampling, validation
├── database/       # SQLAlchemy models and repository
├── scripts/        # Standalone runnable scripts
├── tests/          # Unit tests
├── data_storage/   # raw/ and processed/ data (gitignored)
├── logs/           # Application logs (gitignored)
└── research/       # Notebooks and experiment logs
```

## Watchlist (15 symbols)

NSE:RELIANCE-EQ, NSE:TCS-EQ, NSE:INFY-EQ, NSE:HDFCBANK-EQ, NSE:ICICIBANK-EQ,
NSE:SBIN-EQ, NSE:WIPRO-EQ, NSE:LT-EQ, NSE:TATAMOTORS-EQ, NSE:AXISBANK-EQ,
NSE:BAJFINANCE-EQ, NSE:MARUTI-EQ, NSE:ASIANPAINT-EQ, NSE:TITAN-EQ, NSE:SUNPHARMA-EQ

## Timeframes

| Label | Resolution | Use |
|---|---|---|
| 1min | 1 minute | Base data for resampling |
| 3min | 3 minutes | Primary research timeframe |
| 5min | 5 minutes | Secondary research timeframe |

## Notes

- No real orders until Phase 8+. Paper trading only.
- `.env` is gitignored. Never commit credentials.
- `data_storage/` contents are gitignored (large files). Directory is committed via `.gitkeep`.
