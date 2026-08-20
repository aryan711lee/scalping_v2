import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_STORAGE_DIR = BASE_DIR / "data_storage"
RAW_DIR = DATA_STORAGE_DIR / "raw"
PROCESSED_DIR = DATA_STORAGE_DIR / "processed"
FEATURES_DIR = DATA_STORAGE_DIR / "features"
LOGS_DIR = BASE_DIR / "logs"

# NIFTY index symbol (used as market context feature input, never as trading target)
NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"
NIFTY_SAFE = "NSE_NIFTY50_INDEX"

# Fyers credentials
FYERS_APP_ID = os.getenv("FYERS_APP_ID", "")
FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "")
FYERS_ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN", "")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data_storage' / 'scalping_v2.db'}")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "system.log"))

# Data settings
HISTORICAL_DATA_START = os.getenv("HISTORICAL_DATA_START", "2024-01-01")
PRIMARY_TIMEFRAME = os.getenv("PRIMARY_TIMEFRAME", "3min")

WATCHLIST = [
    "NSE:RELIANCE-EQ",
    "NSE:TCS-EQ",
    "NSE:INFY-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:SBIN-EQ",
    "NSE:WIPRO-EQ",
    "NSE:LT-EQ",
    "NSE:KOTAKBANK-EQ",  # TATAMOTORS replaced — Fyers returns -300 invalid symbol for that ticker
    "NSE:AXISBANK-EQ",
    "NSE:BAJFINANCE-EQ",
    "NSE:MARUTI-EQ",
    "NSE:ASIANPAINT-EQ",
    "NSE:TITAN-EQ",
    "NSE:SUNPHARMA-EQ",
]

TIMEFRAMES = {
    "1min": "1",
    "3min": "3",
    "5min": "5",
}

# Paper trading / backtest config
PAPER_CAPITAL = float(os.getenv("PAPER_CAPITAL", "50000"))
INTRADAY_LEVERAGE = float(os.getenv("INTRADAY_LEVERAGE", "5"))

SIZING_MODE = os.getenv("SIZING_MODE", "fixed_pct")
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "20")) / 100.0

MAX_TRADES_PER_SYMBOL_PER_DAY = int(os.getenv("MAX_TRADES_PER_SYMBOL_PER_DAY", "3"))
TARGET_PCT = float(os.getenv("TARGET_PCT", "0.40")) / 100.0
STOPLOSS_PCT = float(os.getenv("STOPLOSS_PCT", "0.20")) / 100.0
TIME_EXIT_MINUTE = int(os.getenv("TIME_EXIT_MINUTE", "915"))  # minutes from midnight

# Market session in IST
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

# API behaviour
API_RETRY_COUNT = 3
API_RETRY_BACKOFF = 1.0  # seconds, doubles on each retry
API_RATE_LIMIT_SLEEP = 0.5  # seconds between bulk requests
