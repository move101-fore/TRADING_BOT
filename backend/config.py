import os
from dotenv import load_dotenv

load_dotenv()

# MT5 credentials (read from .env, never hard-coded)
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0") or 0)
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trading_bot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Bot settings
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "SIGNAL_ONLY")
PRIMARY_TIMEFRAME = "H1"
API_PORT = 5000

# Map our standard names to your broker's names.
# If Exness uses a suffix (e.g. "EURUSDm"), change it here only.
SYMBOLS = {
    "EURUSD": "EURUSDm",
    "GBPUSD": "GBPUSDm",
    "USDJPY": "USDJPYm",
    "USDCHF": "USDCHFm",
    "AUDUSD": "AUDUSDm",
    "USDCAD": "USDCADm",
    "NZDUSD": "NZDUSDm",
    "XAUUSD": "XAUUSDm",
}

# Risk defaults
RISK_PER_TRADE_PCT = 0.5
MAX_DAILY_LOSS_PCT = 2.0
MAX_OPEN_POSITIONS = 3
MIN_REWARD_RISK = 2.0
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")
MT5_EXPECTED_MODE = os.getenv("MT5_EXPECTED_MODE", "DEMO").upper()
