import os
from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    """A setting in the environment or .env file is invalid. Raised at start-up, never ignored."""


def _env_int(name, default=0):
    raw = (os.getenv(name) or "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number (got {raw!r})") from None


def _env_choice(name, default, allowed):
    """Missing or empty -> the safe default. Anything else must be one of `allowed` (case-insensitive)."""
    raw = (os.getenv(name) or "").strip().upper()
    if raw == "":
        return default
    if raw not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)} (got {raw!r})")
    return raw


# MT5 credentials (read from .env, never hard-coded)
MT5_LOGIN = _env_int("MT5_LOGIN", 0)
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trading_bot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Bot settings
# Only SIGNAL_ONLY exists today. Order-placing modes must be added here deliberately, in the
# release that builds the execution gateway, so a typo or stray value can never enable trading.
ALLOWED_EXECUTION_MODES = ("SIGNAL_ONLY",)
EXECUTION_MODE = _env_choice("EXECUTION_MODE", "SIGNAL_ONLY", ALLOWED_EXECUTION_MODES)
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
MT5_EXPECTED_MODE = _env_choice("MT5_EXPECTED_MODE", "DEMO", ("DEMO", "REAL", "CONTEST"))
