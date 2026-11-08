import math
import time
from dataclasses import dataclass, field

import pandas as pd

from backend import config
from backend.risk.models import SymbolSpec

TIMEFRAME_NAMES = ("M15", "H1", "H4", "D1")


# ------------------------------------------------------------------ errors
class MT5Error(Exception):
    """Base class for every MT5 safety failure."""


class MT5ConnectionError(MT5Error):
    """Terminal not running, not connected to the broker, or request timed out."""


class MT5AccountError(MT5Error):
    """Account missing, wrong login, wrong server, or wrong account type."""


class MT5PermissionError(MT5Error):
    """Trading is not permitted by the terminal or the account."""


class MT5SymbolError(MT5Error):
    """Symbol missing or its broker specification is incomplete or unsafe."""


class MT5RequestError(MT5Error):
    """A request to MT5 failed after the allowed attempts, or returned bad data."""


@dataclass
class PermissionReport:
    trading_ok: bool
    issues: list = field(default_factory=list)


def _default_api():
    import MetaTrader5 as api          # imported lazily so tests can run with a fake
    return api


def _positive(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)


# ------------------------------------------------------------------ symbol validation
def validate_symbol_info(info, api=None):
    """Return a list of problems with a broker symbol record (empty list = complete and safe)."""
    if info is None:
        return ["symbol not found at the broker"]
    problems = []
    required = ["point", "trade_tick_size", "trade_tick_value", "trade_contract_size",
                "volume_min", "volume_max", "volume_step", "digits", "trade_stops_level", "trade_mode"]
    for name in required:
        if not hasattr(info, name):
            problems.append(f"missing field: {name}")
    if problems:
        return problems

    tick_value = getattr(info, "trade_tick_value_loss", 0) or info.trade_tick_value
    for name, value in (("point", info.point), ("trade_tick_size", info.trade_tick_size),
                        ("trade_tick_value", tick_value), ("trade_contract_size", info.trade_contract_size),
                        ("volume_min", info.volume_min), ("volume_max", info.volume_max),
                        ("volume_step", info.volume_step)):
        if not _positive(value):
            problems.append(f"{name} is missing, zero or invalid ({value!r})")
    if not problems:
        if info.volume_min > info.volume_max:
            problems.append("volume_min is larger than volume_max")
        ratio = info.volume_min / info.volume_step
        if abs(ratio - round(ratio)) > 1e-6:
            problems.append("volume_min is not a multiple of volume_step")
    if not (isinstance(info.digits, int) and info.digits >= 0):
        problems.append(f"digits is invalid ({info.digits!r})")
    if not (isinstance(info.trade_stops_level, (int, float)) and info.trade_stops_level >= 0):
        problems.append(f"trade_stops_level is invalid ({info.trade_stops_level!r})")
    full = getattr(api, "SYMBOL_TRADE_MODE_FULL", 4) if api is not None else 4
    if info.trade_mode != full:
        problems.append(f"trading on this symbol is not fully enabled (trade_mode={info.trade_mode})")
    return problems


def validate_candles(df):
    """Reject clearly corrupt candle data. Returns the frame unchanged when it looks sane."""
    need = ["time", "open", "high", "low", "close"]
    if any(c not in df.columns for c in need):
        raise MT5RequestError("candle data is missing required columns")
    if df[need].isna().any().any():
        raise MT5RequestError("candle data contains missing values")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise MT5RequestError("candle data contains non-positive prices")
    if (df["high"] < df["low"]).any():
        raise MT5RequestError("candle data has a high below its low")
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        raise MT5RequestError("candle times are not strictly increasing")
    return df


# ------------------------------------------------------------------ connector
class MT5Connector:
    """
    Safe wrapper around the MetaTrader5 package.

    * Verifies login, server and account type after EVERY connection or login.
    * Verifies terminal and account trading permissions before any trading use.
    * Retries only read requests. Order requests (idempotent=False) are never retried,
      so a timeout can never create a duplicate order.
    * On failure it shuts the connection down and, if configured, trips the kill switch.
    """

    def __init__(self, api=None, expected_login=None, expected_server=None, expected_mode=None,
                 path=None, password=None, symbols=None, retries=3, backoff_seconds=1.0,
                 sleep=time.sleep, kill_switch=None, kill_on_failure=False):
        self.api = api if api is not None else _default_api()
        self.expected_login = expected_login if expected_login is not None else getattr(config, "MT5_LOGIN", 0)
        self.expected_server = expected_server if expected_server is not None else getattr(config, "MT5_SERVER", "")
        self.expected_mode = (expected_mode or getattr(config, "MT5_EXPECTED_MODE", "DEMO")).upper()
        self.path = path if path is not None else getattr(config, "MT5_PATH", None)
        self.password = password if password is not None else getattr(config, "MT5_PASSWORD", "")
        self.symbols = symbols if symbols is not None else dict(getattr(config, "SYMBOLS", {}))
        self.retries = max(1, int(retries))
        self.backoff = backoff_seconds
        self.sleep = sleep
        self.kill = kill_switch
        self.kill_on_failure = kill_on_failure
        self.connected = False
        self.last_permissions = None

    # ----- failure handling
    def _shutdown_quietly(self):
        self.connected = False
        try:
            self.api.shutdown()
        except Exception:
            pass

    def _fail(self, exc):
        """Shut down, optionally trip the kill switch, then raise."""
        self._shutdown_quietly()
        if self.kill is not None and self.kill_on_failure:
            try:
                self.kill.activate(f"MT5: {type(exc).__name__}: {exc}", source="mt5")
            except Exception:
                pass
        raise exc

    # ----- account checks
    def _mode_name(self, value):
        api = self.api
        names = {getattr(api, "ACCOUNT_TRADE_MODE_DEMO", 0): "DEMO",
                 getattr(api, "ACCOUNT_TRADE_MODE_CONTEST", 1): "CONTEST",
                 getattr(api, "ACCOUNT_TRADE_MODE_REAL", 2): "REAL"}
        return names.get(value, "UNKNOWN")

    def _check_configured(self):
        if not (isinstance(self.expected_login, int) and self.expected_login > 0):
            raise MT5AccountError("MT5_LOGIN is not configured; refusing to connect")
        if not str(self.expected_server).strip():
            raise MT5AccountError("MT5_SERVER is not configured; refusing to connect")
        if self.expected_mode not in ("DEMO", "REAL", "CONTEST"):
            raise MT5AccountError(f"expected account type is invalid ({self.expected_mode!r})")

    def account_problems(self, acct):
        if acct is None:
            return ["account information is missing"]
        login = getattr(acct, "login", None)
        server = getattr(acct, "server", None)
        if not login or not server:
            return ["account login or server is missing from the account information"]
        problems = []
        if login != self.expected_login:
            problems.append(f"wrong account: connected to {login}, expected {self.expected_login}")
        if str(server).strip().casefold() != str(self.expected_server).strip().casefold():
            problems.append(f"wrong server: connected to {server}, expected {self.expected_server}")
        mode = self._mode_name(getattr(acct, "trade_mode", None))
        if mode != self.expected_mode:
            problems.append(f"wrong account type: {mode}, expected {self.expected_mode}")
        return problems

    def check_permissions(self, term=None, acct=None):
        term = term if term is not None else self.api.terminal_info()
        acct = acct if acct is not None else self.api.account_info()
        issues = []
        if term is None or acct is None:
            issues.append("terminal or account information is unavailable")
        else:
            if getattr(term, "trade_allowed", False) is not True:
                issues.append("Algo Trading is disabled in the MT5 terminal (click the Algo Trading button)")
            if getattr(term, "tradeapi_disabled", True) is not False:
                issues.append("the MT5 terminal has its trading API disabled")
            if getattr(acct, "trade_allowed", False) is not True:
                issues.append("trading is not allowed on this account")
            if getattr(acct, "trade_expert", False) is not True:
                issues.append("automated (expert) trading is not allowed on this account")
        return PermissionReport(trading_ok=not issues, issues=issues)

    def _check_session(self, require_trading):
        term = self.api.terminal_info()
        if term is None:
            raise MT5ConnectionError("terminal information is unavailable")
        if not getattr(term, "connected", False):
            raise MT5ConnectionError("the MT5 terminal is not connected to the broker server")
        acct = self.api.account_info()
        problems = self.account_problems(acct)
        if problems:
            raise MT5AccountError("; ".join(problems))
        self.last_permissions = self.check_permissions(term, acct)
        if require_trading and not self.last_permissions.trading_ok:
            raise MT5PermissionError("; ".join(self.last_permissions.issues))
        return acct

    # ----- connect / disconnect
    def connect(self, require_trading=False):
        """Connect, log in if needed, then VERIFY account, server, type and permissions."""
        try:
            self._check_configured()
        except MT5Error as e:
            self._fail(e)

        last = None
        for attempt in range(self.retries):
            kwargs = {"timeout": 120000}
            if self.path:
                kwargs["path"] = self.path
            if self.api.initialize(**kwargs):
                break
            last = self.api.last_error()
            if attempt < self.retries - 1:
                self.sleep(self.backoff * (2 ** attempt))
        else:
            self._fail(MT5ConnectionError(f"MT5 initialize failed after {self.retries} attempts: {last}"))

        try:
            acct = self.api.account_info()
            if acct is None or getattr(acct, "login", None) != self.expected_login:
                if not self.password:
                    raise MT5AccountError("the terminal is not on the expected account and no password is configured")
                if not self.api.login(self.expected_login, password=self.password, server=self.expected_server):
                    raise MT5AccountError(f"login to the expected account failed: {self.api.last_error()}")
            acct = self._check_session(require_trading)
        except MT5Error as e:
            self._fail(e)
        self.connected = True
        return acct

    def disconnect(self):
        self._shutdown_quietly()

    def ensure_connected(self, require_trading=False):
        """Cheap re-check before every request; reconnects once with full verification if the link dropped."""
        try:
            return self._check_session(require_trading)
        except (MT5AccountError, MT5PermissionError) as e:
            self._fail(e)
        except MT5ConnectionError:
            pass
        return self.connect(require_trading)          # raises (and trips the kill switch) if it cannot recover

    # ----- requests
    def request(self, label, fn, *args, idempotent=True, validate=None, **kwargs):
        """
        Run an MT5 call safely. Read requests are retried with backoff. A request that
        changes state (idempotent=False, e.g. an order) is attempted ONCE and never retried.
        """
        attempts = self.retries if idempotent else 1
        last = None
        for i in range(attempts):
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                result, last = None, repr(exc)
            else:
                if result is not None and (validate is None or validate(result)):
                    return result
                last = self.api.last_error()
            if i < attempts - 1:
                self.sleep(self.backoff * (2 ** i))
        raise MT5RequestError(f"{label} failed after {attempts} attempt(s): {last}")

    # ----- symbols
    def broker_symbol(self, symbol):
        return self.symbols.get(symbol, symbol)

    def load_symbol_spec(self, symbol):
        """Load and validate a complete broker specification. Raises MT5SymbolError if anything is off."""
        self.ensure_connected()
        broker = self.broker_symbol(symbol)
        try:
            self.request("symbol_select", self.api.symbol_select, broker, True, validate=lambda r: r is True)
            info = self.request("symbol_info", self.api.symbol_info, broker)
        except MT5RequestError as e:
            raise MT5SymbolError(f"{symbol} ({broker}): {e}")
        problems = validate_symbol_info(info, self.api)
        if problems:
            raise MT5SymbolError(f"{symbol} ({broker}): " + "; ".join(problems))
        tick_value = getattr(info, "trade_tick_value_loss", 0) or info.trade_tick_value
        return SymbolSpec(symbol=symbol, point=float(info.point), tick_size=float(info.trade_tick_size),
                          tick_value=float(tick_value), volume_min=float(info.volume_min),
                          volume_max=float(info.volume_max), volume_step=float(info.volume_step),
                          stops_level_points=int(info.trade_stops_level))

    def load_all_specs(self, symbols=None):
        specs, errors = {}, {}
        for name in (symbols or list(self.symbols)):
            try:
                specs[name] = self.load_symbol_spec(name)
            except MT5Error as e:
                errors[name] = str(e)
        return specs, errors

    # ----- candles
    def get_candles(self, symbol, timeframe="H1", count=1000):
        """Last `count` COMPLETED candles (start position 1 skips the forming candle)."""
        if timeframe not in TIMEFRAME_NAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.ensure_connected()
        broker = self.broker_symbol(symbol)
        try:
            self.request("symbol_select", self.api.symbol_select, broker, True, validate=lambda r: r is True)
            tf = getattr(self.api, f"TIMEFRAME_{timeframe}")
            rates = self.request("copy_rates_from_pos", self.api.copy_rates_from_pos, broker, tf, 1, count,
                                 validate=lambda r: len(r) > 0)
        except MT5RequestError as e:
            raise MT5RequestError(f"{symbol}: {e}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
        return validate_candles(df)


# ------------------------------------------------------------------ module-level helpers
# These keep the earlier scripts (candle_manager, tests/check_mt5.py) working unchanged.
_default = None


def _connector():
    global _default
    if _default is None:
        _default = MT5Connector()
    return _default


def connect(require_trading=False):
    return _connector().connect(require_trading)


def disconnect():
    _connector().disconnect()


def get_candles(symbol, timeframe="H1", count=1000):
    return _connector().get_candles(symbol, timeframe, count)


def load_symbol_spec(symbol):
    return _connector().load_symbol_spec(symbol)


def load_all_specs(symbols=None):
    return _connector().load_all_specs(symbols)
