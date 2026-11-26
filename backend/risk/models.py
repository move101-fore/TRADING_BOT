import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Optional

REASON_TEXT = {
    "KILL_SWITCH_ACTIVE": "The global kill switch is active; no new orders are allowed.",
    "INVALID_INPUT": "A required input is missing or invalid, so the trade was rejected (fail closed).",
    "STALE_DATA": "Account data is too old to trust.",
    "EQUITY_FLOOR_BREACHED": "Equity fell below the minimum equity floor.",
    "DAILY_LOSS_LIMIT_REACHED": "The daily loss limit has been reached.",
    "DAILY_LOSS_LIMIT": "This trade would take today's potential loss past the daily limit.",
    "MAX_DRAWDOWN_REACHED": "The maximum account drawdown has been reached.",
    "DUPLICATE_POSITION": "A position on this symbol is already open.",
    "MAX_POSITIONS": "The maximum number of open positions is reached.",
    "TOTAL_RISK_CAP": "Total open risk would exceed the portfolio risk cap.",
    "CURRENCY_EXPOSURE": "Too many open positions already lean the same way on one currency.",
    "STOP_WRONG_SIDE": "The stop-loss is on the wrong side of the entry price.",
    "TP_WRONG_SIDE": "The take-profit is on the wrong side of the entry price.",
    "STOP_TOO_CLOSE": "The stop-loss is closer than the broker's minimum stop distance.",
    "TP_TOO_CLOSE": "The take-profit is closer than the broker's minimum stop distance.",
    "STOP_TIGHT_VS_SPREAD": "The stop-loss is too tight compared with the spread.",
    "STOP_TOO_WIDE": "The stop-loss is too wide compared with current volatility (ATR).",
    "RR_TOO_LOW": "Reward-to-risk is below the required minimum.",
    "SPREAD_TOO_WIDE": "The spread is too wide for this trade.",
    "LOT_BELOW_MIN": "Risking the allowed amount gives a lot size below the broker minimum.",
    "RISK_EXCEEDS_CAP": "The rounded lot size would risk more than the per-trade cap.",
    "INTERNAL_ERROR": "The risk check hit an unexpected error, so the trade was rejected (fail closed).",
    "AUDIT_LOG_FAILED": "The decision could not be written to the audit log, so approval was withheld.",
    "KILL_SWITCH_WRITE_FAILED": "The kill switch could not be saved; stop all trading manually.",
}


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    point: float
    tick_size: float
    tick_value: float          # account-currency value of ONE tick for ONE lot
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int = 0  # broker's minimum stop distance, in points


@dataclass(frozen=True)
class RiskConfig:
    """Immutable, fully validated risk settings. Bad values raise ValueError at construction."""
    reference_balance: float   # REQUIRED: the balance the equity floor is measured from
    floor_pct_of_reference: float = 90.0
    risk_per_trade_pct: float = 0.5
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 3
    max_total_risk_pct: float = 1.5
    max_drawdown_pct: float = 10.0
    min_reward_risk: float = 2.0
    max_same_currency_positions: int = 2
    min_stop_spread_mult: float = 3.0
    max_stop_atr_mult: float = 4.0
    max_spread_frac_of_stop: float = 0.25
    max_spread_points: dict = field(default_factory=dict)  # per-symbol limit, in points
    max_data_age_seconds: float = 60.0
    close_all_on_floor: bool = True
    close_all_on_daily_loss: bool = False
    close_all_on_drawdown: bool = False

    def __post_init__(self):
        def number(name, lo, hi, lo_incl=False, hi_incl=True):
            v = getattr(self, name)
            ok = (finite_number(v)
                  and (v >= lo if lo_incl else v > lo)
                  and (v <= hi if hi_incl else v < hi))
            if not ok:
                raise ValueError(f"{name} must be in {'[' if lo_incl else '('}{lo}, {hi}"
                                 f"{']' if hi_incl else ')'} (got {v!r})")

        def count(name):
            v = getattr(self, name)
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                raise ValueError(f"{name} must be a whole number of at least 1 (got {v!r})")

        if not finite_positive(self.reference_balance):
            raise ValueError("reference_balance must be positive")
        number("floor_pct_of_reference", 0, 100, hi_incl=False)
        number("risk_per_trade_pct", 0, 2)
        number("max_daily_loss_pct", 0, 10)
        number("max_total_risk_pct", 0, 10)
        number("max_drawdown_pct", 0, 100, hi_incl=False)
        number("min_reward_risk", 0, 100)
        number("min_stop_spread_mult", 0, 100, lo_incl=True)
        number("max_stop_atr_mult", 0, 100)
        number("max_spread_frac_of_stop", 0, 1)
        number("max_data_age_seconds", 0, 86400)
        count("max_open_positions")
        count("max_same_currency_positions")
        if self.max_total_risk_pct < self.risk_per_trade_pct:
            raise ValueError("max_total_risk_pct must be at least risk_per_trade_pct")
        for name in ("close_all_on_floor", "close_all_on_daily_loss", "close_all_on_drawdown"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False")
        msp = self.max_spread_points
        if not isinstance(msp, dict) or not all(
                isinstance(k, str) and k and finite_positive(v) for k, v in msp.items()):
            raise ValueError("max_spread_points must map symbol names to positive numbers")
        object.__setattr__(self, "max_spread_points", MappingProxyType(dict(msp)))

    @property
    def equity_floor(self):
        return self.reference_balance * self.floor_pct_of_reference / 100.0


@dataclass
class AccountState:
    balance: float
    equity: float
    day_start_equity: float
    peak_equity: float
    data_age_seconds: float   # REQUIRED: no default, so missing data can never look fresh


@dataclass
class OpenPosition:
    symbol: str
    direction: int              # +1 long, -1 short
    volume: float
    entry: float
    stop_loss: float
    risk_amount: float          # money at risk if the stop is hit (0 if stop is at/after breakeven)


@dataclass
class TradeProposal:
    symbol: str
    direction: int              # +1 buy, -1 sell
    entry: float
    stop_loss: float
    take_profit: float
    spread: float = 0.0         # in price units
    atr: Optional[float] = None
    strategy: str = ""


@dataclass
class Decision:
    approved: bool
    lots: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    codes: list = field(default_factory=list)

    @property
    def reasons(self):
        return [REASON_TEXT.get(c, c) for c in self.codes]


@dataclass
class MonitorResult:
    actions: list = field(default_factory=list)   # BLOCK_NEW_ORDERS / CLOSE_ALL_POSITIONS
    codes: list = field(default_factory=list)


def finite_number(*values):
    """True only for real, finite numbers. Booleans, NaN, infinity, strings and None are refused."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in values)


def finite_positive(*values):
    return all(finite_number(v) and v > 0 for v in values)
