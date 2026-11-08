import math
from dataclasses import dataclass, field
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


@dataclass
class RiskConfig:
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
        if not (self.reference_balance > 0):
            raise ValueError("reference_balance must be positive")
        if not (0 < self.risk_per_trade_pct <= 2):
            raise ValueError("risk_per_trade_pct must be in (0, 2]")
        if not (0 < self.max_daily_loss_pct <= 10):
            raise ValueError("max_daily_loss_pct must be in (0, 10]")
        if not (0 < self.floor_pct_of_reference < 100):
            raise ValueError("floor_pct_of_reference must be in (0, 100)")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")

    @property
    def equity_floor(self):
        return self.reference_balance * self.floor_pct_of_reference / 100.0


@dataclass
class AccountState:
    balance: float
    equity: float
    day_start_equity: float
    peak_equity: float
    data_age_seconds: float = 0.0


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


def finite_positive(*values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values)
