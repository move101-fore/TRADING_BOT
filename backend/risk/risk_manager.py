import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from backend.risk.kill_switch import KillSwitch
from backend.risk.models import (
    AccountState, Decision, MonitorResult, OpenPosition, RiskConfig, SymbolSpec,
    TradeProposal, finite_positive, REASON_TEXT,
)
from backend.risk.position_sizing import lots_for_risk, risk_money
from backend.risk.stop_loss import validate_levels

LOG_COLUMNS = ["time", "symbol", "direction", "strategy", "entry", "stop_loss",
               "take_profit", "approved", "lots", "risk_amount", "risk_pct", "codes", "reasons"]


class RiskEventLog:
    """Appends one CSV row for every approved or rejected trade."""

    def __init__(self, path=Path("data") / "risk_events.csv"):
        self.path = Path(path)

    def record(self, proposal, decision):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_COLUMNS)
            w.writerow([
                datetime.now(timezone.utc).isoformat(), proposal.symbol, proposal.direction,
                proposal.strategy, proposal.entry, proposal.stop_loss, proposal.take_profit,
                decision.approved, decision.lots, round(decision.risk_amount, 2),
                round(decision.risk_pct, 4), "|".join(decision.codes), " ".join(decision.reasons),
            ])


def currency_legs(symbol, direction):
    """Which currencies a position leans on. Gold is tracked on its own leg only."""
    s = symbol.upper()
    base, quote = s[:3], s[3:6]
    if base == "XAU":
        return {"XAU": direction}
    return {base: direction, quote: -direction}


class RiskManager:
    """Single gate that every order must pass. Fails closed on bad input."""

    def __init__(self, config: RiskConfig, kill_switch: KillSwitch, log: RiskEventLog = None):
        self.cfg = config
        self.kill = kill_switch
        self.log = log

    # ------------------------------------------------------------------ helpers
    def _account_problems(self, a: AccountState):
        c = self.cfg
        if not finite_positive(a.balance, a.equity, a.day_start_equity, a.peak_equity):
            return ["INVALID_INPUT"]
        if not (isinstance(a.data_age_seconds, (int, float)) and math.isfinite(a.data_age_seconds)):
            return ["INVALID_INPUT"]
        if a.data_age_seconds > c.max_data_age_seconds:
            return ["STALE_DATA"]
        return []

    def _spec_problems(self, spec: SymbolSpec):
        ok = finite_positive(spec.point, spec.tick_size, spec.tick_value,
                             spec.volume_min, spec.volume_max, spec.volume_step)
        return [] if ok else ["INVALID_INPUT"]

    def _breaches(self, a: AccountState):
        """Hard limits. Returns list of (code, close_all)."""
        c = self.cfg
        out = []
        base = min(a.balance, a.equity)                 # floating losses count
        if base < c.equity_floor:
            out.append(("EQUITY_FLOOR_BREACHED", c.close_all_on_floor))
        day_loss = max(0.0, a.day_start_equity - a.equity)
        if day_loss >= c.max_daily_loss_pct / 100.0 * a.day_start_equity - 1e-9:
            out.append(("DAILY_LOSS_LIMIT_REACHED", c.close_all_on_daily_loss))
        dd = (a.peak_equity - a.equity) / a.peak_equity * 100.0
        if dd >= c.max_drawdown_pct - 1e-9:
            out.append(("MAX_DRAWDOWN_REACHED", c.close_all_on_drawdown))
        return out

    def _lock(self, breaches):
        codes = [b[0] for b in breaches]
        self.kill.activate("AUTO: " + ", ".join(codes), source="risk_manager")
        return codes

    # ------------------------------------------------------------------ monitor
    def monitor(self, account: AccountState, positions=None):
        """
        Call regularly (for example every few seconds). Locks trading when a hard
        limit is hit and tells the execution layer what to do. This module does not
        close trades itself; it returns CLOSE_ALL_POSITIONS for the executor to act on.
        """
        problems = self._account_problems(account)
        if problems:
            self.kill.activate("AUTO: " + problems[0], source="risk_manager")
            return MonitorResult(["BLOCK_NEW_ORDERS"], problems)
        breaches = self._breaches(account)
        if not breaches:
            return MonitorResult([], [])
        codes = self._lock(breaches)
        actions = ["BLOCK_NEW_ORDERS"]
        if any(b[1] for b in breaches):
            actions.append("CLOSE_ALL_POSITIONS")
        return MonitorResult(actions, codes)

    # ----------------------------------------------------------------- evaluate
    def evaluate(self, p: TradeProposal, account: AccountState, positions, spec: SymbolSpec):
        decision = self._evaluate(p, account, list(positions or []), spec)
        if self.log is not None:
            self.log.record(p, decision)
        return decision

    def _reject(self, codes):
        return Decision(approved=False, codes=list(codes))

    def _evaluate(self, p, account, positions, spec):
        c = self.cfg
        if self.kill.is_active():
            return self._reject(["KILL_SWITCH_ACTIVE"])

        problems = self._account_problems(account) + self._spec_problems(spec)
        if p.symbol != spec.symbol:
            problems.append("INVALID_INPUT")
        if problems:
            return self._reject(dict.fromkeys(problems))

        breaches = self._breaches(account)
        if breaches:
            return self._reject(self._lock(breaches))

        codes = []
        base = min(account.balance, account.equity)

        if any(pos.symbol == p.symbol for pos in positions):
            codes.append("DUPLICATE_POSITION")
        if len(positions) >= c.max_open_positions:
            codes.append("MAX_POSITIONS")

        level_codes, stop_dist, rr = validate_levels(
            p.direction, p.entry, p.stop_loss, p.take_profit, p.spread, p.atr, spec, c)
        if "INVALID_INPUT" in level_codes:
            return self._reject(level_codes)
        codes += level_codes

        lots, risk_amt = 0.0, 0.0
        if stop_dist > 0:
            lots = lots_for_risk(base, c.risk_per_trade_pct, stop_dist, spec)
            if lots < spec.volume_min - 1e-12:
                codes.append("LOT_BELOW_MIN")
                lots = 0.0
            else:
                risk_amt = risk_money(stop_dist, lots, spec)
                if risk_amt > base * c.risk_per_trade_pct / 100.0 * (1 + 1e-9):
                    codes.append("RISK_EXCEEDS_CAP")

        if risk_amt > 0:
            open_risk = sum(max(0.0, pos.risk_amount) for pos in positions)
            day_loss = max(0.0, account.day_start_equity - account.equity)
            daily_limit = c.max_daily_loss_pct / 100.0 * account.day_start_equity
            if day_loss + open_risk + risk_amt > daily_limit + 1e-9:
                codes.append("DAILY_LOSS_LIMIT")
            if open_risk + risk_amt > c.max_total_risk_pct / 100.0 * base + 1e-9:
                codes.append("TOTAL_RISK_CAP")

        new_legs = currency_legs(p.symbol, p.direction)
        for cur, side in new_legs.items():
            same = sum(1 for pos in positions
                       if currency_legs(pos.symbol, pos.direction).get(cur) == side)
            if same + 1 > c.max_same_currency_positions:
                codes.append("CURRENCY_EXPOSURE")
                break

        codes = list(dict.fromkeys(codes))
        if codes:
            return self._reject(codes)
        return Decision(approved=True, lots=lots, risk_amount=risk_amt,
                        risk_pct=risk_amt / base * 100.0, codes=[])
