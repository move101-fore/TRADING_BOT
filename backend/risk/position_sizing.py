import math
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

from backend.risk.models import SymbolSpec


def round_down_to_step(value, step):
    """Round down conservatively to a volume-step multiple."""
    if isinstance(value, bool) or isinstance(step, bool):
        return 0.0
    if not isinstance(value, (int, float)) or not isinstance(step, (int, float)):
        return 0.0

    try:
        if not (math.isfinite(value) and math.isfinite(step) and step > 0):
            return 0.0

        value_decimal = Decimal(str(value))
        step_decimal = Decimal(str(step))
        if not value_decimal.is_finite() or not step_decimal.is_finite():
            return 0.0

        units = (value_decimal / step_decimal).to_integral_value(
            rounding=ROUND_FLOOR
        )
        return float(units * step_decimal)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return 0.0


def risk_per_lot(stop_distance, spec: SymbolSpec):
    """Money lost on ONE lot if the stop is hit (price distance -> ticks -> tick value)."""
    return (stop_distance / spec.tick_size) * spec.tick_value


def risk_money(stop_distance, lots, spec: SymbolSpec):
    return risk_per_lot(stop_distance, spec) * lots


def lots_for_risk(equity_base, risk_pct, stop_distance, spec: SymbolSpec):
    """
    Calculate position size using decimal arithmetic, round down to the
    broker volume step, and never exceed the risk budget or broker maximum.
    Returns 0.0 when inputs cannot be processed safely.
    """
    values = (
        equity_base, risk_pct, stop_distance,
        spec.tick_size, spec.tick_value,
        spec.volume_step, spec.volume_max,
    )
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           for v in values):
        return 0.0

    try:
        if any(not math.isfinite(v) for v in values):
            return 0.0
        if (equity_base <= 0 or risk_pct <= 0 or stop_distance <= 0
                or spec.tick_size <= 0 or spec.tick_value <= 0
                or spec.volume_step <= 0 or spec.volume_max <= 0):
            return 0.0

        equity = Decimal(str(equity_base))
        risk = Decimal(str(risk_pct))
        stop = Decimal(str(stop_distance))
        tick_size = Decimal(str(spec.tick_size))
        tick_value = Decimal(str(spec.tick_value))
        step = Decimal(str(spec.volume_step))
        maximum = Decimal(str(spec.volume_max))

        budget = equity * risk / Decimal("100")
        loss_per_lot = (stop / tick_size) * tick_value

        if budget <= 0 or loss_per_lot <= 0:
            return 0.0

        raw_lots = budget / loss_per_lot
        capped_lots = min(raw_lots, maximum)

        units = (capped_lots / step).to_integral_value(
            rounding=ROUND_FLOOR
        )
        result = units * step
        return float(result)

    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return 0.0
