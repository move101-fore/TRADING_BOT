import math

from backend.risk.models import SymbolSpec


def round_down_to_step(value, step):
    """Round DOWN to a multiple of step (never up, so the risk cap is never exceeded)."""
    if not (math.isfinite(value) and step > 0):
        return 0.0
    return round(math.floor(value / step + 1e-9) * step, 8)


def risk_per_lot(stop_distance, spec: SymbolSpec):
    """Money lost on ONE lot if the stop is hit (price distance -> ticks -> tick value)."""
    return (stop_distance / spec.tick_size) * spec.tick_value


def risk_money(stop_distance, lots, spec: SymbolSpec):
    return risk_per_lot(stop_distance, spec) * lots


def lots_for_risk(equity_base, risk_pct, stop_distance, spec: SymbolSpec):
    """
    Largest allowed lot size such that the loss at the stop is <= risk_pct of
    equity_base, rounded DOWN to the broker's volume step and capped at the
    broker's maximum. Returns 0.0 when it cannot be computed safely.
    """
    per_lot = risk_per_lot(stop_distance, spec)
    if not (math.isfinite(per_lot) and per_lot > 0 and equity_base > 0):
        return 0.0
    budget = equity_base * risk_pct / 100.0
    lots = round_down_to_step(budget / per_lot, spec.volume_step)
    cap = round_down_to_step(spec.volume_max, spec.volume_step)
    return min(lots, cap)
