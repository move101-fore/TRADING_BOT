import math

from backend.risk.models import RiskConfig, SymbolSpec, finite_positive


def validate_levels(direction, entry, stop, target, spread, atr, spec: SymbolSpec, cfg: RiskConfig):
    """
    Validate stop-loss, take-profit, reward/risk and spread conditions.
    Returns (codes, stop_distance, reward_risk). An empty codes list means OK.
    """
    if direction not in (1, -1) or not finite_positive(entry, stop, target):
        return ["INVALID_INPUT"], 0.0, 0.0
    if not (isinstance(spread, (int, float)) and math.isfinite(spread) and spread >= 0):
        return ["INVALID_INPUT"], 0.0, 0.0

    codes = []
    if direction == 1:
        if not stop < entry:
            codes.append("STOP_WRONG_SIDE")
        if not target > entry:
            codes.append("TP_WRONG_SIDE")
    else:
        if not stop > entry:
            codes.append("STOP_WRONG_SIDE")
        if not target < entry:
            codes.append("TP_WRONG_SIDE")
    if codes:
        return codes, 0.0, 0.0

    stop_dist = abs(entry - stop)
    reward_dist = abs(target - entry)
    rr = reward_dist / stop_dist

    min_dist = spec.stops_level_points * spec.point
    if stop_dist < min_dist:
        codes.append("STOP_TOO_CLOSE")
    if reward_dist < min_dist:
        codes.append("TP_TOO_CLOSE")
    if spread > 0 and stop_dist < cfg.min_stop_spread_mult * spread:
        codes.append("STOP_TIGHT_VS_SPREAD")
    if atr is not None and math.isfinite(atr) and atr > 0 and stop_dist > cfg.max_stop_atr_mult * atr:
        codes.append("STOP_TOO_WIDE")
    if rr + 1e-9 < cfg.min_reward_risk:
        codes.append("RR_TOO_LOW")
    if spread > cfg.max_spread_frac_of_stop * stop_dist:
        codes.append("SPREAD_TOO_WIDE")
    limit_pts = cfg.max_spread_points.get(spec.symbol)
    if limit_pts is not None and spread / spec.point > limit_pts:
        if "SPREAD_TOO_WIDE" not in codes:
            codes.append("SPREAD_TOO_WIDE")
    return codes, stop_dist, rr
