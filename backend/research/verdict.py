import pandas as pd

from backend.research.event_study import HORIZONS

STATUS_FAIL = "FAIL"
STATUS_ELIGIBLE = "ELIGIBLE_FOR_ONE_VALIDATION_LOOK"
STATUS_PASS = "PASS_DEVELOPMENT_AND_VALIDATION"


def row_passes(r, min_events=300, max_placebo_p=0.025):
    """One (kind, period, horizon) row: net AND excess lower bounds above 0, enough events, placebo p small."""
    ok = bool(r["net_lo"] > 0 and r["excess_lo"] > 0 and r["events"] >= min_events)
    p = r.get("placebo_p")
    if p is not None and not pd.isna(p):
        ok = ok and bool(p <= max_placebo_p)
    return ok


def passing_horizons(rows, **kw):
    return sorted(int(r["horizon"]) for _, r in rows.iterrows() if row_passes(r, **kw))


def verdict(results, need_horizons=2, min_halves=4, total_halves=6, **kw):
    """
    Apply the COMPLETE research-registry rule to a results table. One row per kind:
      development : same kind must pass at >= need_horizons of the 3 horizons
      validation  : (only if it was viewed) the same, plus at least one horizon passing in both periods
      half-years  : at the first horizon passing in both periods, excess positive in >= 4 of 6 half-year slices
    Without validation results the best possible status is ELIGIBLE_FOR_ONE_VALIDATION_LOOK.
    """
    rows = []
    for kind, g in results.groupby("kind"):
        dev = g[g["period"] == "development"]
        val = g[g["period"] == "validation"]
        reasons = []
        dev_ok = passing_horizons(dev, **kw)
        if len(dev_ok) < need_horizons:
            reasons.append(f"development: passes at {len(dev_ok)} of {len(HORIZONS)} horizons, needs {need_horizons}")
            rows.append({"kind": kind, "status": STATUS_FAIL, "reasons": "; ".join(reasons)})
            continue
        if val.empty:
            rows.append({"kind": kind, "status": STATUS_ELIGIBLE,
                         "reasons": f"development passes at horizons {dev_ok}; validation not viewed yet"})
            continue
        val_ok = passing_horizons(val, **kw)
        if len(val_ok) < need_horizons:
            reasons.append(f"validation: passes at {len(val_ok)} of {len(HORIZONS)} horizons, needs {need_horizons}")
        common = sorted(set(dev_ok) & set(val_ok))
        if not common:
            reasons.append("no horizon passes in both development and validation")
        else:
            h = common[0]
            d_row = dev[dev["horizon"] == h].iloc[0]
            v_row = val[val["horizon"] == h].iloc[0]
            total = int(d_row["halves"] + v_row["halves"])
            positive = int(d_row["halves_positive"] + v_row["halves_positive"])
            if total < total_halves:
                reasons.append(f"only {total} half-year slices available, needs {total_halves}")
            elif positive < min_halves:
                reasons.append(f"excess positive in {positive} of {total} half-year slices at {h}h, needs {min_halves}")
        if reasons:
            rows.append({"kind": kind, "status": STATUS_FAIL, "reasons": "; ".join(reasons)})
        else:
            rows.append({"kind": kind, "status": STATUS_PASS,
                         "reasons": f"passes in both periods at horizons {common}; still needs the sealed holdout"})
    return pd.DataFrame(rows, columns=["kind", "status", "reasons"])
