import pandas as pd

from backend.research.verdict import STATUS_ELIGIBLE, STATUS_FAIL, STATUS_PASS, row_passes, verdict


def row(kind, period, horizon, net_lo=0.1, exc_lo=0.1, events=500, hp=3, halves=4, p=None):
    d = {"kind": kind, "period": period, "horizon": horizon, "events": events, "net_lo": net_lo,
         "excess_lo": exc_lo, "halves_positive": hp, "halves": halves}
    if p is not None:
        d["placebo_p"] = p
    return d


def table(*rows):
    return pd.DataFrame(rows)


def dev(h, **kw):
    return row("K", "development", h, halves=4, **kw)


def val(h, **kw):
    return row("K", "validation", h, halves=2, hp=kw.pop("hp", 2), **kw)


def test_two_of_three_horizons_in_development_makes_it_eligible_only():
    v = verdict(table(dev(6), dev(12), dev(24, net_lo=-0.1)))
    assert v.iloc[0]["status"] == STATUS_ELIGIBLE


def test_one_horizon_is_not_enough():
    v = verdict(table(dev(6), dev(12, net_lo=-0.1), dev(24, exc_lo=-0.1)))
    assert v.iloc[0]["status"] == STATUS_FAIL and "1 of 3" in v.iloc[0]["reasons"]


def test_both_lower_bounds_are_required():
    assert not row_passes(dev(6, net_lo=-0.01))             # profitable on average is not enough
    assert not row_passes(dev(6, exc_lo=-0.01))             # information without profit is not enough
    assert row_passes(dev(6))


def test_minimum_event_count():
    assert not row_passes(dev(6, events=299)) and row_passes(dev(6, events=300))


def test_placebo_p_value_gate():
    assert not row_passes(dev(6, p=0.03)) and row_passes(dev(6, p=0.02))


def test_full_pass_needs_validation_and_half_years():
    t = table(dev(6, hp=3), dev(12, hp=3), dev(24, net_lo=-1), val(6, hp=2), val(12, hp=2), val(24, net_lo=-1))
    v = verdict(t)
    assert v.iloc[0]["status"] == STATUS_PASS


def test_validation_failure_fails_the_candidate():
    t = table(dev(6), dev(12), dev(24), val(6), val(12, net_lo=-0.1), val(24, exc_lo=-0.1))
    v = verdict(t)
    assert v.iloc[0]["status"] == STATUS_FAIL and "validation" in v.iloc[0]["reasons"]


def test_half_year_rule_needs_four_of_six():
    t = table(dev(6, hp=2), dev(12, hp=2), dev(24, net_lo=-1), val(6, hp=0), val(12, hp=0), val(24, net_lo=-1))
    v = verdict(t)
    assert v.iloc[0]["status"] == STATUS_FAIL and "2 of 6" in v.iloc[0]["reasons"]


def test_incomplete_half_year_coverage_fails():
    t = table(dev(6), dev(12), dev(24), val(6, hp=1), val(12, hp=1), val(24))
    t.loc[t["period"] == "validation", "halves"] = 1                   # only 5 slices in total
    v = verdict(t)
    assert v.iloc[0]["status"] == STATUS_FAIL and "slices" in v.iloc[0]["reasons"]


def test_a_horizon_must_pass_in_both_periods():
    t = table(dev(6), dev(12, net_lo=-1), dev(24, net_lo=-1), val(6, net_lo=-1), val(12, net_lo=-1), val(24))
    v = verdict(t, need_horizons=1)
    assert v.iloc[0]["status"] == STATUS_FAIL and "both" in v.iloc[0]["reasons"]


def test_each_kind_is_judged_separately():
    t = table(row("A", "development", 6, halves=4), row("A", "development", 12, halves=4),
              row("B", "development", 6, net_lo=-1, halves=4), row("B", "development", 12, net_lo=-1, halves=4))
    v = verdict(t).set_index("kind")
    assert v.loc["A", "status"] == STATUS_ELIGIBLE and v.loc["B", "status"] == STATUS_FAIL
