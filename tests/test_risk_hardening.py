"""
Release A, batch 1: malformed-input and fail-closed tests for the risk layer.

Rule under test: bad, missing or unexpected input must produce a REJECTION with a
reason code. It must never produce an approval and never an unhandled exception.
"""
import dataclasses

import pytest

from backend.risk.kill_switch import KillSwitch
from backend.risk.models import (AccountState, OpenPosition, RiskConfig, SymbolSpec,
                                 TradeProposal, finite_number, finite_positive)
from backend.risk.risk_manager import RiskEventLog, RiskManager

NAN, INF = float("nan"), float("inf")
EURUSD = SymbolSpec("EURUSD", 0.00001, 0.00001, 1.0, 0.01, 100.0, 0.01, 0)


def cfg(**kw):
    return RiskConfig(reference_balance=kw.pop("reference_balance", 10000.0), **kw)


def acct(**kw):
    base = dict(balance=10000.0, equity=10000.0, day_start_equity=10000.0,
                peak_equity=10000.0, data_age_seconds=0.0)
    base.update(kw)
    return AccountState(**base)


def buy(**kw):
    base = dict(symbol="EURUSD", direction=1, entry=1.1000, stop_loss=1.0975,
                take_profit=1.1050, spread=0.00008, atr=0.0010)
    base.update(kw)
    return TradeProposal(**base)


def good_pos(**kw):
    base = dict(symbol="GBPUSD", direction=1, volume=0.1, entry=1.3, stop_loss=1.29, risk_amount=10.0)
    base.update(kw)
    return OpenPosition(**base)


def manager(tmp_path, **cfgkw):
    ks = KillSwitch(tmp_path / "ks.json")
    return RiskManager(cfg(**cfgkw), ks, RiskEventLog(tmp_path / "log.csv")), ks


# ------------------------------------------------------------------ baseline
def test_baseline_valid_trade_is_approved(tmp_path):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(buy(), acct(), [], EURUSD)
    assert d.approved, d.codes


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize("v", [True, False, NAN, INF, -INF, "1", None])
def test_finite_helpers_reject_bool_nan_inf_and_non_numbers(v):
    assert not finite_number(v)
    assert not finite_positive(v)


def test_finite_helpers_accept_normal_numbers():
    assert finite_number(0, -1.5, 3)
    assert finite_positive(1, 0.001)
    assert not finite_positive(0)


# ------------------------------------------------------------------ RiskConfig validation
BAD_CONFIG = [
    ("reference_balance", NAN), ("reference_balance", INF), ("reference_balance", "10000"),
    ("reference_balance", True), ("reference_balance", -5),
    ("risk_per_trade_pct", NAN), ("risk_per_trade_pct", "0.5"), ("risk_per_trade_pct", 0),
    ("risk_per_trade_pct", True),
    ("max_daily_loss_pct", NAN), ("max_daily_loss_pct", 0), ("max_daily_loss_pct", 50),
    ("floor_pct_of_reference", NAN), ("floor_pct_of_reference", 100), ("floor_pct_of_reference", 0),
    ("max_open_positions", 0), ("max_open_positions", 2.5), ("max_open_positions", True),
    ("max_open_positions", "3"), ("max_open_positions", None),
    ("max_total_risk_pct", 0), ("max_total_risk_pct", NAN), ("max_total_risk_pct", 50),
    ("max_drawdown_pct", 0), ("max_drawdown_pct", NAN), ("max_drawdown_pct", 100),
    ("max_drawdown_pct", 1000),
    ("min_reward_risk", 0), ("min_reward_risk", NAN), ("min_reward_risk", -1),
    ("max_same_currency_positions", 0), ("max_same_currency_positions", 1.5),
    ("min_stop_spread_mult", -1), ("min_stop_spread_mult", NAN),
    ("max_stop_atr_mult", 0), ("max_stop_atr_mult", NAN),
    ("max_spread_frac_of_stop", 0), ("max_spread_frac_of_stop", 2), ("max_spread_frac_of_stop", NAN),
    ("max_data_age_seconds", NAN), ("max_data_age_seconds", INF), ("max_data_age_seconds", 0),
    ("max_data_age_seconds", -1), ("max_data_age_seconds", "60"),
    ("max_spread_points", {"EURUSD": NAN}), ("max_spread_points", {"EURUSD": -5}),
    ("max_spread_points", {"EURUSD": "20"}), ("max_spread_points", {1: 20}),
    ("max_spread_points", "bad"), ("max_spread_points", None),
    ("close_all_on_floor", "yes"), ("close_all_on_daily_loss", 1), ("close_all_on_drawdown", None),
]


@pytest.mark.parametrize("name,value", BAD_CONFIG)
def test_bad_config_is_refused(name, value):
    with pytest.raises(ValueError):
        cfg(**{name: value})


def test_contradictory_config_is_refused():
    # total-risk cap below the per-trade risk would silently block every trade
    with pytest.raises(ValueError):
        cfg(risk_per_trade_pct=1.0, max_total_risk_pct=0.5)


def test_config_cannot_be_changed_after_validation():
    c = cfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.risk_per_trade_pct = 50
    with pytest.raises(TypeError):
        c.max_spread_points["EURUSD"] = 1


def test_default_and_sample_configs_still_load():
    cfg()
    cfg(max_spread_points={"EURUSD": 20, "XAUUSD": 500}, close_all_on_daily_loss=True)


# ------------------------------------------------------------------ AccountState
def test_data_age_is_required_so_it_cannot_default_to_fresh():
    with pytest.raises(TypeError):
        AccountState(balance=1.0, equity=1.0, day_start_equity=1.0, peak_equity=1.0)


@pytest.mark.parametrize("age", [-1.0, -9999.0, NAN, INF, True, "0", None])
def test_bad_data_age_is_rejected(tmp_path, age):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(buy(), acct(data_age_seconds=age), [], EURUSD)
    assert not d.approved and d.codes == ["INVALID_INPUT"]


def test_stale_data_is_rejected(tmp_path):
    rm, _ = manager(tmp_path, max_data_age_seconds=30)
    d = rm.evaluate(buy(), acct(data_age_seconds=31), [], EURUSD)
    assert not d.approved and d.codes == ["STALE_DATA"]


@pytest.mark.parametrize("field", ["balance", "equity", "day_start_equity", "peak_equity"])
@pytest.mark.parametrize("bad", [True, NAN, INF, 0, -1, "10000", None])
def test_bad_account_values_are_invalid_input(tmp_path, field, bad):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(buy(), acct(**{field: bad}), [], EURUSD)
    assert not d.approved and d.codes == ["INVALID_INPUT"]


# ------------------------------------------------------------------ OpenPosition validation
BAD_POSITIONS = [
    good_pos(symbol=None), good_pos(symbol=""), good_pos(symbol=123),
    good_pos(direction=0), good_pos(direction=2), good_pos(direction=True), good_pos(direction="long"),
    good_pos(volume=NAN), good_pos(volume=0), good_pos(volume=-1), good_pos(volume=True),
    good_pos(entry=NAN), good_pos(entry=0),
    good_pos(stop_loss=0),            # 0 means "no stop-loss": unlimited risk, must be refused
    good_pos(stop_loss=NAN), good_pos(stop_loss=None),
    good_pos(risk_amount=NAN), good_pos(risk_amount=INF), good_pos(risk_amount="10"),
    good_pos(risk_amount=None),
    None, {"symbol": "GBPUSD"}, "GBPUSD",
]


@pytest.mark.parametrize("bad", BAD_POSITIONS, ids=lambda p: repr(p)[:60])
def test_bad_open_position_blocks_new_trades_without_crashing(tmp_path, bad):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(buy(), acct(), [bad], EURUSD)
    assert not d.approved and "INVALID_INPUT" in d.codes


@pytest.mark.parametrize("bad", BAD_POSITIONS, ids=lambda p: repr(p)[:60])
def test_bad_open_position_makes_monitor_block_and_lock(tmp_path, bad):
    rm, ks = manager(tmp_path)
    r = rm.monitor(acct(), [bad])
    assert "BLOCK_NEW_ORDERS" in r.actions and "INVALID_INPUT" in r.codes
    assert ks.is_active()


def test_non_iterable_positions_are_rejected(tmp_path):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(buy(), acct(), 5, EURUSD)
    assert not d.approved and "INVALID_INPUT" in d.codes


def test_profit_locked_position_with_negative_risk_amount_is_still_valid(tmp_path):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(buy(), acct(), [good_pos(risk_amount=-5.0)], EURUSD)
    assert d.approved, d.codes


def test_monitor_with_good_positions_does_nothing(tmp_path):
    rm, ks = manager(tmp_path)
    r = rm.monitor(acct(), [good_pos()])
    assert r.actions == [] and not ks.is_active()


# ------------------------------------------------------------------ wrong object types
@pytest.mark.parametrize("p,a,s", [
    (None, acct(), EURUSD), (buy(), None, EURUSD), (buy(), acct(), None),
    ("EURUSD", acct(), EURUSD), (buy(), {"equity": 1}, EURUSD), (buy(), acct(), {"point": 1}),
])
def test_wrong_object_types_are_rejected_not_raised(tmp_path, p, a, s):
    rm, _ = manager(tmp_path)
    d = rm.evaluate(p, a, [], s)
    assert not d.approved and d.codes == ["INVALID_INPUT"]


def test_monitor_with_wrong_account_type_blocks_and_locks(tmp_path):
    rm, ks = manager(tmp_path)
    r = rm.monitor(None, [])
    assert "BLOCK_NEW_ORDERS" in r.actions
    assert ks.is_active()


# ------------------------------------------------------------------ catch-all
def test_unexpected_internal_error_becomes_a_rejection(tmp_path, monkeypatch):
    rm, _ = manager(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(rm, "_breaches", boom)
    d = rm.evaluate(buy(), acct(), [], EURUSD)
    assert not d.approved and d.codes == ["INTERNAL_ERROR"]
    assert d.reasons and d.reasons[0] != "INTERNAL_ERROR"       # has readable text


def test_monitor_internal_error_blocks_and_locks(tmp_path, monkeypatch):
    rm, ks = manager(tmp_path)
    monkeypatch.setattr(rm, "_breaches", lambda a: (_ for _ in ()).throw(RuntimeError("bug")))
    r = rm.monitor(acct(), [])
    assert "BLOCK_NEW_ORDERS" in r.actions and "INTERNAL_ERROR" in r.codes
    assert ks.is_active()


# ------------------------------------------------------------------ audit log failure policy
def _broken_log(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")                       # a FILE where the log's folder should be
    return RiskEventLog(blocker / "log.csv")


def test_approval_is_withheld_when_the_decision_cannot_be_logged(tmp_path):
    rm = RiskManager(cfg(), KillSwitch(tmp_path / "ks.json"), _broken_log(tmp_path))
    d = rm.evaluate(buy(), acct(), [], EURUSD)
    assert not d.approved and d.codes == ["AUDIT_LOG_FAILED"]


def test_rejection_still_returned_when_logging_fails(tmp_path):
    rm = RiskManager(cfg(), KillSwitch(tmp_path / "ks.json"), _broken_log(tmp_path))
    d = rm.evaluate(buy(), acct(equity=1.0), [], EURUSD)
    assert not d.approved and "AUDIT_LOG_FAILED" not in d.codes


# ------------------------------------------------------------------ state paths
def test_default_paths_are_absolute_and_ignore_the_working_directory(tmp_path, monkeypatch):
    before_ks, before_log = KillSwitch().path, RiskEventLog().path
    monkeypatch.chdir(tmp_path)
    assert KillSwitch().path == before_ks and before_ks.is_absolute()
    assert RiskEventLog().path == before_log and before_log.is_absolute()
