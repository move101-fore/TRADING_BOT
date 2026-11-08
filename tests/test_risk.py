import math
import random

import pytest

from backend.risk.kill_switch import KillSwitch
from backend.risk.models import (AccountState, OpenPosition, RiskConfig, SymbolSpec,
                                 TradeProposal)
from backend.risk.position_sizing import lots_for_risk, risk_money, round_down_to_step
from backend.risk.risk_manager import RiskEventLog, RiskManager
from backend.risk.stop_loss import validate_levels

EURUSD = SymbolSpec("EURUSD", point=0.00001, tick_size=0.00001, tick_value=1.0,
                    volume_min=0.01, volume_max=100.0, volume_step=0.01, stops_level_points=0)
GBPUSD = SymbolSpec("GBPUSD", 0.00001, 0.00001, 1.0, 0.01, 100.0, 0.01)
AUDUSD = SymbolSpec("AUDUSD", 0.00001, 0.00001, 1.0, 0.01, 100.0, 0.01)
USDJPY = SymbolSpec("USDJPY", 0.001, 0.001, 0.65, 0.01, 100.0, 0.01)


def cfg(**kw):
    return RiskConfig(reference_balance=kw.pop("reference_balance", 10000.0), **kw)


def acct(equity=10000.0, balance=None, day_start=10000.0, peak=10000.0, age=0.0):
    return AccountState(balance=equity if balance is None else balance, equity=equity,
                        day_start_equity=day_start, peak_equity=peak, data_age_seconds=age)


def manager(tmp_path, **kw):
    ks = KillSwitch(tmp_path / "ks.json")
    return RiskManager(cfg(**kw), ks, RiskEventLog(tmp_path / "log.csv")), ks


def buy(symbol="EURUSD", entry=1.1000, stop=1.0975, tp=1.1050, spread=0.00008, atr=0.0010, **kw):
    return TradeProposal(symbol, 1, entry, stop, tp, spread, atr, **kw)


def pos(symbol, direction, risk=10.0):
    return OpenPosition(symbol, direction, 0.1, 1.0, 0.9, risk)


# ---------------------------------------------------------------- sizing
def test_lot_size_known_value():
    # 10000 * 0.5% = 50; stop 25 pips = 250 ticks = $250 per lot -> 0.2 lots
    assert lots_for_risk(10000, 0.5, 0.0025, EURUSD) == 0.2


def test_lot_size_rounds_down_never_up():
    lots = lots_for_risk(10000, 0.5, 0.0030, EURUSD)      # raw 0.1667
    assert lots == 0.16
    assert risk_money(0.0030, lots, EURUSD) <= 50.0


def test_round_down_helper():
    assert round_down_to_step(0.169, 0.01) == 0.16
    assert round_down_to_step(0.17, 0.01) == 0.17
    assert round_down_to_step(float("nan"), 0.01) == 0.0


def test_fuzz_sizing_never_exceeds_cap():
    rnd = random.Random(1)
    for _ in range(3000):
        equity = rnd.uniform(50, 200000)
        stop = rnd.uniform(0.0002, 0.02)
        spec = SymbolSpec("X", 0.00001, 0.00001, rnd.uniform(0.3, 2.0), 0.01, 50.0,
                          rnd.choice([0.01, 0.1, 1.0]))
        lots = lots_for_risk(equity, 0.5, stop, spec)
        assert risk_money(stop, lots, spec) <= equity * 0.5 / 100 + 1e-9
        assert lots <= 50.0 + 1e-9
        assert abs(lots / spec.volume_step - round(lots / spec.volume_step)) < 1e-6


# ---------------------------------------------------------------- approvals
def test_clean_trade_is_approved_with_correct_risk(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(), acct(), [], EURUSD)
    assert d.approved and d.lots == 0.2
    assert d.risk_pct <= 0.5 + 1e-9


def test_risk_uses_lower_of_balance_and_equity(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(), acct(equity=9500, balance=10000, day_start=9500, peak=10000), [], EURUSD)
    assert d.approved and d.risk_amount <= 9500 * 0.005 + 1e-9


def test_lot_below_minimum_is_rejected_not_rounded_up(tmp_path):
    m, _ = manager(tmp_path, reference_balance=500)
    d = m.evaluate(buy(stop=1.0970, tp=1.1060), acct(equity=500, day_start=500, peak=500), [], EURUSD)
    assert not d.approved and "LOT_BELOW_MIN" in d.codes


# ---------------------------------------------------------------- levels
def test_stop_wrong_side(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(stop=1.1010), acct(), [], EURUSD)
    assert "STOP_WRONG_SIDE" in d.codes and not d.approved


def test_take_profit_wrong_side(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(tp=1.0990), acct(), [], EURUSD)
    assert "TP_WRONG_SIDE" in d.codes


def test_sell_levels_checked_mirrored(tmp_path):
    m, _ = manager(tmp_path)
    ok = TradeProposal("EURUSD", -1, 1.1000, 1.1025, 1.0950, 0.00008, 0.001)
    assert m.evaluate(ok, acct(), [], EURUSD).approved
    bad = TradeProposal("EURUSD", -1, 1.1000, 1.0975, 1.0950, 0.00008, 0.001)
    assert "STOP_WRONG_SIDE" in m.evaluate(bad, acct(), [], EURUSD).codes


def test_reward_risk_below_two_rejected(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(tp=1.1030), acct(), [], EURUSD)        # 30 / 25 = 1.2
    assert "RR_TOO_LOW" in d.codes


def test_reward_risk_exactly_two_accepted():
    codes, dist, rr = validate_levels(1, 1.1000, 1.0975, 1.1050, 0.00008, 0.001, EURUSD, cfg())
    assert codes == [] and abs(rr - 2.0) < 1e-9


def test_stop_closer_than_broker_minimum():
    spec = SymbolSpec("EURUSD", 0.00001, 0.00001, 1.0, 0.01, 100.0, 0.01, stops_level_points=300)
    codes, _, _ = validate_levels(1, 1.1000, 1.0990, 1.1020, 0.00001, 0.001, spec, cfg())
    assert "STOP_TOO_CLOSE" in codes


def test_stop_too_tight_versus_spread():
    codes, _, _ = validate_levels(1, 1.1000, 1.0998, 1.1004, 0.0001, 0.001, EURUSD, cfg())
    assert "STOP_TIGHT_VS_SPREAD" in codes


def test_stop_too_wide_versus_atr():
    codes, _, _ = validate_levels(1, 1.1000, 1.0900, 1.1200, 0.00008, 0.0010, EURUSD, cfg())
    assert "STOP_TOO_WIDE" in codes


def test_spread_too_wide_fraction_and_absolute():
    codes, _, _ = validate_levels(1, 1.1000, 1.0975, 1.1050, 0.0008, 0.001, EURUSD, cfg())
    assert "SPREAD_TOO_WIDE" in codes
    c2 = cfg(max_spread_points={"EURUSD": 5})
    codes, _, _ = validate_levels(1, 1.1000, 1.0975, 1.1050, 0.00008, 0.001, EURUSD, c2)
    assert "SPREAD_TOO_WIDE" in codes


# ---------------------------------------------------------------- limits
def test_duplicate_position_rejected(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(), acct(), [pos("EURUSD", 1)], EURUSD)
    assert "DUPLICATE_POSITION" in d.codes


def test_max_open_positions(tmp_path):
    m, _ = manager(tmp_path)
    three = [pos("GBPUSD", 1, 1), pos("USDJPY", 1, 1), pos("XAUUSD", 1, 1)]
    d = m.evaluate(buy(), acct(), three, EURUSD)
    assert "MAX_POSITIONS" in d.codes


def test_total_open_risk_cap(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(), acct(), [pos("GBPUSD", -1, 120.0)], EURUSD)   # 120 + 50 > 150
    assert "TOTAL_RISK_CAP" in d.codes


def test_currency_exposure_cap(tmp_path):
    m, _ = manager(tmp_path)
    held = [pos("EURUSD", 1, 5), pos("GBPUSD", 1, 5)]                  # both short USD
    d = m.evaluate(buy("AUDUSD", 0.6500, 0.6475, 0.6550), acct(), held, AUDUSD)
    assert "CURRENCY_EXPOSURE" in d.codes
    sell = TradeProposal("AUDUSD", -1, 0.6500, 0.6525, 0.6450, 0.00008, 0.001)
    assert m.evaluate(sell, acct(), held, AUDUSD).approved            # opposite USD side is fine


def test_daily_loss_projected_limit(tmp_path):
    m, ks = manager(tmp_path)
    ok = m.evaluate(buy(), acct(equity=9850), [], EURUSD)              # 150 + 47.5 is under the 200 limit
    assert ok.approved
    bad = m.evaluate(buy(), acct(equity=9830, day_start=10000, peak=10000), [], EURUSD)   # 170 + 47.5 > 200
    assert "DAILY_LOSS_LIMIT" in bad.codes and not bad.approved


def test_daily_loss_reached_locks_trading(tmp_path):
    m, ks = manager(tmp_path)
    r = m.monitor(acct(equity=9800), [])
    assert "BLOCK_NEW_ORDERS" in r.actions and "CLOSE_ALL_POSITIONS" not in r.actions
    assert ks.is_active()
    d = m.evaluate(buy(), acct(), [], EURUSD)
    assert d.codes == ["KILL_SWITCH_ACTIVE"]


def test_max_drawdown_locks_without_closing(tmp_path):
    m, ks = manager(tmp_path, reference_balance=5000)
    r = m.monitor(acct(equity=8999, day_start=8999, peak=10000), [])
    assert "MAX_DRAWDOWN_REACHED" in r.codes and "CLOSE_ALL_POSITIONS" not in r.actions
    assert ks.is_active()


# ---------------------------------------------------------------- equity floor
def test_equity_floor_kills_and_closes_everything(tmp_path):
    m, ks = manager(tmp_path, reference_balance=500)
    a = acct(equity=449, balance=449, day_start=449, peak=449)
    r = m.monitor(a, [pos("EURUSD", 1)])
    assert "EQUITY_FLOOR_BREACHED" in r.codes
    assert "CLOSE_ALL_POSITIONS" in r.actions and "BLOCK_NEW_ORDERS" in r.actions
    assert ks.is_active()


def test_equity_floor_uses_lower_of_balance_and_equity(tmp_path):
    m, ks = manager(tmp_path, reference_balance=500)
    a = acct(equity=470, balance=449, day_start=480, peak=480)
    r = m.monitor(a, [])
    assert "EQUITY_FLOOR_BREACHED" in r.codes


def test_above_floor_does_not_trigger(tmp_path):
    m, ks = manager(tmp_path, reference_balance=500)
    r = m.monitor(acct(equity=455, balance=455, day_start=455, peak=500), [])
    assert r.actions == [] and not ks.is_active()


def test_floor_breach_blocks_new_order_even_without_monitor(tmp_path):
    m, ks = manager(tmp_path, reference_balance=500)
    d = m.evaluate(buy(), acct(equity=440, day_start=440, peak=440), [], EURUSD)
    assert not d.approved and "EQUITY_FLOOR_BREACHED" in d.codes and ks.is_active()


# ---------------------------------------------------------------- kill switch
def test_kill_switch_persists_across_restart(tmp_path):
    path = tmp_path / "k.json"
    KillSwitch(path).activate("manual test")
    assert KillSwitch(path).is_active()


def test_kill_switch_corrupt_file_fails_closed(tmp_path):
    path = tmp_path / "k.json"
    path.write_text("{not json")
    assert KillSwitch(path).is_active()


def test_kill_switch_reset_needs_confirm_and_reason(tmp_path):
    ks = KillSwitch(tmp_path / "k.json")
    ks.activate("x")
    with pytest.raises(ValueError):
        ks.reset("ok", confirm=False)
    with pytest.raises(ValueError):
        ks.reset("   ", confirm=True)
    assert ks.is_active()
    ks.reset("checked the account", confirm=True)
    assert not ks.is_active()
    assert [h["event"] for h in ks.status()["history"]] == ["activated", "reset"]


def test_kill_switch_never_resets_itself(tmp_path):
    m, ks = manager(tmp_path)
    m.monitor(acct(equity=9800), [])
    assert ks.is_active()
    assert m.evaluate(buy(), acct(), [], EURUSD).codes == ["KILL_SWITCH_ACTIVE"]   # next "day", still locked


# ---------------------------------------------------------------- fail closed + log
def test_invalid_inputs_fail_closed(tmp_path):
    m, _ = manager(tmp_path)
    assert "INVALID_INPUT" in m.evaluate(buy(), acct(equity=float("nan")), [], EURUSD).codes
    bad_spec = SymbolSpec("EURUSD", 0.00001, 0.00001, 0.0, 0.01, 100.0, 0.01)
    assert "INVALID_INPUT" in m.evaluate(buy(), acct(), [], bad_spec).codes
    nan_trade = buy(entry=float("nan"))
    assert not m.evaluate(nan_trade, acct(), [], EURUSD).approved


def test_stale_account_data_rejected(tmp_path):
    m, _ = manager(tmp_path)
    d = m.evaluate(buy(), acct(age=300), [], EURUSD)
    assert "STALE_DATA" in d.codes and not d.approved


def test_every_decision_is_logged(tmp_path):
    m, _ = manager(tmp_path)
    m.evaluate(buy(), acct(), [], EURUSD)
    m.evaluate(buy(tp=1.1030), acct(), [], EURUSD)
    lines = (tmp_path / "log.csv").read_text().strip().splitlines()
    assert len(lines) == 3 and "RR_TOO_LOW" in lines[2]


def test_config_refuses_unsafe_values():
    with pytest.raises(ValueError):
        RiskConfig(reference_balance=500, risk_per_trade_pct=5)
    with pytest.raises(ValueError):
        RiskConfig(reference_balance=0)


def test_fuzz_approved_trades_always_satisfy_every_rule(tmp_path):
    rnd = random.Random(7)
    m, _ = manager(tmp_path)
    approved = 0
    for _ in range(2500):
        direction = rnd.choice([1, -1])
        entry = 1.1
        stop_d = rnd.uniform(0.0003, 0.01)
        rr = rnd.uniform(1.0, 3.0)
        stop = entry - direction * stop_d
        tp = entry + direction * stop_d * rr
        p = TradeProposal("EURUSD", direction, entry, stop, tp, rnd.uniform(0, 0.0003), 0.001)
        equity = rnd.uniform(9500, 20000)
        d = m.evaluate(p, acct(equity=equity, day_start=equity, peak=equity), [], EURUSD)
        if d.approved:
            approved += 1
            assert d.risk_pct <= 0.5 + 1e-9
            assert abs(tp - entry) / abs(entry - stop) >= 2.0 - 1e-9
            assert d.lots >= 0.01
    assert approved > 0
