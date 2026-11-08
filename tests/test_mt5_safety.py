from types import SimpleNamespace

import numpy as np
import pytest

from backend.market_data.mt5_connector import (
    MT5AccountError, MT5Connector, MT5ConnectionError, MT5PermissionError,
    MT5RequestError, MT5SymbolError, validate_candles, validate_symbol_info,
)
from backend.risk.kill_switch import KillSwitch

RATE_DTYPE = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
              ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]


def good_symbol(**over):
    base = dict(point=0.00001, trade_tick_size=0.00001, trade_tick_value=1.0, trade_tick_value_loss=1.0,
                trade_contract_size=100000.0, volume_min=0.01, volume_max=100.0, volume_step=0.01,
                digits=5, trade_stops_level=0, trade_mode=4, visible=True)
    base.update(over)
    return SimpleNamespace(**base)


def good_rates(n=5):
    t = 1_700_000_000
    rows = [(t + i * 3600, 1.1, 1.101, 1.099, 1.1005, 100, 8, 0) for i in range(n)]
    return np.array(rows, dtype=RATE_DTYPE)


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    SYMBOL_TRADE_MODE_FULL = 4
    TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 15, 16385, 16388, 16408

    def __init__(self, login=123456, server="Broker-Demo", mode=0):
        self.account = SimpleNamespace(login=login, server=server, trade_mode=mode, trade_allowed=True,
                                       trade_expert=True, balance=10000.0, equity=10000.0)
        self.terminal = SimpleNamespace(connected=True, trade_allowed=True, tradeapi_disabled=False)
        self.init_failures = 0
        self.init_calls = 0
        self.login_calls = []
        self.login_ok = True
        self.login_becomes = None              # (login, server) the account ends up on after login
        self.shutdown_calls = 0
        self.symbols = {"EURUSDm": good_symbol()}
        self.select_ok = True
        self.rates = good_rates()
        self.rates_failures = 0
        self.rates_calls = 0
        self._err = (1, "Success")

    def initialize(self, **kwargs):
        self.init_calls += 1
        if self.init_calls <= self.init_failures:
            self._err = (-10005, "IPC timeout")
            return False
        return True

    def shutdown(self):
        self.shutdown_calls += 1

    def login(self, login, password="", server=""):
        self.login_calls.append((login, server))
        if not self.login_ok:
            self._err = (-6, "Authorization failed")
            return False
        if self.login_becomes:
            self.account.login, self.account.server = self.login_becomes
        else:
            self.account.login, self.account.server = login, server
        return True

    def last_error(self):
        return self._err

    def account_info(self):
        return self.account

    def terminal_info(self):
        return self.terminal

    def symbol_select(self, name, enable):
        return self.select_ok

    def symbol_info(self, name):
        return self.symbols.get(name)

    def copy_rates_from_pos(self, name, tf, start, count):
        self.rates_calls += 1
        if self.rates_calls <= self.rates_failures:
            self._err = (-1, "no data")
            return None
        return self.rates


def make(fake=None, **kw):
    fake = fake or FakeMT5()
    kw.setdefault("expected_login", 123456)
    kw.setdefault("expected_server", "Broker-Demo")
    kw.setdefault("expected_mode", "DEMO")
    kw.setdefault("password", "pw")
    kw.setdefault("symbols", {"EURUSD": "EURUSDm"})
    kw.setdefault("sleep", lambda s: None)
    return MT5Connector(api=fake, **kw), fake


# ---------------------------------------------------------------- accounts and servers
def test_correct_account_and_server_are_accepted():
    c, fake = make()
    info = c.connect()
    assert info.login == 123456 and c.connected


def test_wrong_account_is_rejected_and_connection_closed():
    c, fake = make(FakeMT5(login=999999))
    fake.login_ok = False
    with pytest.raises(MT5AccountError):
        c.connect()
    assert fake.shutdown_calls >= 1 and not c.connected


def test_wrong_server_is_rejected():
    c, fake = make(FakeMT5(server="Other-Server"))
    with pytest.raises(MT5AccountError, match="wrong server"):
        c.connect()
    assert fake.shutdown_calls >= 1


def test_server_comparison_ignores_case_and_spaces():
    c, _ = make(FakeMT5(server=" broker-demo "))
    assert c.connect().login == 123456


def test_missing_account_info_is_rejected():
    c, fake = make()
    fake.account = None
    fake.login_ok = False
    with pytest.raises(MT5AccountError):
        c.connect()


def test_account_without_login_or_server_is_rejected():
    c, fake = make()
    fake.account = SimpleNamespace(login=123456, server="", trade_mode=0)
    with pytest.raises(MT5AccountError, match="missing"):
        c.connect()


def test_unconfigured_expected_account_refuses_to_connect():
    c, fake = make(expected_login=0)
    with pytest.raises(MT5AccountError, match="MT5_LOGIN"):
        c.connect()
    assert fake.init_calls == 0
    c2, fake2 = make(expected_server="  ")
    with pytest.raises(MT5AccountError, match="MT5_SERVER"):
        c2.connect()


def test_real_account_is_rejected_when_demo_is_expected():
    c, fake = make(FakeMT5(mode=2))
    with pytest.raises(MT5AccountError, match="wrong account type"):
        c.connect()


def test_real_account_accepted_only_when_real_is_expected():
    c, _ = make(FakeMT5(mode=2), expected_mode="REAL")
    assert c.connect().login == 123456


def test_login_is_used_when_terminal_is_on_another_account_then_verified():
    fake = FakeMT5(login=555555, server="Broker-Demo")
    c, _ = make(fake)
    info = c.connect()
    assert fake.login_calls == [(123456, "Broker-Demo")] and info.login == 123456


def test_login_that_lands_on_wrong_account_is_rejected():
    fake = FakeMT5(login=555555)
    fake.login_becomes = (777777, "Broker-Demo")
    c, _ = make(fake)
    with pytest.raises(MT5AccountError, match="wrong account"):
        c.connect()
    assert fake.shutdown_calls >= 1


def test_failed_login_is_rejected():
    fake = FakeMT5(login=555555)
    fake.login_ok = False
    c, _ = make(fake)
    with pytest.raises(MT5AccountError, match="login to the expected account failed"):
        c.connect()


def test_no_password_and_wrong_account_is_rejected_without_login_attempt():
    fake = FakeMT5(login=555555)
    c, _ = make(fake, password="")
    with pytest.raises(MT5AccountError):
        c.connect()
    assert fake.login_calls == []


# ---------------------------------------------------------------- failed connections
def test_initialize_failure_raises_after_retries():
    fake = FakeMT5()
    fake.init_failures = 99
    c, _ = make(fake, retries=3)
    with pytest.raises(MT5ConnectionError, match="3 attempts"):
        c.connect()
    assert fake.init_calls == 3 and fake.shutdown_calls >= 1


def test_initialize_recovers_on_second_attempt():
    fake = FakeMT5()
    fake.init_failures = 1
    c, _ = make(fake)
    assert c.connect().login == 123456 and fake.init_calls == 2


def test_terminal_not_connected_to_broker_is_rejected():
    c, fake = make()
    fake.terminal.connected = False
    with pytest.raises(MT5ConnectionError, match="not connected"):
        c.connect()


# ---------------------------------------------------------------- permissions
def test_data_only_connection_does_not_need_trading_permission():
    c, fake = make()
    fake.terminal.trade_allowed = False
    assert c.connect(require_trading=False).login == 123456
    assert not c.last_permissions.trading_ok
    assert any("Algo Trading" in i for i in c.last_permissions.issues)


@pytest.mark.parametrize("where,attr,value,text", [
    ("terminal", "trade_allowed", False, "Algo Trading"),
    ("terminal", "tradeapi_disabled", True, "trading API"),
    ("account", "trade_allowed", False, "not allowed on this account"),
    ("account", "trade_expert", False, "expert"),
])
def test_trading_mode_rejects_missing_permissions(where, attr, value, text):
    c, fake = make()
    setattr(getattr(fake, where), attr, value)
    with pytest.raises(MT5PermissionError, match=text):
        c.connect(require_trading=True)
    assert fake.shutdown_calls >= 1


def test_missing_permission_fields_fail_closed():
    c, fake = make()
    fake.terminal = SimpleNamespace(connected=True)
    with pytest.raises(MT5PermissionError):
        c.connect(require_trading=True)


def test_all_permissions_ok_in_trading_mode():
    c, _ = make()
    assert c.connect(require_trading=True).login == 123456 and c.last_permissions.trading_ok


# ---------------------------------------------------------------- symbol specifications
def test_complete_symbol_spec_is_loaded():
    c, _ = make()
    c.connect()
    spec = c.load_symbol_spec("EURUSD")
    assert spec.symbol == "EURUSD" and spec.tick_value == 1.0 and spec.volume_step == 0.01


def test_loss_side_tick_value_is_preferred():
    c, fake = make()
    fake.symbols["EURUSDm"] = good_symbol(trade_tick_value=1.0, trade_tick_value_loss=1.2)
    c.connect()
    assert c.load_symbol_spec("EURUSD").tick_value == 1.2


@pytest.mark.parametrize("override,text", [
    ({"trade_tick_value": 0.0, "trade_tick_value_loss": 0.0}, "trade_tick_value"),
    ({"point": 0.0}, "point"),
    ({"trade_tick_size": float("nan")}, "trade_tick_size"),
    ({"volume_step": 0.0}, "volume_step"),
    ({"volume_min": 200.0}, "volume_min is larger"),
    ({"volume_min": 0.015}, "multiple"),
    ({"trade_contract_size": 0.0}, "trade_contract_size"),
    ({"trade_stops_level": -1}, "trade_stops_level"),
    ({"digits": -1}, "digits"),
    ({"trade_mode": 0}, "not fully enabled"),
    ({"trade_mode": 3}, "not fully enabled"),
])
def test_incomplete_or_unsafe_symbol_specs_are_rejected(override, text):
    c, fake = make()
    fake.symbols["EURUSDm"] = good_symbol(**override)
    c.connect()
    with pytest.raises(MT5SymbolError, match=text):
        c.load_symbol_spec("EURUSD")


def test_missing_field_is_rejected():
    info = good_symbol()
    del info.volume_step
    assert any("missing field" in p for p in validate_symbol_info(info))


def test_unknown_symbol_and_unselectable_symbol_are_rejected():
    c, fake = make()
    c.connect()
    with pytest.raises(MT5SymbolError):
        c.load_symbol_spec("GBPUSD")               # not at the broker
    fake.select_ok = False
    with pytest.raises(MT5SymbolError):
        c.load_symbol_spec("EURUSD")


def test_load_all_specs_collects_errors_without_stopping():
    c, fake = make(symbols={"EURUSD": "EURUSDm", "GBPUSD": "GBPUSDm"})
    c.connect()
    specs, errors = c.load_all_specs()
    assert "EURUSD" in specs and "GBPUSD" in errors


# ---------------------------------------------------------------- disconnections and failed requests
def test_read_requests_are_retried():
    c, fake = make()
    fake.rates_failures = 2
    c.connect()
    df = c.get_candles("EURUSD", "H1", 5)
    assert len(df) == 5 and fake.rates_calls == 3


def test_read_request_gives_up_after_retries():
    c, fake = make()
    fake.rates_failures = 99
    c.connect()
    with pytest.raises(MT5RequestError, match="3 attempt"):
        c.get_candles("EURUSD", "H1", 5)


def test_order_style_requests_are_never_retried():
    c, fake = make()
    calls = []

    def send_order():
        calls.append(1)
        return None                                  # simulated timeout / failure

    with pytest.raises(MT5RequestError, match="1 attempt"):
        c.request("order_send", send_order, idempotent=False)
    assert len(calls) == 1


def test_exceptions_inside_requests_are_contained():
    c, _ = make()

    def boom():
        raise RuntimeError("terminal crashed")

    with pytest.raises(MT5RequestError, match="terminal crashed"):
        c.request("anything", boom)


def test_dropped_connection_recovers_with_reverification():
    c, fake = make()
    c.connect()
    fake.terminal.connected = False

    original = fake.initialize

    def initialize_and_heal(**kw):
        fake.terminal.connected = True
        return original(**kw)

    fake.initialize = initialize_and_heal
    assert c.ensure_connected().login == 123456


def test_unrecoverable_disconnect_trips_kill_switch(tmp_path):
    ks = KillSwitch(tmp_path / "ks.json")
    c, fake = make(kill_switch=ks, kill_on_failure=True)
    c.connect()
    fake.terminal.connected = False
    fake.init_failures = 99
    with pytest.raises(MT5ConnectionError):
        c.ensure_connected()
    assert ks.is_active()


def test_account_switched_mid_session_is_caught_and_locks_trading(tmp_path):
    ks = KillSwitch(tmp_path / "ks.json")
    c, fake = make(kill_switch=ks, kill_on_failure=True)
    c.connect()
    fake.account.login = 424242                       # someone logged the terminal into another account
    with pytest.raises(MT5AccountError, match="wrong account"):
        c.get_candles("EURUSD", "H1", 5)
    assert ks.is_active() and fake.shutdown_calls >= 1


def test_permission_lost_mid_session_trips_kill_switch_in_trading_mode(tmp_path):
    ks = KillSwitch(tmp_path / "ks.json")
    c, fake = make(kill_switch=ks, kill_on_failure=True)
    c.connect(require_trading=True)
    fake.terminal.trade_allowed = False
    with pytest.raises(MT5PermissionError):
        c.ensure_connected(require_trading=True)
    assert ks.is_active()


def test_failures_do_not_trip_kill_switch_unless_enabled(tmp_path):
    ks = KillSwitch(tmp_path / "ks.json")
    c, fake = make(FakeMT5(server="Other"), kill_switch=ks, kill_on_failure=False)
    with pytest.raises(MT5AccountError):
        c.connect()
    assert not ks.is_active()


# ---------------------------------------------------------------- candle data checks
def test_corrupt_candles_are_rejected():
    c, fake = make()
    c.connect()
    bad = good_rates()
    bad["high"][2] = 1.0                              # high below low
    fake.rates = bad
    with pytest.raises(MT5RequestError, match="high below its low"):
        c.get_candles("EURUSD", "H1", 5)
    dup = good_rates()
    dup["time"][3] = dup["time"][2]
    fake.rates = dup
    with pytest.raises(MT5RequestError, match="strictly increasing"):
        c.get_candles("EURUSD", "H1", 5)


def test_validate_candles_rejects_nan_and_nonpositive():
    import pandas as pd
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                       "open": [1.0, np.nan], "high": [1.1, 1.1], "low": [0.9, 0.9], "close": [1.0, 1.0]})
    with pytest.raises(MT5RequestError, match="missing values"):
        validate_candles(df)
    df.loc[1, "open"] = -1.0
    with pytest.raises(MT5RequestError, match="non-positive"):
        validate_candles(df)


def test_unsupported_timeframe_is_refused():
    c, _ = make()
    c.connect()
    with pytest.raises(ValueError):
        c.get_candles("EURUSD", "M5", 5)
