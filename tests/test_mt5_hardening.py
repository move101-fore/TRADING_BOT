"""
Release A, batch 2: MT5 connector and configuration hardening.

Rules under test:
  * a trading connection can never exist without an armed kill switch;
  * a failure to trip the kill switch is never silent;
  * raw exceptions from the MT5 package become MT5Error subclasses (fail closed), never leak;
  * garbage from the broker (bad candles, odd symbol records) is a clean MT5 error;
  * a bad .env value stops start-up instead of being accepted.
"""
import importlib
import logging

import numpy as np
import pytest

from backend import config
from backend.market_data.mt5_connector import (
    MT5AccountError, MT5ConnectionError, MT5PermissionError, MT5RequestError, MT5SymbolError,
)
from backend.risk.kill_switch import KillSwitch
from test_mt5_safety import FakeMT5, good_rates, make


def armed(tmp_path, fake=None, **kw):
    ks = KillSwitch(tmp_path / "ks.json")
    c, fake = make(fake, kill_switch=ks, kill_on_failure=True, **kw)
    return c, fake, ks


# ------------------------------------------------------------------ trading needs an armed kill switch
@pytest.mark.parametrize("kill,on_failure", [(None, False), (None, True), ("ks", False)])
def test_trading_connection_is_refused_without_an_armed_kill_switch(tmp_path, kill, on_failure):
    ks = KillSwitch(tmp_path / "ks.json") if kill else None
    c, fake = make(kill_switch=ks, kill_on_failure=on_failure)
    with pytest.raises(MT5PermissionError, match="kill switch"):
        c.connect(require_trading=True)
    assert fake.init_calls == 0 and not c.connected          # never even touched the terminal


@pytest.mark.parametrize("kill,on_failure", [(None, False), ("ks", False)])
def test_ensure_connected_for_trading_is_refused_without_an_armed_kill_switch(tmp_path, kill, on_failure):
    ks = KillSwitch(tmp_path / "ks.json") if kill else None
    c, _ = make(kill_switch=ks, kill_on_failure=on_failure)
    c.connect()                                              # read-only connection is fine
    with pytest.raises(MT5PermissionError, match="kill switch"):
        c.ensure_connected(require_trading=True)
    assert not c.connected                                   # and it was shut down


def test_trading_connection_works_when_kill_switch_is_armed(tmp_path):
    c, _, ks = armed(tmp_path)
    assert c.connect(require_trading=True).login == 123456
    assert not ks.is_active()


def test_read_only_use_does_not_need_a_kill_switch():
    c, fake = make()                                         # what history.py / data_audit.py do
    assert c.connect().login == 123456
    assert c.get_candles("EURUSD", "H1", 5) is not None


# ------------------------------------------------------------------ kill switch failure is never silent
class BrokenKill:
    def is_active(self):
        return True

    def activate(self, reason, source="manual"):
        raise OSError("disk full")


def test_kill_switch_write_failure_is_reported_not_swallowed(tmp_path, caplog):
    c, fake = make(FakeMT5(server="Other-Server"), kill_switch=BrokenKill(), kill_on_failure=True)
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(MT5AccountError) as info:             # the ORIGINAL error still surfaces
            c.connect()
    assert "disk full" in c.last_kill_error
    assert "disk full" in info.value.kill_switch_error
    assert any(r.levelno == logging.CRITICAL and "could NOT be activated" in r.getMessage()
               for r in caplog.records)
    assert fake.shutdown_calls >= 1 and not c.connected


def test_no_kill_error_recorded_when_kill_switch_works(tmp_path):
    c, _, ks = armed(tmp_path, FakeMT5(server="Other-Server"))
    with pytest.raises(MT5AccountError):
        c.connect()
    assert c.last_kill_error is None and ks.is_active()


# ------------------------------------------------------------------ raw exceptions become MT5 errors
def test_terminal_info_raising_mid_session_fails_closed(tmp_path):
    c, fake, ks = armed(tmp_path)
    c.connect()

    def boom():
        raise RuntimeError("IPC pipe broken")

    fake.terminal_info = boom
    with pytest.raises(MT5ConnectionError, match="terminal_info failed"):
        c.get_candles("EURUSD", "H1", 5)
    assert ks.is_active() and fake.shutdown_calls >= 1 and not c.connected


def test_account_info_raising_mid_session_fails_closed(tmp_path):
    c, fake, ks = armed(tmp_path)
    c.connect()

    def boom():
        raise ValueError("bad struct")

    fake.account_info = boom
    with pytest.raises(MT5ConnectionError, match="bad struct"):
        c.ensure_connected()
    assert ks.is_active()


def test_terminal_info_raising_during_permission_check_fails_closed(tmp_path):
    c, fake, _ = armed(tmp_path)

    def boom():
        raise RuntimeError("x")

    fake.terminal_info = boom
    with pytest.raises(MT5ConnectionError):
        c.check_permissions()


def test_last_error_raising_does_not_hide_the_connection_failure():
    fake = FakeMT5()
    fake.init_failures = 99

    def boom():
        raise RuntimeError("no last_error")

    fake.last_error = boom
    c, _ = make(fake, retries=2)
    with pytest.raises(MT5ConnectionError, match="2 attempts"):
        c.connect()


def test_validate_callback_that_raises_counts_as_invalid_data():
    c, _ = make()
    with pytest.raises(MT5RequestError):
        c.request("x", lambda: [1], validate=lambda r: 1 / 0)


# ------------------------------------------------------------------ garbage from the broker
def test_candles_that_cannot_be_parsed_are_a_clean_error():
    c, fake = make()
    c.connect()
    fake.rates = [1, 2, 3]                                   # no named columns at all
    with pytest.raises(MT5RequestError, match="could not be read"):
        c.get_candles("EURUSD", "H1", 5)


def test_candles_with_text_prices_are_a_clean_error():
    c, fake = make()
    c.connect()
    dtype = [("time", "i8"), ("open", "U4"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
             ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
    fake.rates = np.array([(1_700_000_000, "abcd", 1.1, 1.0, 1.05, 1, 1, 0)], dtype=dtype)
    with pytest.raises(MT5RequestError, match="could not be checked"):
        c.get_candles("EURUSD", "H1", 5)


def test_missing_timeframe_constant_is_a_clean_error():
    class NoH1(FakeMT5):
        TIMEFRAME_H1 = None

    c, _ = make(NoH1())
    c.connect()
    with pytest.raises(MT5RequestError, match="TIMEFRAME_H1"):
        c.get_candles("EURUSD", "H1", 5)


def test_symbol_record_that_blows_up_is_a_clean_error():
    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("corrupt record")

    c, fake = make()
    c.connect()
    fake.symbols["EURUSDm"] = Boom()
    with pytest.raises(MT5SymbolError, match="could not be read"):
        c.load_symbol_spec("EURUSD")


def test_good_candles_still_work():
    c, fake = make()
    c.connect()
    fake.rates = good_rates(7)
    assert len(c.get_candles("EURUSD", "H1", 7)) == 7


# ------------------------------------------------------------------ config validation
@pytest.fixture
def fresh_config(monkeypatch):
    for name in ("EXECUTION_MODE", "MT5_EXPECTED_MODE", "MT5_LOGIN"):
        monkeypatch.delenv(name, raising=False)

    def reload():
        return importlib.reload(config)

    yield monkeypatch, reload
    monkeypatch.undo()
    importlib.reload(config)                                 # leave the module in its normal state


def test_default_execution_mode_is_signal_only(fresh_config):
    _, reload = fresh_config
    c = reload()
    assert c.EXECUTION_MODE == "SIGNAL_ONLY" and c.MT5_EXPECTED_MODE == "DEMO"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_execution_mode_falls_back_to_signal_only(fresh_config, value):
    mp, reload = fresh_config
    mp.setenv("EXECUTION_MODE", value)
    assert reload().EXECUTION_MODE == "SIGNAL_ONLY"


def test_execution_mode_is_normalised(fresh_config):
    mp, reload = fresh_config
    mp.setenv("EXECUTION_MODE", " signal_only ")
    assert reload().EXECUTION_MODE == "SIGNAL_ONLY"


@pytest.mark.parametrize("value", ["LIVE", "AUTO", "DEMO", "SIGNAL-ONLY", "SIGNAL_ONLYY", "true", "1"])
def test_unknown_execution_mode_stops_start_up(fresh_config, value):
    mp, reload = fresh_config
    mp.setenv("EXECUTION_MODE", value)
    with pytest.raises(ValueError, match="EXECUTION_MODE") as e:
        reload()
    assert type(e.value).__name__ == "ConfigError"


@pytest.mark.parametrize("value", ["PAPER", "LIVE", "DEMOO"])
def test_unknown_account_mode_stops_start_up(fresh_config, value):
    mp, reload = fresh_config
    mp.setenv("MT5_EXPECTED_MODE", value)
    with pytest.raises(ValueError, match="MT5_EXPECTED_MODE") as e:
        reload()
    assert type(e.value).__name__ == "ConfigError"


def test_account_mode_is_normalised(fresh_config):
    mp, reload = fresh_config
    mp.setenv("MT5_EXPECTED_MODE", "real")
    assert reload().MT5_EXPECTED_MODE == "REAL"


@pytest.mark.parametrize("value", ["abc", "12.5", "12 34", "0x10"])
def test_bad_login_stops_start_up_with_a_clear_message(fresh_config, value):
    mp, reload = fresh_config
    mp.setenv("MT5_LOGIN", value)
    with pytest.raises(ValueError, match="MT5_LOGIN") as e:
        reload()
    assert type(e.value).__name__ == "ConfigError"


def test_login_is_parsed(fresh_config):
    mp, reload = fresh_config
    mp.setenv("MT5_LOGIN", " 123456 ")
    assert reload().MT5_LOGIN == 123456
    mp.setenv("MT5_LOGIN", "")
    assert reload().MT5_LOGIN == 0


def test_config_error_is_a_value_error():
    assert issubclass(config.ConfigError, ValueError)
