from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backend.market_data import history
from backend.market_data.history import (coverage_stats, download_since, gap_report, load, read_manifest,
                                         update_manifest, windows)
from backend.market_data.mt5_connector import MT5Error

DTYPE = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
         ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]


class FakeAPI:
    TIMEFRAME_H1 = 16385

    def __init__(self, missing_windows=0, first_call_fails=False):
        self.calls = 0
        self.first_call_fails = first_call_fails
        self.select_calls = []

    def symbol_select(self, name, flag):
        self.select_calls.append(name)
        return True

    def copy_rates_range(self, broker, tf, a, b):
        self.calls += 1
        if self.first_call_fails and self.calls == 1:
            return None
        hours = int((b - a).total_seconds() // 3600)
        rows = [(int((a + timedelta(hours=h)).timestamp()), 1.1, 1.101, 1.099, 1.1, 10, 8, 0) for h in range(hours)]
        return np.array(rows, dtype=DTYPE)


class FakeConn:
    def __init__(self, api):
        self.api = api

    def broker_symbol(self, s):
        return s + "m"


def test_windows_cover_the_range_without_overlap():
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2022, 4, 1, tzinfo=timezone.utc)
    w = list(windows(start, end))
    assert w[0][0] == start and w[-1][1] == end
    assert all(w[i][1] == w[i + 1][0] for i in range(len(w) - 1))


def test_download_drops_the_forming_candle_and_duplicates():
    api = FakeAPI()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 10, 5, 30, tzinfo=timezone.utc)
    df, empty = download_since(FakeConn(api), "EURUSD", "H1", start, end, sleep=lambda s: None)
    assert df["time"].is_unique and df["time"].is_monotonic_increasing
    assert df["time"].iloc[-1] <= pd.Timestamp(end) - pd.Timedelta(hours=1)
    assert api.select_calls == ["EURUSDm"]


def test_download_retries_when_history_is_still_syncing():
    api = FakeAPI(first_call_fails=True)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df, empty = download_since(FakeConn(api), "EURUSD", "H1", start, start + timedelta(days=5), sleep=lambda s: None)
    assert len(df) > 0 and empty == 0


def test_download_with_no_data_raises():
    class Empty(FakeAPI):
        def copy_rates_range(self, *a):
            return None

    with pytest.raises(MT5Error):
        download_since(FakeConn(Empty()), "EURUSD", "H1", datetime(2024, 1, 1, tzinfo=timezone.utc),
                       datetime(2024, 1, 20, tzinfo=timezone.utc), sleep=lambda s: None)


def test_gap_report_ignores_weekends_and_flags_real_gaps():
    # Mon 2024-01-01 .. Fri evening, normal weekend, then a missing Tuesday-Wednesday
    times = list(pd.date_range("2024-01-01 00:00", "2024-01-05 20:00", freq="h", tz="UTC"))
    times += list(pd.date_range("2024-01-07 22:00", "2024-01-09 06:00", freq="h", tz="UTC"))
    times += list(pd.date_range("2024-01-10 18:00", "2024-01-12 12:00", freq="h", tz="UTC"))
    df = pd.DataFrame({"time": times})
    gaps, by_year = gap_report(df, "H1")
    assert len(gaps) == 1 and abs(gaps.iloc[0]["hours"] - 36) < 1e-9          # the missing Wednesday, not the weekend
    assert by_year.iloc[0]["year"] == 2024 and by_year.iloc[0]["gaps"] == 1


def test_gap_report_on_clean_data_is_empty():
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")})
    gaps, by_year = gap_report(df, "H1")
    assert gaps.empty and by_year.empty


def test_empty_monthly_chunk_is_an_error_unless_explicitly_allowed():
    class MissingMiddle(FakeAPI):
        def copy_rates_range(self, broker, tf, a, b):
            if a.month == 2:
                return None
            return super().copy_rates_range(broker, tf, a, b)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 3, 15, tzinfo=timezone.utc)
    with pytest.raises(MT5Error, match="incomplete"):
        download_since(FakeConn(MissingMiddle()), "EURUSD", "H1", start, end, sleep=lambda s: None)
    df, empty = download_since(FakeConn(MissingMiddle()), "EURUSD", "H1", start, end,
                               sleep=lambda s: None, allow_missing=True)
    assert empty == 1 and len(df) > 0


def test_coverage_stats_measures_missing_time():
    clean = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")})
    assert coverage_stats(clean, "H1")["missing_pct"] == 0.0
    times = list(pd.date_range("2024-01-01 00:00", "2024-01-02 12:00", freq="h", tz="UTC"))
    times += list(pd.date_range("2024-01-04 06:00", "2024-01-05 20:00", freq="h", tz="UTC"))
    s = coverage_stats(pd.DataFrame({"time": times}), "H1")
    assert s["gaps"] == 1 and abs(s["missing_hours"] - 41.0) < 1e-9          # 42h between bars, minus the normal 1h step
    assert 45 < s["missing_pct"] < 55 and s["bad_months"] == []              # 41h is under the 48h month alert


def test_load_refuses_unverified_or_incomplete_history(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DATA_DIR", tmp_path)
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
                       "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "tick_volume": 1, "spread": 8})
    df.to_csv(history.path_for("EURUSD", "H1"), index=False, compression="gzip")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(MT5Error, match="no completeness record"):
        load("EURUSD", "H1", manifest_path=manifest)
    update_manifest("EURUSD_H1", {"missing_pct": 9.0}, manifest)
    with pytest.raises(MT5Error, match="refusing"):
        load("EURUSD", "H1", manifest_path=manifest)
    update_manifest("EURUSD_H1", {"missing_pct": 1.0}, manifest)
    assert len(load("EURUSD", "H1", manifest_path=manifest)) == 48
    assert len(load("EURUSD", "H1", require_complete=False, manifest_path=tmp_path / "none.json")) == 48
    assert read_manifest(manifest)["EURUSD_H1"]["missing_pct"] == 1.0
