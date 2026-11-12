import numpy as np

from backend.market_data.data_audit import WEEK_BARS, candle_coverage, sample_week, tick_coverage


class FakeAPI:
    TIMEFRAME_H1, TIMEFRAME_M15, TIMEFRAME_M1 = 16385, 15, 1
    COPY_TICKS_ALL = -1

    def __init__(self, first_year):
        self.first_year = first_year

    def copy_rates_range(self, symbol, tf, start, end):
        if start.year < self.first_year:
            return None
        return np.zeros(WEEK_BARS["H1"] if tf == self.TIMEFRAME_H1 else WEEK_BARS["M15"] // 2)

    def copy_ticks_range(self, symbol, start, end, flags):
        if start.year < self.first_year:
            return None
        n = 100
        return np.array([(1.1, 1.1001)] * n, dtype=[("bid", "f8"), ("ask", "f8")])


def test_sample_week_is_a_monday_to_saturday_window():
    start, end = sample_week(2024)
    assert start.weekday() == 0 and (end - start).days == 5


def test_candle_coverage_reports_missing_years_and_partial_years():
    api = FakeAPI(first_year=2020)
    h1 = candle_coverage(api, "X", "H1", [2019, 2020])
    assert h1[2019] == 0.0 and h1[2020] == 1.0
    m15 = candle_coverage(api, "X", "M15", [2021])
    assert m15[2021] == 0.5


def test_tick_coverage_detects_missing_history_and_quotes():
    api = FakeAPI(first_year=2023)
    t = tick_coverage(api, "X", [2022, 2023])
    assert t[2022] == (0, False) and t[2023] == (100, True)
