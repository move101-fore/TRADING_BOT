from datetime import datetime, timedelta, timezone

from backend import config
from backend.market_data.mt5_connector import MT5Connector, MT5Error

# Expected bars in a full Monday-to-Friday sample week
WEEK_BARS = {"M1": 5 * 24 * 60, "M15": 5 * 24 * 4, "H1": 5 * 24}


def sample_week(year):
    """A Monday-to-Friday window in mid-March of the given year (UTC)."""
    start = datetime(year, 3, 10, tzinfo=timezone.utc)
    start -= timedelta(days=start.weekday())            # back to Monday
    return start, start + timedelta(days=5)


def candle_coverage(api, broker_symbol, tf_name, years):
    """Share of expected candles present in one sample week per year (0.0 to 1.0)."""
    tf = getattr(api, f"TIMEFRAME_{tf_name}")
    out = {}
    for year in years:
        start, end = sample_week(year)
        rates = api.copy_rates_range(broker_symbol, tf, start, end)
        n = 0 if rates is None else len(rates)
        out[year] = round(n / WEEK_BARS[tf_name], 2)
    return out


def tick_coverage(api, broker_symbol, years):
    """Tick count and whether bid/ask are present, for a one-hour sample per year."""
    out = {}
    for year in years:
        start, _ = sample_week(year)
        start = start + timedelta(days=1, hours=10)       # Tuesday 10:00 UTC
        ticks = api.copy_ticks_range(broker_symbol, start, start + timedelta(hours=1), api.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            out[year] = (0, False)
        else:
            has_quotes = bool((ticks["bid"] > 0).all() and (ticks["ask"] > 0).all())
            out[year] = (len(ticks), has_quotes)
    return out


def main():
    years = list(range(2018, datetime.now(timezone.utc).year + 1))
    c = MT5Connector()
    try:
        c.connect()
        for name in ("EURUSD", "XAUUSD"):
            broker = c.broker_symbol(name)
            c.api.symbol_select(broker, True)
            print(f"\n=== {name} ({broker}) ===")
            print("year     " + "".join(f"{y:>7}" for y in years))
            for tf in ("H1", "M15", "M1"):
                cov = candle_coverage(c.api, broker, tf, years)
                print(f"{tf:<9}" + "".join(f"{cov[y] * 100:>6.0f}%" for y in years))
            ticks = tick_coverage(c.api, broker, years)
            print("ticks/hr " + "".join(f"{ticks[y][0]:>7}" for y in years))
            print("bid+ask  " + "".join(f"{('yes' if ticks[y][1] else 'no'):>7}" for y in years))
        print("\nReading guide: 100% means a full sample week was available; 0% means no history that year.")
    except MT5Error as exc:
        print("REJECTED:", exc)
    finally:
        c.disconnect()


if __name__ == "__main__":
    main()
