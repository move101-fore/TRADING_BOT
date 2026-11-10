from backend.market_data.mt5_connector import MT5Connector, MT5Error

c = MT5Connector()
try:
    c.connect()
    for name in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAUUSD"]:
        info = c.api.symbol_info(c.broker_symbol(name))
        if info is None:
            print(name, "not found")
            continue
        print(f"{name}: swap_long={info.swap_long} swap_short={info.swap_short} "
              f"swap_mode={info.swap_mode} triple_swap_day={info.swap_rollover3days}")
except MT5Error as exc:
    print("REJECTED:", exc)
finally:
    c.disconnect()
