from backend.market_data import mt5_connector as c
import MetaTrader5 as mt5

c.connect()
names = [s.name for s in mt5.symbols_get()]
print("total symbols:", len(names))
for key in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "XAU"]:
    print(key, "->", [n for n in names if key in n][:6])
c.disconnect()
