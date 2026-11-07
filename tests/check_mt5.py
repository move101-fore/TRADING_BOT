
from backend.market_data import mt5_connector as mt5c

info = mt5c.connect()
print("Connected. Account:", info.login, "| Server:", info.server, "| Balance:", info.balance)

df = mt5c.get_candles("EURUSD", "H1", 500)
print("Candles received:", len(df))
print(df.head(3))
print(df.tail(3))

mt5c.disconnect()
 | Set-Content tests\check_mt5.py
