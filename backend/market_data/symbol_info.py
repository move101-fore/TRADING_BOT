import json
from pathlib import Path

import MetaTrader5 as mt5

from backend import config
from backend.market_data import mt5_connector as mt5c

OUT = Path("data") / "symbols.json"


def main():
    mt5c.connect()
    info = {}
    try:
        for name, broker in config.SYMBOLS.items():
            mt5.symbol_select(broker, True)
            s = mt5.symbol_info(broker)
            if s is None:
                print(f"{name}: not found ({broker})")
                continue
            info[name] = {
                "broker_symbol": broker,
                "point": s.point,
                "digits": s.digits,
                "tick_size": s.trade_tick_size,
                "tick_value": s.trade_tick_value,
                "contract_size": s.trade_contract_size,
                "volume_min": s.volume_min,
                "volume_step": s.volume_step,
            }
            print(f"{name}: point={s.point} digits={s.digits} contract={s.trade_contract_size}")
    finally:
        mt5c.disconnect()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
