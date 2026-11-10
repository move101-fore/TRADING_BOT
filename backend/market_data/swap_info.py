import json
from pathlib import Path

from backend import config
from backend.market_data.mt5_connector import MT5Connector, MT5Error

OUT = Path("data") / "swaps.json"


def main():
    c = MT5Connector()
    try:
        c.connect()
        data = {}
        for name in config.SYMBOLS:
            info = c.api.symbol_info(c.broker_symbol(name))
            if info is None:
                print(f"{name}: not found")
                continue
            data[name] = {
                "swap_long": info.swap_long, "swap_short": info.swap_short,
                "swap_mode": info.swap_mode, "triple_day": info.swap_rollover3days,
            }
            print(f"{name}: long={info.swap_long} short={info.swap_short} "
                  f"mode={info.swap_mode} triple_day={info.swap_rollover3days}")
    except MT5Error as exc:
        print("REJECTED:", exc)
        raise SystemExit(1)
    finally:
        c.disconnect()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
