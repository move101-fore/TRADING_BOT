import json
from dataclasses import asdict
from pathlib import Path

from backend import config
from backend.market_data import mt5_connector as mt5c

OUT = Path("data") / "symbols.json"


def main():
    mt5c.connect()
    try:
        specs, errors = mt5c.load_all_specs()
    finally:
        mt5c.disconnect()

    for name, spec in specs.items():
        print(f"{name}: OK  point={spec.point} tick_value={spec.tick_value} "
              f"lots={spec.volume_min}-{spec.volume_max} step={spec.volume_step} "
              f"stops_level={spec.stops_level_points}")
    for name, problem in errors.items():
        print(f"{name}: PROBLEM  {problem}")

    if errors:
        print("\nNot saved: at least one symbol failed validation. The existing file was left unchanged.")
        raise SystemExit(1)

    data = {}
    for name, spec in specs.items():
        d = asdict(spec)
        d["broker_symbol"] = config.SYMBOLS.get(name, name)
        data[name] = d
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
