from backend.market_data.mt5_connector import MT5Connector, MT5Error

c = MT5Connector()
try:
    info = c.connect(require_trading=False)
    print("ACCOUNT VERIFIED:", info.login, "|", info.server, "|", c._mode_name(info.trade_mode))
    rep = c.last_permissions
    print("TRADING PERMISSIONS OK:", rep.trading_ok)
    for issue in rep.issues:
        print("   -", issue)
    specs, errors = c.load_all_specs()
    for name, s in specs.items():
        print(f"{name}: OK  point={s.point} tick_value={s.tick_value} lots={s.volume_min}-{s.volume_max} "
              f"step={s.volume_step} stops_level={s.stops_level_points}")
    for name, problem in errors.items():
        print(f"{name}: PROBLEM  {problem}")
except MT5Error as exc:
    print("REJECTED:", type(exc).__name__, "-", exc)
finally:
    c.disconnect()
