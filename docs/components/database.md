# Database

`database.py` manages SQLite persistence for sessions, trades, plans, positions, orders, telemetry, and caches.

## Key tables
- `sessions`: session metadata, totals, start/end balances.
- `trades`: executed trades (symbol, action, qty, price, fee, liquidity, reason, trade_id, timestamp).
- `trade_plans`: stop/target plans; status is independent of order cancels.
- `positions`: snapshot of current positions.
- `open_orders`: snapshot of exchange open orders (not linked to plans).
- `session_stats_cache`: aggregates for restart resilience.

## Helpers
- `create_trade_plan`, `update_trade_plan_prices/status`, `get_open_trade_plans`.
- `get_trades_for_session`, `get_trade_count`, `log_trade`, `log_estimated_fee`.
- `replace_open_orders`, `replace_positions` for snapshots.

## Tips
- Plans don’t auto-close on order cancel; close logic lives in runner monitors.
- Trade timestamps should be ISO/UTC; dashboard localizes them for display.
