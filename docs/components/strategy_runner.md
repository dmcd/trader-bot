# Strategy Runner

The main orchestration loop (`strategy_runner.py`) pulls market data, asks the LLM for actions, enforces risk, and routes orders.

## Flow
- Initializes exchange, DB, cost tracker, risk manager, and strategy.
- Main loop: fetch market data → build trading context → call `LLMStrategy.decide` → risk checks → place order(s) → log/telemetry → sleep.
- Periodic sync: rebuild session stats, refresh open orders/positions, sync recent trades from exchange.

## Orders vs Plans
- Plans are only created when an entry has a stop or target. Signals without stop/target still place orders but skip plan creation (no plan to monitor later).
- Flatten/partial-close flows (`CLOSE_POSITION`, `PARTIAL_CLOSE`) deliberately place orders without plans.
- Plans are tracked in `trade_plans` and monitored for stop/target hits, age, or day-end. They are **not** tied to order cancel events. Canceling an order leaves the plan open until a monitor condition closes it.
- Open orders are snapshots from the exchange and are not linked to plan_id. The bot stores a reason per order_id (`order_reasons`) for sync, but does not persist plan_id → order_id mapping.
- Trade sync (`sync_trades_from_exchange`) now filters to our `clientOrderId` prefix; trades without the prefix are ignored.

## Signals and actions
- `LLMStrategy` emits actions like BUY/SELL (entry), HOLD, CLOSE_POSITION, PARTIAL_CLOSE, UPDATE_PLAN, PAUSE_TRADING.
- Risk guards include max exposure, order value, spread, slippage, trade spacing, and stacking checks.

## Telemetry and logging
- Writes `bot_actions_logger` entries for major actions.
- Telemetry includes status, prices, slippage, and order_result payloads.

## Practical tips
- If you want every entry to have a plan, require the LLM to include stop/target or add a fallback to create a plan even without them.
- To auto-close plans on cancel, wire order cancel events to `update_trade_plan_status` once a symbol has no open orders/positions.
