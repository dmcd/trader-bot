# Strategy Runner Decomposition TODO

Context: `trader_bot/strategy_runner.py` is a 1.8k+ line monolith mixing orchestration, exchange I/O, risk gating, plan management, telemetry, and persistence. The goal is to split it into focused services with clear contracts and test coverage while avoiding blocking SQLite calls and ccxt calls in the main loop.

## Work Breakdown (Task 1 from technical_review.md)
- [ ] Establish baseline map
  - [ ] Trace current responsibilities inside `StrategyRunner` (init, run loop, reconciliation, trade handling, risk, telemetry) and list the concrete call graph into `TradingDatabase`, `GeminiTrader`, `RiskManager`, `PlanMonitor`, `DataFetchCoordinator`, and `LLMStrategy`.
  - [ ] Document current async vs sync paths (ccxt async calls, DB writes, telemetry logger) and where blocking happens.
  - [ ] Inventory the existing tests that cover runner behavior (see `tests/test_*runner*`, trade sync, OHLCV capture, slippage guard) to understand what contracts must be preserved.

- [ ] Define target service boundaries and interfaces
  - [ ] Draft lightweight interfaces/protocols for:
    - [ ] `ExchangeGateway` (connect, order placement, open orders, positions, equity, OHLCV, trades) wrapping ccxt and handling reconnect/backoff.
    - [ ] `MarketDataService` (poll tickers/OB/trades, normalize snapshots, cache/dedupe, publish to consumers; likely adapts `DataFetchCoordinator`).
    - [ ] `RiskService` (exposure tracking, stacking guards, min/max order value buffering, RR filters, daily loss state) backed by `RiskManager` but detached from runner state mutations.
    - [ ] `PlanService` (plan lifecycle, trail/age flattening; adapt `PlanMonitor`), with pure functions for stop/target decisions.
    - [ ] `Persistence/TelemetryService` (structured telemetry emit + DB writes off the main loop; optionally a queue + worker to avoid synchronous SQLite hits).
    - [ ] `ExecutionService` (orders + fee estimation + holdings/PnL accounting + cost tracking).
  - [ ] Define the data contracts exchanged between services (market snapshot schema, order result schema, risk check result, plan record shape) and write type hints for them.

- [ ] Carve out execution and exchange I/O
  - [ ] Introduce `exchange_gateway.py` (or similar) that encapsulates ccxt client, reconnect logic, pause windows, and order/trade fetch helpers currently embedded in runner.
  - [ ] Move fee estimation + `_update_holdings_and_realized` into an `ExecutionService` that depends only on the gateway and cost tracker; keep a deterministic path for unit tests.
  - [ ] Add adapter methods so the existing runner uses the new gateway without changing strategy/plan code initially.

- [ ] Isolate data/telemetry persistence
  - [ ] Extract DB writes (trades, plans, market data, equity snapshots, stats cache, telemetry) into a persistence helper.

- [ ] Separate market data/provider logic
  - [ ] Reuse `DataFetchCoordinator` behind a `MarketDataProvider` interface shared by runner and `LLMStrategy` to eliminate duplicate fetches/normalization.
  - [ ] Provide deterministic stubs/mocks for unit tests and make rate-limit/dedup concerns local to this service.
  - [ ] Move OHLCV capture/pruning into this service with configurable timeframes and spacing; keep retention controls.

- [ ] Runner orchestration refactor
  - [ ] Shrink `StrategyRunner` to orchestration: session bootstrap, scheduling tasks, delegating to services, and high-level control flow (pause, kill switch).
  - [ ] Replace direct DB mutations with service calls; ensure restart recovery uses gateway snapshots + persistence service.
  - [ ] Clarify concurrency model (what runs in loop vs background tasks) and ensure shutdown/cleanup cancels tasks safely.

- [ ] Risk/plan wiring cleanup
  - [ ] Route risk checks through the new `RiskService` instead of touching `RiskManager` state from multiple places.
  - [ ] Ensure plan monitoring uses the shared market snapshot + exposure view; remove duplicate price lookups per plan.
  - [ ] Define unit tests for RR filter, stacking guard, exposure caps using pure functions.

- [ ] Testing and migration
  - [ ] Add unit tests per service (gateway behavior with stubbed ccxt, persistence worker, risk filters, plan closure logic, holdings/PnL math).
  - [ ] Provide integration smoke tests for the refactored runner using fakes/mocks (similar to `tests/test_runner_integration_stub.py`).
  - [ ] Plan incremental migration: introduce services behind facades first, then gradually move call sites and deprecate old helper methods; keep tests green at each step.

- [ ] Observability and config follow-up
  - [ ] Centralize telemetry emission (bot_actions + structured telemetry) in the new service; include health state updates.
  - [ ] Review config flags in `config.py` for new services (poll cadence, queue sizes, reconnect/backoff) and document defaults.
