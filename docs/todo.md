# Strategy Runner Decomposition TODO

Context: `trader_bot/strategy_runner.py` is a 1.8k+ line monolith mixing orchestration, exchange I/O, risk gating, plan management, telemetry, and persistence. The goal is to split it into focused classes with their own unit test coverage.

## Checklist
- [ ] Draft a decomposition plan that maps existing responsibilities in `StrategyRunner` (run loop, health circuits, holdings, plan handling, data capture, reconciliation) to new services and public interfaces.
- [ ] Carve out an exchange health/circuit manager (covers `_record_exchange_failure`, `_record_tool_failure/_success`, pause/unpause, `_reconnect_bot`, `_is_stale_market_data`) with unit tests around streak thresholds and pause windows.
- [ ] Extract holdings and session accounting into a dedicated portfolio tracker (currently `_update_holdings_and_realized`, `_apply_trade_to_holdings`, `_apply_fill_to_session_stats`, `_rebuild_session_stats_from_trades`, `_apply_exchange_trades_for_rebuild`) and cover with tests for realized PnL, fee handling, and cache writes.
- [ ] Move signal/plan execution paths into a trade action handler (UPDATE_PLAN/PARTIAL_CLOSE/CLOSE_POSITION/PAUSE_TRADING plus sizing helpers like `_apply_order_value_buffer`, `_passes_rr_filter`, `_slippage_within_limit`, `_stacking_block`, `_liquidity_ok`) that can be tested without the main loop.
- [ ] Pull OHLCV capture and market data gating into a data service (using `_capture_ohlcv`, `_timeframe_to_seconds`, spacing/retention logic, market staleness checks) with coverage for spacing and pruning rules.
- [ ] Isolate reconciliation flows (`_reconcile_exchange_state`, `_reconcile_open_orders`, `sync_trades_from_exchange`) into a resync service with tests for filtering to our client IDs and DB snapshot replacement.
- [ ] Slim the main orchestrator: wrap command processing, risk budget checks, plan monitor coordination, and telemetry emission into a higher-level `StrategyOrchestrator` that wires the new services together; add harness-level tests for start/stop/cleanup sequences.
- [ ] Update architecture docs with the new module boundaries and how `strategy_runner` will delegate to them.
