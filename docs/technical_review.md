# Technical Review

## Bugs / Correctness Risks
- [x] `trader_bot/strategy_runner.py:740-836` references `regime_flags` inside `_monitor_trade_plans` without defining it in that scope. The NameError is swallowed by the `try/except`, so volatility-aware trailing never actually runs and the code hides the bug instead of applying the intended logic.
- [x] `trader_bot/strategy_runner.py:543-569` executes partial close orders but never shrinks or annotates the underlying trade plan. The plan remains at full size, so the monitor can attempt to close the original size again and double-sell after a partial close.
- [x] `trader_bot/strategy_runner.py:1509-1520` hard-codes `symbol = 'BTC/USD'` in the main loop. Any open positions or orders on other symbols are ignored by data fetch, risk checks, and decisioning, so the bot can miss risk exposure or orphaned orders.
- [ ] `trader_bot/risk_manager.py:92-105` only applies the order value buffer when the order already exceeds `MAX_ORDER_VALUE`. Orders just under the cap are not trimmed to `MAX_ORDER_VALUE - ORDER_VALUE_BUFFER` despite the log messaging, so we can exceed the intended safety buffer.
- [ ] `trader_bot/database.py:268-272` adds a unique index on `sessions(bot_version)`, forcing a single session per version. New runs overwrite the same session rather than creating per-day/per-run rows, breaking daily loss accounting and making historical session analysis impossible.
- [ ] `trader_bot/strategy_runner.py:1692-1786` treats `_apply_fill_to_session_stats` as potentially awaitable even though it always returns `None`. This is a code smell that hides async/sync confusion and makes error handling around session stats brittle.
- [ ] `trader_bot/strategy_runner.py:240-321` rebuilds holdings and stats from exchange trades but does not handle fee fields or errors defensively; malformed trades can zero holdings and stats without alerts.

## Risk & Safety Gaps
- [ ] Exposure calculations depend on prices in stored positions, but `GeminiTrader.get_positions_async` returns positions with `avg_price=None` and no live pricing. Until another lookup fills prices, `risk_manager.get_total_exposure` can undercount exposure and allow oversizing.
- [ ] Trade sync deduplication (`trader_bot/strategy_runner.py:1138-1237`) keeps `processed_trade_ids` only in memory. After restarts the same fills can be re-logged and PnL inflated because on-disk dedup relies on raw SQL checks against `trades.trade_id` and skips client order attribution for anything missing `_client_oid`.
- [ ] `_capture_ohlcv` runs every loop for four timeframes with no retention or throttling, so `ohlcv_bars` and prompts can grow unbounded and slow SQLite/telemetry.
- [ ] Kill-switch behavior sets `_kill_switch` but keeps the loop running (`trader_bot/strategy_runner.py:1370-1395`, 1840-1857), continuing to log/loop indefinitely instead of exiting cleanly or surfacing an operator signal.

## Architecture / Maintainability
- [ ] `trader_bot/strategy_runner.py` is a 1.8k+ line monolith mixing orchestration, risk gating, exchange reconciliation, plan management, telemetry, and fee/PnL accounting. The main loop performs synchronous SQLite writes and blocking ccxt calls, making it hard to test and reason about. Consider splitting into services (data fetching, execution, risk gating, plan monitor, telemetry/persistence) with clearer contracts and unit coverage.
- [ ] Strategy execution, risk, and DB concerns are tightly coupled: the runner calls DB methods directly from many paths, while `RiskManager` holds mutable state but depends on the runner to hydrate prices. A clearer boundary (e.g., an `ExposureService` fed by normalized market snapshots) would simplify tests and reduce duplicated price lookups.
- [ ] The LLM strategy builds prompts and tools inline but the runner still fetches its own market data and OHLCV separately. A single data provider interface would reduce duplication and make unit tests deterministic.
- [ ] SQLite access uses a shared connection with `check_same_thread=False` but no locking; multiple async tasks call into the DB concurrently (equity logging, trade syncing, plan updates) which risks intermittent failures and partial writes.

## Observability / Ops
- [ ] No retention or pruning on telemetry tables (`llm_traces`, `market_data`, `ohlcv_bars`, `commands`), so long-running sessions can bloat the DB and slow prompt/memory retrieval.
- [ ] Health-state telemetry logs exchange/tool circuits but does not emit metrics for key risk counters (daily loss, exposure headroom, fee ratio) or LLM budget state, making it difficult to monitor drift or cost leaks.
- [ ] Equity sanity checks (`trader_bot/strategy_runner.py:117-205`) are info-level only and not surfaced to health state; discrepancies between broker equity and reconstructed PnL can go unnoticed.

## Testing Gaps / Suggestions
- [ ] Add regression tests for partial close plan accounting to ensure plan size/status is updated and monitor does not double-close.
- [ ] Add coverage for volatility-aware trailing logic in `_monitor_trade_plans` to catch the undefined `regime_flags` reference and validate trail behavior across regimes.
- [ ] Add integration coverage for multi-symbol exposure handling (positions without prices, open orders across symbols) and ensure the runner does not hard-code a single symbol.
- [ ] Add tests for the order value buffer to assert it clamps to `MAX_ORDER_VALUE - ORDER_VALUE_BUFFER` and logs appropriately.
- [ ] Add persistence tests around session creation to ensure new runs create new sessions rather than reusing the same `bot_version` row.
