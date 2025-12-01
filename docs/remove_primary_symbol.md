# Removing the primary_symbol shortcut

- [x] Audit current primary-only logic (market health, liquidity gate, OHLCV capture, context/regime flags, slippage decision price) and list call sites that assume a single symbol.
  - `strategy_runner.py`: selects `primary_symbol = symbols[0]` and uses it for health gating (`emit_market_health` -> now `evaluate_market_health`), liquidity filter (`_liquidity_ok`), OHLCV capture (`_capture_ohlcv`), slippage decision price, and loop continuation when the primary fetch fails.
  - `strategy.py` prompt build: seeds `symbol = available_symbols[0]`; only that symbol gets context summary, memory snapshot, recent OHLCV fetch, and regime flags in the LLM prompt, so other symbols lack per-symbol context/flags.
  - Telemetry/logs: decision logs omit symbol/price on HOLD; telemetry includes symbol but logs don’t surface it.
- [x] Expand market data handling so freshness/health checks, telemetry, and risk headroom are evaluated per symbol (include tests for multi-symbol health paths).
  - Market data fetch no longer short-circuits on the first symbol; per-symbol freshness is evaluated via `evaluate_market_health`, stale symbols are pruned, and the loop skips only when none are fresh.
  - Operational metrics now include per-symbol exposure details (via `RiskManager.compute_exposure`/`get_exposure_breakdown`), and health telemetry is emitted per symbol.
  - Tests added/updated: `tests/unit/trader_bot/services/test_strategy_orchestrator.py`, `tests/unit/trader_bot/test_manager.py`, `tests/integration/test_strategy_runner_control_paths.py`.
- [ ] Run liquidity/microstructure filters per symbol and gate trading decisions accordingly (add tests covering wide spreads and insufficient top-of-book notional across symbols).
- [ ] Capture and persist OHLCV for every active symbol each loop, with retention and spacing respected (test multi-symbol OHLCV capture).
- [ ] Build context summaries and regime flags per symbol for the LLM prompt; ensure the prompt enumerates per-symbol flags and recent bars without exceeding token budgets (add prompt-construction tests).
- [ ] Refactor strategy orchestration/execution to handle per-symbol decisions (including tool planning/execution) without relying on a primary symbol; ensure execution uses symbol-scoped market data and price lookups (add integration-style tests or fakes).
- [ ] Update bot action logging to include symbol and latest price in HOLD/other decision lines for clarity; adjust telemetry assertions if needed (add log-format tests).
- [ ] End-to-end test pass (`python -m pytest`) and cleanup (docs/config notes if any new tunables are introduced).
