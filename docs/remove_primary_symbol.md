# Removing the primary_symbol shortcut

- [ ] Audit current primary-only logic (market health, liquidity gate, OHLCV capture, context/regime flags, slippage decision price) and list call sites that assume a single symbol.
- [ ] Expand market data handling so freshness/health checks, telemetry, and risk headroom are evaluated per symbol (include tests for multi-symbol health paths).
- [ ] Run liquidity/microstructure filters per symbol and gate trading decisions accordingly (add tests covering wide spreads and insufficient top-of-book notional across symbols).
- [ ] Capture and persist OHLCV for every active symbol each loop, with retention and spacing respected (test multi-symbol OHLCV capture).
- [ ] Build context summaries and regime flags per symbol for the LLM prompt; ensure the prompt enumerates per-symbol flags and recent bars without exceeding token budgets (add prompt-construction tests).
- [ ] Refactor strategy orchestration/execution to handle per-symbol decisions (including tool planning/execution) without relying on a primary symbol; ensure execution uses symbol-scoped market data and price lookups (add integration-style tests or fakes).
- [ ] Update bot action logging to include symbol and latest price in HOLD/other decision lines for clarity; adjust telemetry assertions if needed (add log-format tests).
- [ ] End-to-end test pass (`python -m pytest`) and cleanup (docs/config notes if any new tunables are introduced).
