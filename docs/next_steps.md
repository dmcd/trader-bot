## Project Review (LLM Day-Trading Bot)

### Snapshot
- LLMStrategy (Gemini/OpenAI) drives decisions with planner + decision turns; tool requests are validated/cached via DataFetchCoordinator; prompts include context summaries, plan caps, and regime flags.
- StrategyRunner enforces spacing, break-glass priority moves, RR/slippage/liquidity checks, and risk gating before execution through GeminiTrader (limit orders with maker-intent retry).
- RiskManager tracks exposure caps, order value/min size, daily loss (abs/%), pending order stacking, and max positions; CostTracker measures trading + LLM costs; TradingDatabase persists sessions, trades, plans, prompts, OHLCV, positions, and telemetry.
- Telemetry is rich (full prompts/traces, tool requests, bot_actions log), but operational guardrails for stale data, repeated errors, or restart recovery are still thin.

### Key Risks & Gaps
- Execution realism is coarse: fixed slippage pct, simple liquidity filter, no maker/taker economics simulation, and no latency/clock skew handling between decision and fill.
- Resilience/ops gaps: no circuit breaker on repeated exchange/tool failures, limited restart recovery for open orders vs. DB, no stale-data detection, and no heartbeat/alerting for manual ops.
- Risk breadth: exposure is gross/notional only (no correlation buckets or per-symbol headroom tuning), and stacking logic is limited to counts; plan lifecycle ties to runner, not exchange fills.
- LLM cost/context hygiene: spacing exists, but no per-turn byte budget, freshness tags, or delta summaries to prevent repetitive prompts; no guard against tool thrash or oversized payloads beyond coarse clamps.
- UX: dashboard shows status but lacks quick controls (pause, flatten, reduce caps) and visibility into LLM cost burn rate vs. session cap.

### Next Steps Checklist
- Reliability & Ops
  - [x] Add circuit breaker / auto-pause on consecutive exchange or tool failures; surface state in telemetry + dashboard.
  - [x] Tag data freshness/latency for ticker, books, and OHLCV; skip or down-weight stale feeds; add stale-feed alerts.
  - [x] Improve restart recovery: reconcile open orders/positions vs. exchange on startup and patch DB snapshots accordingly.
- Execution & Risk
  - [x] Replace fixed slippage pct with symbol-aware band driven by depth/vol; block fills when book notional is thin.
  - [x] Add maker/taker policy toggle (per symbol) with retries and fee modeling; record maker vs. taker in telemetry.
  - [x] Enforce min RR per plan with live price tolerance and auto-cancel of stale plans; trail stops using volatility-aware bands.
  - [x] Introduce simple correlation buckets (e.g., BTC/ETH majors vs. alts) to cap same-direction stacking across related symbols.
- LLM Context & Costs
  - [x] Add per-turn byte budget and trimming (plans, orders, trades, memory); include change-log deltas instead of full snapshots.
  - [x] Expose data freshness, spread/depth, and headroom explicitly in prompts; flag when plan caps or cooldowns block actions.
  - [x] Rate-limit tool requests per symbol/timeframe combo and deduplicate across planner turns; log tool thrash metrics.
  - [x] Track LLM cost burn rate vs. session budget and feed it into prompts plus dashboard alerts.
- Product & UX
  - [ ] Add integration test that runs StrategyRunner against a stub exchange + stub LLM to verify spacing, slippage, and risk gates.
  - [ ] Property tests for tool payload clamps (json size) and timeframe normalization; regression tests for plan trail-to-breakeven logic.
  - [ ] Backfill metrics validation: reconcile DB session stats vs. equity snapshots and report drift beyond threshold.
