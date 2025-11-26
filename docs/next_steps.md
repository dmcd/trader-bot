## Trading Bot Next Steps

### Strategic priorities
- Prove reliability before sizing: stabilize data/tooling, harden risk checks, and run paper sessions with telemetry replay to validate LLM/tool loop.
- Reduce prompt bloat and move to deterministic guards: let tools supply data; keep Python in charge of risk, sizing, and execution.
- Build evaluation loop: backtest/sim against recorded market data and replay LLM/tool envelopes to measure edge vs. costs.

### Execution checklist
- [ ] **Symbol allowlist + tool guardrails**: enforce allowed symbols/venues in tool requests; reject disallowed params and surface errors to LLM.
- [ ] **Per-tool rate limits**: config-driven throttle for market_data/order_book/trades to avoid LLM thrash; log drops.
- [ ] **Telemetry persistence**: store tool requests/responses (with trace_id) in DB for replay/analysis; keep JSON byte size recorded.
- [ ] **Runner integration polish**: decision prompt should state tool_responses supersede inline snapshots; strip any legacy multi-TF summaries when tools are used.
- [ ] **Risk overlays**: enforce min RR, slippage/vol scaling, and position stacking rules at execution; align with regime flags.
- [ ] **Integration test (runner loop)**: simulate planner→tool→decision within the runner, asserting clamps/truncation flags propagate.
- [ ] **Backtest/sim harness**: use ccxt historical data + DB schema to replay LLM decisions with deterministic fills; compute EV after fees/latency.
- [ ] **LLM cost guard**: cap planner/decision frequency and budget per session; fail-safe to HOLD on repeated errors.

### Opinionated assessment
- Strengths: solid logging scaffolding (telemetry + traces), move to tool-driven data (full books, multi-timeframe OHLCV), and risk primitives already present (caps, cooldowns). The trades→candles fallback reduces exchange gaps.
- Gaps: no proof of edge yet—LLM decisions are untested vs. baseline strategies. Execution layer still thin on slippage/latency modeling. IB support remains partial until ticks/trades are wired. Tool allowlists/rate limits need tightening to be production-safe.
- Path to profitability: decouple “idea generation” (LLM narrative/tool requests) from “order auth” (deterministic filters on RR, vol-adjusted sizing, slippage guard). Build a replay harness to measure Sharpe/EV before risking capital. Keep session-level budget for LLM cost and reject churn.

### Suggested immediate steps
1) Add symbol/param allowlists and per-tool rate limits; persist tool envelopes to DB.  
2) Implement runner-level integration test and a basic backtest harness using recorded ccxt data.  
3) Layer deterministic overlays (min RR, slippage check vs. decision price, anti-stack rules) before execution.  
4) Run paper sessions with telemetry review; optionally add a simple pause/HOLD control once tools are stable.  
