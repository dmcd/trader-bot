## Trading Bot Next Steps

### Strategic priorities
- Prove reliability before sizing: stabilize data/tooling, harden risk checks, and run paper sessions with telemetry to validate LLM/tool loop.
- Reduce prompt bloat and move to deterministic guards: let tools supply data; keep Python in charge of risk, sizing, and execution.

### Execution checklist
- [ ] **Runner integration polish**: decision prompt should state tool_responses supersede inline snapshots; strip any legacy multi-TF summaries when tools are used.
- [ ] **Risk overlays**: enforce min RR, slippage/vol scaling, and position stacking rules at execution; align with regime flags.
- [ ] **Integration test (runner loop)**: simulate planner→tool→decision within the runner, asserting clamps/truncation flags propagate.
- [ ] **LLM cost guard**: cap planner/decision frequency and budget per session; fail-safe to HOLD on repeated errors.

### Opinionated assessment
- Strengths: solid logging scaffolding (telemetry + traces), move to tool-driven data (full books, multi-timeframe OHLCV), and risk primitives already present (caps, cooldowns). The trades→candles fallback reduces exchange gaps.
- Gaps: no proof of edge yet—LLM decisions are untested vs. baseline strategies. Execution layer still thin on slippage/latency modeling.
- Path to profitability: decouple “idea generation” (LLM narrative/tool requests) from “order auth” (deterministic filters on RR, vol-adjusted sizing, slippage guard). Build a replay harness to measure Sharpe/EV before risking capital. Keep session-level budget for LLM cost and reject churn.

### Suggested immediate steps
1) Implement runner-level integration test. 
2) Layer deterministic overlays (min RR, slippage check vs. decision price, anti-stack rules) before execution.  
3) Run paper sessions with telemetry review; optionally add a simple pause/HOLD control once tools are stable.  

### Findings from latest code review
- `strategy_runner.py`: `_passes_rr_filter` returns `True` when risk is zero/negative, allowing stop/target on the wrong side of entry; tighten to reject invalid RR inputs instead of letting them pass risk gating.
