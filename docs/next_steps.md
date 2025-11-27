## Project Review (LLM Day-Trading Bot)

### Snapshot
- Bot is LLM-driven with deterministic risk overlays (order caps, exposure limits, cooldowns, stacking guard) and tool-based market data.
- Context to LLM includes plan usage, open orders, recent decisions, and regime flags; prompts are now tool-first with truncation guards and cost/frequency limits.
- Runs locally from a laptop in live-only mode; execution routed via exchange adapter with light slippage checks.

### Key Risks & Gaps
- Execution realism is thin: slippage guard is a simple pct check; no latency model, liquidity impact, or maker/taker fill simulation; no PnL attribution per decision vs. drift.
- Limited resilience/ops: no circuit breaker on repeated exchange/tool failures; manual laptop ops risk losing state or missing fills if process dies.
- Risk controls stop at per-trade/per-symbol caps: no portfolio-level VaR, correlation/hedging awareness, or multi-symbol stacking logic beyond simple counts.
- Observability gaps for manual ops: logs exist, but no concise “what changed since last turn” dashboard/alerting for cost, exposure, error streaks, or missed tool requests.

### Recommended Next Steps (ordered)
1) Improve live market data coverage and resilience: stream depth/NBBO where possible, detect stale feeds, buffer short outages, and tag freshness/latency in the context so the LLM knows data quality.
2) Tighten LLM context management: summarize decision history and market deltas into a rolling digest; pin critical risk params; enforce turn-level truncation budgets; dedup repeated signals and enrich with per-symbol change logs.
3) Strengthen execution overlays: maker/taker preference toggle with fallback, per-symbol slippage band based on depth/vol, and a “no-trade” mode on repeated partial fills/rejections.
4) Add ops guardrails: circuit-breaker on consecutive exchange/tool errors, auto-pause on missing market data, and a heartbeat that surfaces current state (exposure, plans, open orders, LLM error streak, cost burn, data freshness) in one summary.
5) Broaden risk analytics: compute per-symbol and portfolio exposure with simple correlation/sector buckets; block same-direction adds when correlated names already at cap.
6) Manual control UX: add a lightweight CLI/Streamlit pane to pause/resume, flatten, or lower caps quickly; show LLM cost pace vs. session budget.
7) Data quality checks: enforce spread/depth thresholds per symbol before letting the LLM act; drop or down-weight symbols with stale prices or thin books.
