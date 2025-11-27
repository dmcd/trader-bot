## Prompt Bloat & Context Handling

### Findings
- Prompt bloat risk: `trading_context.py:15-178` emits a large multiline narrative (session stats, positions, open orders, last 5 trades, market trend) that is injected wholesale into every planner/decision prompt via `context_block` in `strategy.py:770-794`; none of it is size-limited or summarized for the LLM’s needs, so prompts grow with session age.
- Tool payloads are not actually clamped: `llm_tools.py:317-330` only adds `truncated=True` without shrinking JSON, so large order books/candle sets can overflow `TOOL_MAX_JSON_BYTES` and bloat the prompt when appended in `strategy.py:883-887`.
- Past analysis/decisions are not surfaced: trade plans (`database.py:768-820`), prior prompts/responses (`llm_traces`), and execution outcomes are persisted but never fed back into the next prompt, so the LLM has no memory of its own reasoning beyond what’s re-summarized each call.

### How the LLM sees context today
- Every decision call in `strategy.py:654-930` builds `base_prompt` from `llm_prompt_template.txt`, injecting current market snapshot, risk caps, regime flags, cooldown/headroom/open-order summary (`prompt_context`), and the verbose session summary from `TradingContext.get_context_summary`.
- If tools are available, a planner prompt (same market/risk/context summary, no cooldown rules) runs first; tool responses are appended verbatim as JSON to the decision prompt.
- Minimal historical memory: `TradingContext` derives win rate, PnL, positions, open orders, and last 5 trades from the DB; `last_rejection_reason` is included once; otherwise each call is stateless—prior plans/decisions are only logged to DB/telemetry, not re-injected.

### Checklist to reduce prompt bloat and strengthen recall
- [x] Trim and structure context: convert `get_context_summary` to a compact, capped JSON block with only essential fields (win rate/PnL, positions summary, open orders summary, last N trades), and prefer deltas since last turn.
- [x] Enforce deterministic size limits on tool data: make `clamp_payload_size` actually prune payloads (drop deep book levels/old candles/verbose fields) and apply it before logging/embedding tool responses; set per-tool byte caps and brief summaries.
- [x] Surface prior intent/outcomes: inject a small “memory” block containing open trade plans (id/side/size/stop/target/reason), last N decisions with execution results, and recent risk rejections—kept under a strict byte budget.
- [x] Separate static instructions from live state: keep playbook/rules in template/system prompt; ensure per-turn prompt only includes live metrics, compact memory, and tool responses.
- [x] Add guardrails that indirectly cut bloat: symbol/param allowlists and per-tool rate limits so the LLM cannot request or see extraneous data.
