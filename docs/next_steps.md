## Trading Bot Next Steps

### Priorities
- Prove reliability before sizing (stabilize data/tooling, harden risk checks, validate LLM/tool loop with telemetry).
- Keep Python in charge of risk/sizing/execution; tools supply data, prompts stay lean.

### Remaining work
- [x] Runner integration polish: decision prompt should state tool_responses supersede inline snapshots; strip any legacy multi-TF summaries when tools are used.
- [x] Risk overlays: enforce slippage/vol scaling and position stacking rules at execution; align with regime flags.
- [x] Integration test (runner loop): simulate planner→tool→decision within the runner, asserting clamps/truncation flags propagate.
- [ ] LLM cost guard: cap planner/decision frequency and budget per session; fail-safe to HOLD on repeated errors.

### Open finding
- `strategy_runner.py`: `_passes_rr_filter` returns `True` when risk is zero/negative, allowing stop/target on the wrong side of entry; tighten to reject invalid RR inputs instead of letting them pass risk gating.
