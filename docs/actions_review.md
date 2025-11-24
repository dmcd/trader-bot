# LLM Action Space Review

## Findings (verbatim)
- Current actions are thin (BUY/SELL/HOLD/CANCEL with optional stop/target). This forces “one-shot” orders and limits nuanced control; the LLM can’t adjust plans or scale without new orders.
- Missing plan-aware updates: no ability to `UPDATE_PLAN`/`REPLAN` to trail stops, tighten targets, reduce size, or revise logic without reissuing an order.
- No scaling controls: can’t `SCALE_IN`/`SCALE_OUT` or `PARTIAL_CLOSE` against an existing plan/position; every adjustment is a fresh BUY/SELL.
- No explicit flatten/risk-off control: cannot issue `CLOSE_POSITION` on a symbol when regime/liquidity changes.
- No cooldown/throttle intents: LLM cannot request a temporary `PAUSE_TRADING` window when conditions are bad.
- Order hygiene is coarse: only `CANCEL` (order id) exists; no `CANCEL_PLAN` to withdraw a stale plan or cancel a set of related orders.
- No data/tool intents: the LLM can’t `REQUEST_DATA` (specific lookbacks/indicators) or set alerts; it relies solely on what we feed it.
- Schema is minimal, so the runner can’t distinguish intent granularity; validation covers only the current four actions.
- Runner-side guardrails aren’t wired for richer intents (e.g., validated plan updates, scale adjustments, partial exits).
- Tests cover existing actions only; new intents would need schema validation and runner behavior coverage to stay safe.

## Task Checklist
- [x] Expand action schema/prompt to include `UPDATE_PLAN` (stop/target/size_factor) and `PARTIAL_CLOSE`; keep strict validation.
- [x] Implement runner handling for `UPDATE_PLAN` and `PARTIAL_CLOSE`, with plan versioning and risk/rr/slippage guards.
- [x] Add tests for new actions (schema validation + runner behavior paths).
- [ ] Add optional `CLOSE_POSITION` and `PAUSE_TRADING` intents with runner enforcement and tests.
- [ ] Add `REQUEST_DATA`/`SET_ALERT` intents (tool/control only, no direct execution) and tests.
