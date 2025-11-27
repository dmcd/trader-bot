## Plan Handling – Findings & Next Steps

### What I saw in logs
- Repeated SELL decisions referencing “Executing Plan 3” were blocked with `plan cap reached for BTC/USD (2/2)`. No trades executed.
- This means the LLM is aware of the plan (mentions it) but is issuing a new BUY/SELL intent instead of managing existing plans; the runner enforces the cap and blocks it.

### Current capabilities
- The LLM already receives plan details (ids/sides/size/stop/target/reason) via the memory block.
- The runner supports plan management actions: `UPDATE_PLAN`, `PARTIAL_CLOSE`, `CLOSE_POSITION`, `CANCEL`.
- Plan cap is enforced in the runner (per-symbol limit). There is no “replace plan” shortcut beyond UPDATE_PLAN+CANCEL.

### Next steps
1) **Prompt rules**: Add an explicit rule: “If plan cap reached for a symbol, do not place new BUY/SELL. Use UPDATE_PLAN (with plan_id/stop/target/size_factor) or CANCEL/PARTIAL_CLOSE existing plans.” **(done)**
2) **Expose plan caps clearly**: In the prompt context, include plan cap per symbol (used/cap) and highlight when at cap; already partially present, but add a rule line so the LLM must obey. **(done)**
3) **Runner guardrails**: When a BUY/SELL is blocked due to plan cap, feed that rejection back into `last_rejection_reason` so the next prompt explicitly warns the LLM. **(done)**
4) **Replace flow (optional)**: Add a helper to “replace plan” by auto-cancelling the oldest plan when cap is hit and a new plan is requested. Gate behind a config flag; log aggressively. **(done; AUTO_REPLACE_PLAN_ON_CAP)**
5) **Telemetry check**: Add a dashboard/summary of plan counts per symbol and last LLM action type (UPDATE_PLAN vs BUY) to see if guidance is working.
6) **Test**: Add a unit/integration test ensuring the LLM prompt contains plan cap info and that cap rejections populate `last_rejection_reason`. **(prompt/rule present; rejection feedback wired)**
