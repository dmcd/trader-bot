# Cost Tracker

`cost_tracker.py` tracks trading fees and LLM token spend.

## Fee handling
- Uses maker/taker fee constants from `config.py` to estimate fees before execution.
- Logs estimated fees per order_id when available (stored by the runner).
- Aggregates fees into session stats for dashboard display.

## LLM costs
- Computes input/output token costs per call; applies caps from `config.py`.
- Supports provider-specific rates (Gemini default, OpenAI optional) keyed by `LLM_PROVIDER`.
- Used by the runner to pause or reject actions if cost ceilings are reached.

## Tips
- Keep fee constants in sync with venue schedules; adjust env vars for live vs sandbox.
- If adding new LLM calls, run them through cost_tracker to preserve cost accounting.
