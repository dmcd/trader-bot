# LLM Strategy

`strategy.py` houses `LLMStrategy`, which converts market state into structured actions.

## Inputs
- Trading context from `trading_context.py` (positions, prices, indicators, caps, cooldowns).
- System prompt + tools for Gemini 2.5 Flash.

## Outputs
- Actions: `BUY`/`SELL` entries (may include `stop_price`/`target_price`), `HOLD`, `CLOSE_POSITION`, `PARTIAL_CLOSE` (with `plan_id` and fraction), `UPDATE_PLAN`, `PAUSE_TRADING`.
- Reasons and notes logged for telemetry and dashboard.

## Prompts and validation
- Schema enforced via Pydantic; malformed responses increment schema error metrics.
- Ratelimit and cost constraints enforced upstream by `strategy_runner`.

## Tips
- To ensure plans are created, make the LLM always supply stop and/or target with entries.
- Keep prompts short; high token counts hit cost guards in `cost_tracker`.
