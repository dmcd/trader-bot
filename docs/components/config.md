# Configuration

`config.py` centralizes tunables, primarily from environment variables.

## Key areas
- Trading mode/keys: `TRADING_MODE`, `ACTIVE_EXCHANGE`, sandbox vs live API keys.
- Risk limits: daily loss caps, max order value, exposure caps, spacing, slippage, spread/notional guards.
- Cadence: loop interval, trade spacing, priority move thresholds.
- LLM limits: cost caps, call spacing, max consecutive errors.
- Plan settings: max per symbol, max age, trailing-to-breakeven pct.
- Tool limits: max bars/trades/depth, allowed symbols/timeframes.
- Order routing: `CLIENT_ORDER_PREFIX` used for clientOrderId tagging and trade sync filtering.

## Tips
- Set `.env` values for sandbox vs live before starting the runner or dashboard.
- Changes require restarting the bot to take effect.
