# Risk Manager

`risk_manager.py` enforces guardrails before orders are sent.

## Checks
- Daily loss caps (absolute and % of equity).
- Max order value and total exposure caps (includes pending buy orders via runner logic).
- Trade spacing (`MIN_TRADE_INTERVAL_SECONDS`) and stacking prevention per symbol.
- Spread width and top-of-book notional requirements.
- Slippage guard: runner compares decision vs refreshed price and skips if drift exceeds `MAX_SLIPPAGE_PCT`.

## Usage
- `check_trade_allowed(symbol, side, size, price)` returns `allowed` and a reason; the runner blocks trades when false.
- `update_pending_orders` lets the risk manager account for open orders when computing exposure headroom.

## Tips
- Tune caps via `config.py` env vars.
- If you add new risk signals (e.g., volatility bands), keep returns structured so the runner can emit telemetry.
