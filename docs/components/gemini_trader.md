# Gemini Trader

`gemini_trader.py` is the ccxt-based adapter for Gemini (sandbox and live).

## Connection
- Uses sandbox keys in PAPER mode, live keys otherwise (see `config.py`).
- Fills missing sandbox precision by loading live market metadata.

## Orders
- Only limit orders; prefers maker (`postOnly`) then retries taker on rejection.
- Generates `clientOrderId` with `CLIENT_ORDER_PREFIX` + timestamp + suffix; returned in `client_order_id` for tracking.
- Snapshots open orders via `fetch_open_orders` (no plan linkage).

## Trades and data
- `fetch_my_trades` used for sync; runner filters to our `clientOrderId` prefix to avoid sandbox noise.
- Provides tickers, balances, positions valuation, and OHLCV helpers.

## Tips
- Set `CLIENT_ORDER_PREFIX` in `.env` to isolate our fills in sandbox sync.
- If sandbox lacks precision, defaults to sensible tick sizes are applied; adjust in code if venue adds new pairs.
