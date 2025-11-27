# Dashboard

`dashboard.py` is a Streamlit UI for monitoring and control.

## Panels
- Session Performance: PnL (net/gross/realized/unrealized), fees, LLM costs, trade spacing, exposure, loop interval.
- Active Positions: open holdings with unrealized PnL.
- Trade History: today’s trades, localized to user timezone (`LOCAL_TIMEZONE`/`TZ` env fallback; defaults to system local → UTC). Shows Date/Time, symbol, action, price, qty, PnL, fee, liquidity, reason.
- Open Orders: current exchange open orders snapshot.
- Trade Plans: open plans with stop/target and opened time.
- LLM Decisions: recent counts/errors/clamped stops.
- Logs: tail of `bot.log`.

## Controls
- Start/Stop bot, Close All Positions.

## Notes
- Trade history is session-scoped (today). Local time conversion happens after reading UTC timestamps from DB.
- Plans and orders are displayed separately; an order cancel does not auto-close a plan.
