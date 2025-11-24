# Code Review Findings & Action Items

## Findings (verbatim)
- strategy_runner.py:832 — HOLD decisions fall through to `_apply_order_value_buffer(quantity, price)` with `price` undefined; a HOLD from the LLM will crash the loop. Add an early continue or default `price = 0` before the guardrails.
- strategy.py:292‑317 — Stop/target clamping uses a long-only band even when `action == 'SELL'`. A short trade will have its stop forced below the entry price, removing loss protection and letting the bot ride losses. Clamp directions separately (short stops above, targets below).
- risk_manager.py:19‑68 and strategy_runner.py:302‑673 — Daily loss guard resets on restart. `start_of_day_equity` is set to the current equity when None; if you restart after losing, the drawdown baseline moves down and the bot can keep trading past the original limit. Persist start-of-day equity (e.g., in DB/session) and reuse it on restart.
- risk_manager.py:79‑133 — MAX_POSITIONS is documented but never enforced. You warn on open-order counts in the runner but the core risk check always returns allowed after exposure checks. Add a hard cap per symbol/overall in `check_trade_allowed` to keep the LLM from stacking legs.
- Architecture risk — Core decisioning is delegated to a live LLM with minimal deterministic guardrails and no backtesting/simulation harness. There’s no offline validation of prompts/sizing logic against historical data or shadow-mode execution to verify PnL/fee impact before trading real money.
- Volume/order book signal gap — For a day-trader-style bot, add volume/OB features before new behaviors: rolling volume delta and VWAP drift; volume profile for session control; tape-based momentum (uptick/downtick imbalance); order-book imbalance at best levels and within top N; spread/impact estimates to choose maker vs taker. Persist these in `market_data` (volume column exists) and feed them into the prompt and a deterministic prefilter (e.g., skip entries when spread > X or depth on the opposite side thins out).

## Tasks (tick as we go)
- [x] Fix HOLD path crash in `strategy_runner.py` by early-continuing or defaulting price before buffer/guardrails.
- [x] Clamp stops/targets correctly for shorts in `strategy.py` (stop above entry, target below).
- [x] Persist start-of-day equity in the DB/session and reuse after restart so daily loss limits hold.
- [x] Enforce `MAX_POSITIONS`/per-symbol caps inside `RiskManager.check_trade_allowed`.
- [ ] Add volume + order-book ingestion, persistence, and pretrade filters (spread/imbalance/impact), surface in LLM prompt.
- [ ] Add shadow/backtest harness to validate prompts/sizing/fees offline before live trading.
