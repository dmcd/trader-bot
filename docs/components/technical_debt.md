# Technical Review Findings

Verbatim observations from the code review.

## Findings
- `strategy_runner.py`: Session stats rebuild hardcodes `symbol = 'BTC/USD'` and only fetches that pair from the exchange. Any other symbols traded that day won’t be counted in holdings/PnL on restart, so exposure and loss guards can be wrong after a restart or symbol change. Expand to all active symbols or query exchange trades without a single-symbol assumption.
- `strategy_runner.py`: Plan monitor uses a single `price_now` for every open plan; it’s explicitly “single-symbol.” Stops/targets and trailing for other symbols never fire, so those plans can sit forever. Needs per-plan pricing (latest tick per symbol) before triggering closes.
- `risk_manager.py`: Exposure sums pending sells as positive (`pending_sell_exposure`) and adds it to `get_total_exposure`, then uses that inflated number when checking sells. A flattening sell can be blocked because pending sells are treated as extra exposure instead of reducing it. Consider netting pending sells against positions or separating gross/offsetting paths so risk checks don’t reject risk-reducing orders.
- `strategy_runner.py`: Trade sync fetches only the last 20 trades and doesn’t paginate by `since`; high-activity sessions or downtime will drop fills and leave DB/session stats short. Also only called for one symbol path. Add `since` pagination and per-symbol coverage to avoid gaps.
- General linkage/cleanup: Plans are not tied to order lifecycle; cancels/expirations leave plans open indefinitely, and there’s no `plan_id` stored alongside order_id for reconciliation. If you want parity between plans and orders, wire `plan_id` through placement/snapshots and close plans when their orders are canceled/fully flattened. (Doc’d, but remains tech debt.)
- Minor: `MIN_TRADE_INTERVAL_SECONDS` is imported in `strategy_runner.py` but unused; dead import suggests spacing guard only lives in `strategy.py`. Remove or implement consistently.

## Checklist
- [ ] Remove single-symbol assumption in session stats rebuild; fetch trades across all active symbols and paginate by time.
- [ ] Make plan monitoring per-symbol with fresh prices; ensure stops/targets/age apply to all plans.
- [ ] Correct exposure math for pending sells so risk checks don’t block risk-reducing sells.
- [ ] Add paginated trade sync (by `since`) across symbols to avoid missed fills.
- [ ] Link plans to orders (store `plan_id` with order_id/clientOrderId) and close plans on cancel/fill/flatten.
- [ ] Clean up unused `MIN_TRADE_INTERVAL_SECONDS` import or apply a consistent spacing guard in the runner.
