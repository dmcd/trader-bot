# Test Coverage Plan

## Coverage snapshot (python -m pytest)
- Total coverage: 61% (1878 stmts missed, 339 partial branches).
- Biggest gaps: `trader_bot/dashboard.py` (0%), `trader_bot/gemini_trader.py` (19%), `trader_bot/strategy_runner.py` (51%), `trader_bot/trading_context.py` (51%), `trader_bot/strategy.py` (69%), `trader_bot/database.py` (72%), `trader_bot/data_fetch_coordinator.py` (73%).

## Priority 1: Core loop, risk, and context
- [x] `trader_bot/strategy_runner.py` (target ≥70%): add async harness with stub bot/DB to cover kill-switch propagation, command stop vs risk stop, and equity fetch failure short-circuit; assert telemetry/log hooks are invoked. Cover `_get_active_symbols/_get_sync_symbols` dedupe and fallback to BTC, market-data pause/health gating, and `_record_operational_metrics` budget/fee ratio branches. Exercise cancel/update/partial-close/close-position action paths with mocked `TradeActionHandler` to confirm routing and open-order refresh.
- [x] `trader_bot/strategy.py` (target ≥80%): unit tests for `_priority_signal` spacing, `_build_timeframe_summary` output, and `_compute_regime_flags` (trend/volatility). Cover prompt/template loading cache, `_enforce_prompt_budget` truncation, and `_parse_tool_requests` error handling. Add `_get_llm_decision` tests for tool request flow (market/order-book/trades) including can_trade=False and plan-count headroom, plus `on_trade_rejected/on_trade_executed` cooldown resets. Validate `_clamp_quantity` with headroom/order-value buffer and min trade size.
- [x] `trader_bot/trading_context.py` (target ≥80%): tests for `_filter_our_orders` prefix match, `_net_quantity_for_session` with long/short baselines, and `set_position_baseline` merge behavior. Cover `get_context_summary` win-rate/trend calculations and trimming of positions/open orders, `get_memory_snapshot` size trimming and JSON failures, and `get_recent_performance` profit/win-rate defaults when trades are empty vs mixed P/L.
- [x] `trader_bot/risk_manager.py` (target ≥90%): cover `apply_order_value_buffer` edge cases (under/over cap, zero price), correlation bucket rejection in `check_trade_allowed`, and baseline exposure handling in `_net_quantity_for_exposure`/`get_total_exposure` including pending buy/sell offsets.
- [x] `trader_bot/services/trade_action_handler.py` + `plan_monitor.py` + `resync_service.py` (targets ≥85%): add tests for slippage guard and RR filters in action handler, plan age/trailing/breakeven handling in plan monitor, and resync trade/position reconciliation with error callbacks; ensure telemetry/actions_logger calls occur on success and failure paths.

## Priority 2: Data/adapter surfaces
- [x] `trader_bot/data_fetch_coordinator.py` (target ≥85%): rate-limit window reset and dedupe cache reuse (meta.deduped), symbol allowlist rejection, cache TTL expiry, and fallback trades→candles when OHLCV fails. Cover clamp_payload_size invocation and error/success callbacks.
- [x] `trader_bot/gemini_trader.py` (target ≥70%): stub ccxt exchange to test precision population, post-only price computation, maker/taker preference flag, and rejection handling in `place_order_async`. Cover mark-price aggregation in `_fetch_mark_prices`, `get_trades_from_timestamp` pagination window, and `fetch_ohlcv` passthrough with unsupported exchange methods raising.
- [x] `trader_bot/database.py` (target ≥80%): tests for health state setters/getters, command retention pruning, trade plan CRUD (open/update/close) with versioning, equity snapshot logging/pruning, and graceful handling of missing rows (e.g., get_open_orders when empty). Include concurrency-safe path for `log_llm_trace` JSON serialization failures.
- [x] `trader_bot/services/market_data_service.py` + `portfolio_tracker.py` (targets ≥85%): ensure timeframe parsing, OHLCV pruning limits, and capture spacing guards; portfolio tracker rebuild from trades, fee netting, and cache invalidation on replace_positions.

## Priority 3: Calculations, config, utilities
- [x] `trader_bot/technical_analysis.py` (target ≥85%): cover RSI/MACD/Bollinger edge cases (insufficient bars, NaN handling) and signal summary branches. 
- [x] `trader_bot/config.py` (target ≥95%): `_parse_maker_overrides` truthy/falsey parsing, `_parse_correlation_buckets` empty tokens, and `CLIENT_ORDER_PREFIX` default composition when BOT_VERSION missing.
- [x] `trader_bot/cost_tracker.py` (target ≥98%): `calculate_llm_burn` ISO parsing with "Z" suffix vs bad strings and min-window normalization.
- [ ] `trader_bot/utils.py` + `trader_bot/trader.py` (targets ≥90%): `get_client_order_id` fallbacks, trader session lifecycle logging, and error branch when strategy missing.
- [ ] `trader_bot/logger_config.py` + `metrics_validator.py` (targets ≥95%): ensure logger setup respects env vars/handlers and metrics validator rejects missing/negative fields.

## Priority 4: Dashboard/UI
- [ ] `trader_bot/dashboard.py`: add lightweight Streamlit smoke tests using `pytest` + `streamlit.testing` or function-level unit tests for data shaping (metrics tables, trend calculations) to lift coverage off 0% without spinning up full app.
