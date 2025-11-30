# Interactive Brokers (AU) Support Plan

## Context / Constraints
- Current stack is Gemini-only via `ccxt` with a single `BaseTrader` implementation (`GeminiTrader`). Exchange selection is gated by `ACTIVE_EXCHANGE`, risk sizing is USD-centric, and LLM tools pull market data straight from the ccxt exchange.
- `StrategyRunner` owns dependency wiring (bot, tool coordinator, services) and assumes the exchange exposes ccxt-style methods for market data, trades, orders, and OHLCV.
- `DataFetchCoordinator`, `PlanMonitor`, `ResyncService`, and `PortfolioTracker` expect ccxt-shaped payloads: symbols like `BTC/USD`, fields such as `price/bid/ask`, `amount/filled/remaining`, `fee.cost`, `liquidity`, and clientOrderId on orders/trades.
- Cost tracking and config only include Gemini fees; session accounting assumes prices and fees are in the same quote currency (currently USD).

## Assumptions for IB (Australia)
- Use Interactive Brokers TWS/Gateway with `ib_insync` as the client library (async-friendly). Paper trading uses paper account endpoints, live uses production.
- Primary instruments: ASX equities/ETFs and optionally IBKR-listed FX (AUD base). No crypto via IBKR AU.
- Symbols will be normalized internally to `SYMBOL/CCY` (e.g., `BHP/AUD`, `AUD/USD` for FX) to keep risk/TA coherent; IB contracts require exchange and secType metadata.
- Market data cadence similar to current loop (5m default), with historical bars from IBKR’s `reqHistoricalData` (1m–1d) and top-of-book via `reqMktData`.
- Order types: limit as default; allow market for flattening and plan exits where permitted. Post-only is not available; maker/taker fees are not meaningful—use commission estimates instead.


## Work Plan
1) **Dependencies & Config** ✅
   - [x] Add `ib_insync` to dependencies and document install prerequisites (TWS/Gateway running, Java requirement, sockets open).
   - [x] Extend `config.py` and `.env` samples with IB fields: `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `IB_ACCOUNT_ID`, `IB_PAPER=true/false`, `IB_BASE_CURRENCY` (default AUD), `IB_EXCHANGE` defaulting to `SMART/ASX`, per-venue commission overrides, and allowed instrument types.
   - [x] Update `README.md` and `AGENTS.md` venue section to list IB, setup steps, and required environment variables.

2) **Symbol & Contract Normalization**
   - [x] Define a small contract resolver utility that maps `SYMBOL/CCY` plus optional exchange/sectype into an `ib_insync` `Contract` (e.g., `Stock("BHP", "ASX", "AUD")`, `Forex("AUDUSD")`).
   - [x] Introduce a canonical symbol formatter shared by IB adapter, TA capture, and LLM context to keep DB symbols stable (avoid IBKR local symbols leaking through).
   - [x] Add validation/translation for `ALLOWED_SYMBOLS` so mis-specified entries fail fast with actionable errors.

3) **IBTrader Adapter (implements BaseTrader)**
   - [x] Connection lifecycle: manage `IB().connect()` with paper/live routing, heartbeat/ping, and graceful disconnects.
   - [x] Account/equity: fetch `accountValues` in base currency; convert non-base positions using FX quotes (cache quotes to minimize rate calls). Surface `get_equity_async` in AUD.
   - [x] Market data: provide `get_market_data_async` using `reqMktData` top-of-book; populate `price/bid/ask/bid_size/ask_size/volume/spread_pct/ob_imbalance` fields.
   - [x] Orders: implement `place_order_async` with limit first, optional market for flatten, map order status/avg fill/remaining/fee (commission) and generate clientOrderId via `CLIENT_ORDER_PREFIX`. Handle rejected/filled/partial states and timeouts.
   - [x] Positions: map IB portfolio to `symbol/quantity/avg_price/current_price/timestamp`; treat short positions as negative qty.
   - [x] Open orders: list working orders filtered by our clientOrderId prefix; normalize to ccxt-like shape.
   - [x] Trades/history: implement `get_my_trades_async` and `get_trades_from_timestamp` via `executions()`/`fills()` with pagination guards; include commission and liquidity (fallback `unknown`).
   - [x] OHLCV: add `fetch_ohlcv` using `reqHistoricalData`; honor timeframe/limit from TA and tools, map to `[ts,o,h,l,c,v]`, clamp unsupported timeframes, and handle IB pacing/ordering quirks.
   - [x] Add reconnection/backoff hooks compatible with `HealthCircuitManager.maybe_reconnect`.

4) **Data Fetch & Tooling Integration**
   - [x] Wrap IB market data/historical endpoints to mimic ccxt returns so `DataFetchCoordinator` continues to work (candles as `[ts, o, h, l, c, v]`, order book depth as bids/asks arrays, trades with `price/amount/timestamp`).
   - [x] Ensure rate limits and pacing respect IBKR restrictions (max 60 hist requests/10min; streaming top-of-book reuse where possible). Add guardrails to tool coordinator for IB (e.g., clamp timeframe combinations IB supports).
   - [x] Update `TradingContext` and LLM prompt scaffolding to describe IB venue, instrument universe (ASX/FX), and market hours so the LLM doesn’t request unsupported products.

5) **Risk, Sizing, and Fees**
   - [x] Extend `CostTracker` to include IB commission model (per-share with min, FX spread buffer). Allow config overrides for taker/maker not used on IB.
   - [x] Adjust `RiskManager` to accept base-currency conversions and recognize short exposure; ensure `MAX_ORDER_VALUE`, `MAX_TOTAL_EXPOSURE`, and `MIN_TRADE_SIZE` operate in `IB_BASE_CURRENCY`.
     - [x] Wire a currency converter using `IBTrader` market data/quotes so risk checks compare values in base currency.
     - [x] Normalize exposure calculations to treat short positions as negative and cap absolute exposure.
     - [x] Add tests covering AUD base currency sizing for long/short and per-order value validation.
   - [x] Add spread/liquidity heuristics tuned for equities/FX (e.g., min quote size in shares, tick-size aware price nudging). Revisit `MIN_TOP_OF_BOOK_NOTIONAL` defaults for AUD.
     - [x] Choose defaults for ASX equities vs FX pairs (spread caps, min depth, tick size guard).
     - [x] Wire heuristics into order pre-checks and market data sanity checks.
     - [x] Add config entries and docstrings for new knobs; include AUD-specific defaults.
     - [x] Add unit tests covering widening spreads and illiquid books for both equities and FX.
   - [x] Update auto-flatten and plan-monitor exits to prefer marketable limit orders when IB disallows true market for certain contracts.
     - [x] Detect contracts where market orders are disallowed and derive a slippage bump for marketable limits.
     - [x] Update `PlanMonitor` exit path and any auto-flatten helpers to use the marketable limit flow.
     - [x] Add regression tests for flatten/exits choosing limit with slippage bump on IB.

6) **Strategy Runner & Services Wiring**
   - [x] Switch `StrategyRunner` exchange selection to instantiate `IBTrader` when `ACTIVE_EXCHANGE=IB`; pass adapter exchange object (if any) into `DataFetchCoordinator` or stub a ccxt-like wrapper for tooling.
   - [x] Ensure `ResyncService` and `PlanMonitor` work with IB order ids and trade ids (may be alphanumeric). Verify `get_client_order_id` helper handles IB’s permId/clientId tuple.
   - [x] Add AUD baseline handling for sandbox-ignore positions logic (should be no airdrops, but keep compatibility).

7) **Database & Schema Considerations**
   - [x] Decide on currency tagging: store base currency at session level (uppercase) and rely on normalized symbols; optional per-row currency fields not required yet. Update writers/readers and migrations accordingly.
     - [x] Evaluate current schema usages and choose approach for base currency handling.
     - [x] Implement chosen schema/migration path and update ORM/helpers.
     - [x] Add tests validating reads/writes with AUD base and (if applicable) mixed currencies.
   - [x] Validate `market_data` logging supports equity fields (bid/ask sizes are integer shares; volume is shares). No schema changes anticipated beyond optional currency fields.
     - [x] Ensure serializers preserve integer sizes for bids/asks/volume.
     - [x] Add coverage for equity market data rows in logging pipeline.
     - [x] Confirm dashboards/consumers handle the shape without schema drift.

8) **Dashboard & UX**
<<<<<<< HEAD
   - [x] Surface IB account summary (cash balances, margin available) and commission estimates on the Streamlit dashboard.
     - [x] Add a service/DTO for account summary + commission estimates decoupled from `ib_insync` objects.
     - [x] Expose account summary from IB adapter to the dashboard layer.
     - [x] Render balances/margin and per-order commission estimates with AUD labels.
     - [x] Add tests for dashboard data assembly (can be stubbed service).
   - [x] Add venue badge and display base currency in PnL/equity cards. Clarify market hours status and recent connectivity/circuit state for IB.
     - [x] Add dashboard-friendly status payload (venue, base currency, market hours, last circuit state).
     - [x] Show venue indicator and base currency in equity/PnL widgets.
     - [x] Add market hours/open status and last connectivity/circuit state.
     - [x] Add UI tests or snapshot coverage for the new elements.
=======
   - [ ] Surface IB account summary (cash balances, margin available) and commission estimates on the Streamlit dashboard.
     - [ ] Add a service/DTO for account summary + commission estimates decoupled from `ib_insync` objects.
     - [ ] Expose account summary from IB adapter to the dashboard layer.
     - [ ] Render balances/margin and per-order commission estimates with AUD labels.
     - [ ] Add tests for dashboard data assembly (can be stubbed service).
   - [ ] Add venue badge and display base currency in PnL/equity cards. Clarify market hours status and recent connectivity/circuit state for IB.
     - [ ] Add dashboard-friendly status payload (venue, base currency, market hours, last circuit state).
     - [ ] Show venue indicator and base currency in equity/PnL widgets.
     - [ ] Add market hours/open status and last connectivity/circuit state.
     - [ ] Add UI tests or snapshot coverage for the new elements.
>>>>>>> parent of 72112dd (Add IB account summary and commission estimates to dashboard)

9) **Testing & Validation**
   - [x] Unit tests: fake IB client covering symbol resolution, market data normalization, order placement mapping, trade sync paths, and cost tracking. Add risk tests for AUD sizing and short exposure.
     - [x] Build a reusable fake IB client covering contracts, orders, trades, and historical data.
     - [x] Provide fixture factory/helpers to keep scenarios readable and re-usable across tests.
     - [x] Add test toggles for IB vs Gemini paths where logic diverges.
     - [x] Add normalization tests for market data, trades, and OHLCV.
     - [x] Add risk and cost tracker tests for AUD and short handling.
   - [x] Integration/smoke: harness against IB paper account with recorded fixtures for market data, OHLCV, and order lifecycle; mark with `@pytest.mark.integration`.
     - [x] Create fixture capture scripts and store sanitized fixtures for CI-friendly playback.
     - [x] Add playback/stub mode for integration tests so CI can run without live IB Gateway.
     - [x] Wire CI marker/flag to skip live IB tests unless credentials and Gateway are available.
     - [x] Add smoke tests gated by env flag and mark as integration.
     - [x] Document how to record/update fixtures when IB API changes.
   - [x] Backfill docs on how to run IB tests (requires paper creds, Gateway running) and how to toggle to stub mode for CI.
     - [x] Document setup for running unit vs integration IB tests locally.
     - [x] Add `.env.example` entries for IB test creds and fixture playback flags.
     - [x] Add instructions for paper account credentials and TWS/Gateway prerequisites.
     - [x] Explain CI stub mode and how to enable fixture playback.

10) **Rollout & Safety**
   - [x] Log telemetry fields for IB-specific diagnostics (contract id, exchange, commission source) to `telemetry.log` for early triage.
     - [x] Extend telemetry payloads to include IB contract id/exchange/commission source.
     - [x] Add tests/fixtures to assert telemetry entries for IB orders/trades.
     - [x] Document telemetry fields and sampling strategy for IB traffic.
