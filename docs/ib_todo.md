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

## Prior Work to Reuse (commit 8d6e4c8c6b2b0614dd909f36d8699e1c76e467af)
- Deleted `ib_trader.py` already sketched an `ib_insync` adapter: async connect/disconnect, account summary via `accountValues`, simple stock contract builder (`Stock(symbol, exchange, currency)`), market snapshot via `reqMktData`, market-order placement, positions/open orders retrieval, and clientId randomization to avoid collisions. We can cherry-pick these patterns as a starting point but need to harden them (timeouts, streaming reuse, richer status/fees, contract normalization, OHLCV/trade history).
- Config previously included `IB_HOST/IB_PORT/IB_CLIENT_ID` and cost settings (`IB_STOCK_FEE_PER_SHARE`, `IB_MIN_FEE`); reintroduce with better naming (`IB_GATEWAY_HOST`, etc.) and base-currency awareness.
- `CostTracker` had an IB per-share fee/min model; keep the structure but extend for FX/IBKR commissions and currency conversion.
- A lightweight FastMCP `server.py` exposed IB tools; if MCP is still desired, we can restore it behind a feature flag after the core adapter is stable.
- `requirements.txt` previously listed `ib_insync` and `mcp`; re-add `ib_insync` for the adapter, and consider gating MCP until needed.

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
   - [ ] Extend `CostTracker` to include IB commission model (per-share with min, FX spread buffer). Allow config overrides for taker/maker not used on IB.
   - [ ] Adjust `RiskManager` to accept base-currency conversions and recognize short exposure; ensure `MAX_ORDER_VALUE`, `MAX_TOTAL_EXPOSURE`, and `MIN_TRADE_SIZE` operate in `IB_BASE_CURRENCY`.
   - [ ] Add spread/liquidity heuristics tuned for equities/FX (e.g., min quote size in shares, tick-size aware price nudging). Revisit `MIN_TOP_OF_BOOK_NOTIONAL` defaults for AUD.
   - [ ] Update auto-flatten and plan-monitor exits to prefer marketable limit orders when IB disallows true market for certain contracts.

6) **Strategy Runner & Services Wiring**
   - [x] Switch `StrategyRunner` exchange selection to instantiate `IBTrader` when `ACTIVE_EXCHANGE=IB`; pass adapter exchange object (if any) into `DataFetchCoordinator` or stub a ccxt-like wrapper for tooling.
   - [x] Ensure `ResyncService` and `PlanMonitor` work with IB order ids and trade ids (may be alphanumeric). Verify `get_client_order_id` helper handles IB’s permId/clientId tuple.
   - [x] Add AUD baseline handling for sandbox-ignore positions logic (should be no airdrops, but keep compatibility).

7) **Database & Schema Considerations**
   - [ ] Decide on currency tagging: either store base currency at session level and rely on normalized symbols, or add optional `currency` columns to trades/positions if mixed-currency holdings are expected. Update writers/readers and migrations accordingly.
   - [ ] Validate `market_data` logging supports equity fields (bid/ask sizes are integer shares; volume is shares). No schema changes anticipated beyond optional currency fields.

8) **Dashboard & UX**
   - [ ] Surface IB account summary (cash balances, margin available) and commission estimates on the Streamlit dashboard.
   - [ ] Add venue badge and display base currency in PnL/equity cards. Clarify market hours status and recent connectivity/circuit state for IB.

9) **Testing & Validation**
   - [ ] Unit tests: fake IB client covering symbol resolution, market data normalization, order placement mapping, trade sync paths, and cost tracking. Add risk tests for AUD sizing and short exposure.
   - [ ] Integration/smoke: harness against IB paper account with recorded fixtures for market data, OHLCV, and order lifecycle; mark with `@pytest.mark.integration`.
   - [ ] Backfill docs on how to run IB tests (requires paper creds, Gateway running) and how to toggle to stub mode for CI.

10) **Rollout & Safety**
   - [ ] Start with shadow/observe mode on IB (execute_orders=False) to validate data flows, PnL math, and stop/target monitoring without live orders.
   - [ ] Add feature flag to block live IB trading until a manual checklist is completed (account id set, paper validated, market hours check).
   - [ ] Log telemetry fields for IB-specific diagnostics (contract id, exchange, commission source) to `telemetry.log` for early triage.

## Open Questions
- Do we need multi-currency reporting beyond AUD (e.g., USD-denominated FX or US equities) in the first iteration?
- Should we support IBKR bracket orders for stop/target attachment instead of separate plan monitoring?
- Are ASX tick-size/lot-size rules strict enough to require a symbol-specific metadata cache, or can we rely on IB’s `minTick`/`multiplier` at runtime?
