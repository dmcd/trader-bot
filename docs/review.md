# Project Review – Trading Bot

## Findings (verbatim)
- Market context is thin and single-symbol: the loop only pulls one ticker (`BHP` or `BTC/USD`) and feeds essentially a last-tick snapshot plus an optional 1m OHLCV slice into the LLM (strategy_runner.py:726-759, strategy.py:174-206). No multi-window view (5m/1h/daily), no volatility/regime stats, no order book dynamics beyond top-of-book, and no cross-asset/macro context, so the LLM is flying mostly blind and likely to overfit noise.
- LLM traceability is weak: prompts/responses are not persisted, only the first 500 chars of the response text get written (strategy.py:520-535) and stdout is redirected to `console.log` without a dedicated LLM trace. This makes it hard to audit reasoning, compare requests to fills, or debug bad decisions.
- Trade syncing is hard-coded to `BTC/USD` and assumes the Gemini path even when running IB; the IB trader stubs `get_my_trades_async`/`get_trades_from_timestamp`, so reconciliation and PnL rebuilds don’t really work on IB (strategy_runner.py:555-565). You won’t catch missed fills or reconcile exposure for stocks.
- Risk/plan controls are basic: stops/targets are stored per plan but only checked against the latest single price (strategy_runner.py:827-831), no trailing logic, no slippage check beyond a TODO, and no notion of per-symbol volatility to size stops/targets. MAX_POSITIONS is reused as an open-order cap, which can block stacking legit scale-ins while still allowing multiple plans via `max_plans_per_symbol`.
- Profitability guardrails are light: no backtest/forward-test harness, no expected-value checks, and the LLM can still suggest trades even after repeated rejections (only a stored `last_rejection_reason` nudge). Using a general LLM for live decisions without a deterministic layer is high risk.
- Ways to add richer market context: persist multi-timeframe OHLCV (e.g., 1m/5m/1h/daily) and computed features (ATR, realized vol, VWAP, volume trend, rolling spread/imbalance) to the DB and feed those summaries into the prompt instead of just the last tick. Gemini has `fetch_ohlcv`; for IB you’ll need a lightweight candle builder from ticks.
- Add regime detection: simple volatility buckets, trend strength (ADX or slope of VWAP), and liquidity filters (median spread, top-of-book depth, order-book imbalance percentile) over lookbacks so the LLM knows when to sit out.
- Include cross-asset/context fields: BTC.D, ETH/BTC, SPY/QQQ, VIX, DXY if relevant; even static macro “risk-on/off” indicators can be cached hourly and injected into the prompt.
- Making it agentic (safely): introduce a tool layer for the LLM that can request specific data fetches (e.g., “get 1h BTC/USD OHLCV”, “get funding rate” where available) with strict allowlists, rate limits, and max latency; run these as async tasks and merge into context. Keep execution authority in Python; LLM only requests data, not orders.
- Add a planner-executor split: LLM proposes a plan with required data; runner gathers data; LLM revises with the enriched context. Cap the number of revisions per loop to avoid churn.
- More flexible decision/plan management: track plan objects in the DB with versioning and let the strategy re-score open plans each loop: adjust stops/targets (trail to breakeven, partial take-profits), cancel stale/low-conviction plans, or scale in/out when volatility drops/rises. Expose a “replan” action from the LLM (or a deterministic overlay) that updates an existing plan instead of firing new orders.
- Add deterministic overlays: volatility-normalized sizing, minimum RR (e.g., 1.5:1) before accepting a plan, and a slippage-aware check comparing current price vs decision price before execution.
- Logging & eval pipeline: add a dedicated `llm_traces.log` (or DB table) that stores full prompt, full response, parsed decision, market snapshot, and execution result per call. Keep token counts and costs for cost/perf analysis. Gate with a redaction layer for secrets.
- Emit a structured JSON line per loop with inputs (prices, indicators, exposure), LLM decision, risk verdict, and fill outcome. This lets you build offline evaluations and compare “what LLM wanted” vs “what filled”.
- Broader next steps for a profitable path: build a small backtest/simulation harness using your DB schema and ccxt historical data to sanity-check any strategy change before live. Separate deterministic signal layer from the LLM: let the LLM propose narratives/edge cases, but gate entries by tested rules (vol-adjusted breakout/pullback logic) and fixed stop/target logic. Expand IB support: implement trade history/ohlcv for IB so reconciliation and context work on stocks, and remove the `BTC/USD` hardcode.

## Task Checklist
- [x] Add multi-timeframe OHLCV storage (1m/5m/1h/4h/daily) and derived features (ATR, vol, VWAP, spread/imbalance) into DB and prompt.
- [ ] Implement regime detection (volatility buckets, trend strength, liquidity filters) and feed flags to the strategy.
- [ ] Add cross-asset/macro context fetchers (BTC.D/ETHBTC/SPY/QQQ/VIX/DXY or venue-appropriate) with caching.
- [ ] Build LLM tool layer for controlled data fetches; keep execution authority in Python.
- [ ] Introduce planner→executor loop (LLM plans, runner gathers data, LLM revises with cap on revisions).
- [ ] Rework plan management: plan versioning, re-score each loop, trailing/partial exits, and explicit “replan” action.
- [ ] Add deterministic overlays: volatility-normalized sizing, minimum RR filter, slippage check vs decision price.
- [x] Create dedicated LLM trace log/table capturing full prompt/response + parsed decision + market snapshot + execution result; redact secrets.
- [x] Emit structured JSON telemetry per loop (inputs, decision, risk verdict, fill outcome) for offline eval.
- [ ] Implement IB trade history and OHLCV support; remove hardcoded `BTC/USD` in sync paths.
- [ ] Build a backtest/sim harness using DB schema + ccxt historical data to validate strategy changes before live.
- [ ] Separate deterministic signal layer from LLM (LLM proposes, rules gate entries/exits).
