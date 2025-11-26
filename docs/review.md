## LLM Context Revamp (tool-based, JSON-first)
Goal: stop inlining thin snapshots into the prompt and instead let the LLM pull rich, chart-like data via explicit tool calls while Python keeps execution authority.

### Proposed architecture
- Two-phase loop: (1) planner turn where LLM can request data via allowed tools; (2) runner fetches/caches and replies with JSON payloads; optional (3) decision turn that consumes the fetched JSON and returns the trading action.
- Tool contract (example): LLM returns `{"tool_requests":[{"id":"req1","tool":"get_market_data","params":{"symbol":"BTC/USD","timeframes":["1m","5m","15m","1h","4h","1d"],"limit":500,"include_volume":true}},{"id":"req2","tool":"get_order_book","params":{"symbol":"BTC/USD","levels":50}}],"action":"HOLD"}`. Runner replies with `{"tool_response":"req1","data":{...}}` per request, then sends a follow-up message asking for the final decision.
- Safety rails: strict allowlisted tools/params, per-loop/timeframe limits, rate-limit and cache hits, clamp depth (e.g., max 200 levels) and history (e.g., 2k bars), reject unknown symbols, strip PII/secrets, and cap total JSON size before handing to LLM.

### Data catalogue to expose
- OHLCV history with volume for multiple windows a day trader expects: 1m/5m/15m/1h/4h/1d; include up to N bars (configurable) and attach computed summaries (e.g., last price, pct change, rolling vol/ATR, VWAP, RSI/MACD if already computed).
- Volume and spread/imbalance stats: rolling volume sums, average trade size, median spread, top-of-book imbalance percentile per timeframe.
- Full order book snapshot: bids/asks with price, size, cumulative notional, timestamp; configurable depth (e.g., 10/50/200). Optionally include derived metrics (spread, mid, imbalance, top-5/10 depth).
- Recent trades tape: last N trades with side/size/price/timestamp; useful for tape reading without cramming into the prompt.
- Metadata: symbol precision/lot size/min notional, fee tier flags, venue status (sandbox/live), and latency of the fetch.

### Response shaping
- Normalize all payloads to JSON arrays with typed fields: `{"ts": 1732610943000, "o": 86379.0, "h":..., "l":..., "c":..., "v":...}` for candles; order book as `{"bids":[[price, size],...],"asks":[...],"ts":...,"mid":...,"spread_bps":...}`.
- Include per-payload checksums and truncation notices so the LLM knows if data was clipped (e.g., `"truncated": true, "returned": 500, "requested": 2000`).
- Keep a small summary alongside the raw arrays (e.g., last price, pct change over each window, realized vol, VWAP, liquidity flags) so downstream logic can fall back if the LLM ignores raw arrays.

### Implementation steps
1) Define tool schemas (Pydantic dataclasses) for `get_market_data` (multi-timeframe OHLCV+volume), `get_order_book` (depth N), and `get_recent_trades` (last N). Enforce param bounds in Python. Log every request/response in `llm_traces.log`.
2) Add a `DataFetchCoordinator` in the runner that handles cache, batching, and rate limits; prefer ccxt OHLCV where available and build candles from ticks for IB gaps.
3) Update the LLM protocol: first turn accepts tool requests; second turn delivers JSON tool responses (not baked into the instruction prompt) plus a concise context header; third turn expects the action JSON. Keep a cap on total JSON bytes and drop to summaries if exceeded.
4) Extend telemetry to emit the full tool request/response envelopes so we can replay decisions and compare against fills.
5) Add config flags in `config.py` for max bars per timeframe, max depth, and max JSON bytes per round to fail-safe on huge requests.

### Execution checklist
- [x] Add Pydantic tool schemas for `get_market_data`, `get_order_book`, `get_recent_trades` with param bounds (timeframes, max bars, depth caps, symbol allowlist).
- [x] Create `DataFetchCoordinator` to fan out tool requests, hit caches, and apply rate limits; return normalized JSON + truncation metadata.
- [ ] Implement ccxt-backed OHLCV fetch with multi-timeframe support and volume; fall back to tick-to-candle builder for IB gaps.
- [x] Add recent trades/tape fetch and normalize fields (ts, side, price, size).
- [x] Add full-depth order book fetch with configurable depth (10/50/200) and derived metrics (mid, spread_bps, top depth, imbalance).
- [x] Build response-shaping helpers for candles/books/trades to enforce consistent field names and checksums/truncation flags.
- [x] Update LLM turn protocol: planner turn accepts tool requests; runner replies with tool responses plus brief context header; decision turn consumes JSON only.
- [x] Wire new protocol into `strategy_runner.py` and `strategy.py` (planner→data→decision flow), keeping execution authority in Python.
- [x] Add config knobs in `config.py` for max bars per timeframe, max depth, max JSON bytes, and per-tool rate limits.
- [x] Extend telemetry (`llm_traces.log`/DB) to persist full tool request/response envelopes and costs; redact secrets.
- [x] Add unit tests for tool schema validation, response shaping, and `DataFetchCoordinator` caching/truncation behavior.
- [x] Add integration test that simulates an LLM requesting multi-timeframe data + order book and receiving normalized JSON.
