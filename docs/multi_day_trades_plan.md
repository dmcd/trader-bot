# Multi-day Positional Trades Plan

Notes on the current session model:
- A new `sessions` row is created on each runner start with `date` (day-only) and `bot_version`; all state (trades, market data, positions, open orders, plans, indicators, OHLCV, equity snapshots) is keyed by `session_id`.
- Risk is anchored to `start_of_day_equity` stored in `risk_state`/`session_stats_cache`, assuming a single-day session; daily loss checks reset only when a new session baseline is written.
- Resync, portfolio tracking, and plan monitoring all scope to the active `session_id`, so restarting or rolling to a new calendar day drops visibility into prior-day positions/plans unless they are re-synced and reinserted for the new session.
- Dashboard lookups (history, open orders/plans, stats) are session-scoped and select the most recent session for the current `BOT_VERSION`.

## Work Plan

- [x] Clarify requirements and invariants
  - [x] Multi-day positions/plans persist across calendar days and restarts; no max holding duration (LLM decides exits).
  - [x] Rename “session” to “portfolio” as the long-lived scope (run-only session concept not needed).
  - [x] PnL is portfolio/position-based rather than per-day resets.
  - [x] Reporting: cumulative since portfolio start plus per-bot-version slices; daily rollups not required.
  - [x] Risk caps are portfolio-wide (not daily), with identical rules for paper/live.
  - [x] Plan/stop behavior: widen overnight and auto-rearm on restart.
  - [x] Trade/order identity: dedupe processed trades across days/runs using exchange trade/order ids to guard replayed feeds.
  - [x] Timezone for “day” views: Melbourne (AEST) but make configurable.
  - [x] Rollout: can assume fresh DB; no legacy session-only mode needed.

- [x] Data model refactor
  - [x] Add `portfolios` table and config to select/create active portfolio (baseline currency, bot_version tag, created_at).
  - [x] Add `portfolio_id` to trades, positions, open_orders, trade_plans, market_data, ohlcv_bars, indicators, equity_snapshots, llm_calls/llm_traces; add composite indexes on (portfolio_id, symbol/timeframe) where applicable.
  - [x] Introduce optional `run_id` column for telemetry/ops without affecting portfolio scoping.
  - [x] Create `portfolio_days` table (date, timezone, start_equity, end_equity, gross/net pnl, fees, llm_cost, drawdown) for reporting; populate start row on first equity snapshot of the day.
  - [x] Replace `risk_state`/`session_stats_cache` with portfolio-scoped stats cache (exposure totals, gross pnl, fees, llm cost) and drop daily-loss baselines.
    - [x] Add portfolio-scoped stats cache schema without daily baseline columns.
    - [x] Update `PortfolioTracker`/`RiskManager` to read/write portfolio stats cache.
    - [x] Remove `risk_state` start_of_day equity usage across code/tests.
    - [x] Add tests for portfolio stats persistence and restart restore behavior.
  - [x] Deprecate/rename `sessions` table: migrate data into `portfolios` (id swap, bot_version/date preserved), add view/compat shim if needed, and remove session_id foreign keys once portfolio_id is wired.
    - [x] Add migration helpers to backfill `portfolio_id` on sessions and copy bot_version/date into portfolios.
    - [x] Update DAO helpers and services to prefer `portfolio_id` over `session_id` for lookups.
    - [x] Remove `session_id` foreign keys/constraints where portfolio_id is present and add backward-compatible views/shims as needed.
    - [x] Add tests covering migration/backfill and compat accessors.

- [ ] Service initialization changes
  - [x] Update `StrategyRunner` to resolve `portfolio_id` from config/DB (create if missing) and generate a `run_id` for telemetry.
    - [x] Persist `run_id` in telemetry/logging and thread into LLM calls/traces.
  - [x] Thread `portfolio_id` through service constructors and wiring
    - [x] `PortfolioTracker`
    - [x] `RiskManager`
    - [x] `PlanMonitor`
    - [x] `ResyncService`
    - [x] `TradeActionHandler`
    - [x] `MarketDataService`
    - [x] `StrategyOrchestrator`
    - [x] `TradingContext`
- [x] Add portfolio-aware DAO calls in services (positions/orders/trades/market data).
  - [x] `PlanMonitor` uses `portfolio_id` when reading/updating plans and logging trades.
  - [x] `ResyncService` persists and reloads positions/orders/trade sync markers by `portfolio_id`.
  - [x] `StrategyRunner` helper lookups (active/sync symbols, rebuild stats) scope DB reads by `portfolio_id`.
  - [x] `MarketDataService`/OHLCV capture uses `portfolio_id` for logging/pruning.
  - [x] `TradingContext` fetches trades/LLM traces/market data with `portfolio_id` scope.
  - [x] Adjust resync bootstrap to load prior positions/open orders for the portfolio without clearing across restarts.
    - [x] Load stored portfolio positions/open orders into risk manager at startup before exchange fetch.
    - [x] Add restart coverage tests for bootstrap loading without wiping DB snapshots.
  - [ ] Remove session-centric flows (daily session creation, start_of_day reset, session_id logging) and replace with portfolio_id/run_id usage across services and loggers.

- [ ] Accounting and risk
  - [ ] Make `PortfolioTracker` portfolio-scoped: load/apply trades by portfolio_id; persist stats cache keyed by portfolio.
  - [ ] Compute PnL and exposure as portfolio-level metrics; remove daily reset logic and ensure portfolio-days derive mark-to-market deltas for reporting only.
  - [ ] Update `RiskManager` to enforce portfolio-wide caps (order value, exposure, position count) and drop daily-loss gating; honor configurable base currency.
  - [ ] Add configurable timezone handling for portfolio-day reporting (AEST default) when generating `portfolio_days` rows.
  - [ ] Add end-of-day snapshot writer that records equity + open positions/plans without flattening.

- [ ] Trading and monitoring flows
  - [ ] Update `PlanMonitor`/`TradeActionHandler`/`ResyncService` to use `portfolio_id` and preserve open plans/positions across date rollovers.
  - [ ] Implement overnight widening of stops/targets and auto-rearm of plan monitors on restart (configurable policy).
  - [ ] Add portfolio aggregates (PnL, exposure, open positions, costs) to `TradingContext`; include per-bot-version slices if multiple versions share a portfolio.
  - [ ] Dedupe processed trades across days/runs using exchange trade/order ids; add DB uniqueness guard on (portfolio_id, trade_id/client_order_id).

- [ ] Dashboard and telemetry
  - [ ] Add portfolio selector and optional day filter to `dashboard.py`; surface open positions/plans regardless of run.
  - [ ] Show cumulative portfolio metrics and per-bot-version slices; optionally show portfolio-day rollups when available.
  - [ ] Tag telemetry/log messages with `portfolio_id` and `run_id` for traceability.

- [ ] Migrations and tooling
  - [ ] Write a bootstrap/migration script that creates the new schema and seeds an initial portfolio (fresh DB path).

- [ ] Testing and rollout
  - [ ] Expand unit tests for `PortfolioTracker`, `RiskManager`, `ResyncService`, and dashboard loaders to cover portfolio scoping and timezone config.
  - [ ] Add integration tests for a position opened on day 1 and held on day 2 (trade sync, stop/target persistence, overnight widening, risk/exposure checks).
  - [ ] Add tests for trade/order dedupe across runs/days using exchange ids.

- [ ] Documentation and ops
  - [ ] Update `README`/`architecture.md` to describe the new portfolio/daily model and operational scripts.
  - [ ] Document operator flows: starting a new portfolio, rolling daily baselines, and troubleshooting mismatched day PnL vs equity.
