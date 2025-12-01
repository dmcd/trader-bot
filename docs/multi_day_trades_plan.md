# Multi-day Positional Trades Plan

Notes on the current session model:
- A new `sessions` row is created on each runner start with `date` (day-only) and `bot_version`; all state (trades, market data, positions, open orders, plans, indicators, OHLCV, equity snapshots) is keyed by `session_id`.
- Risk is anchored to `start_of_day_equity` stored in `risk_state`/`session_stats_cache`, assuming a single-day session; daily loss checks reset only when a new session baseline is written.
- Resync, portfolio tracking, and plan monitoring all scope to the active `session_id`, so restarting or rolling to a new calendar day drops visibility into prior-day positions/plans unless they are re-synced and reinserted for the new session.
- Dashboard lookups (history, open orders/plans, stats) are session-scoped and select the most recent session for the current `BOT_VERSION`.

## Session cleanup audit (why `session_id` is still hanging around)

- DB tables still require `session_id` as NOT NULL even though `portfolio_id` exists, so the runner must call `get_or_create_portfolio_session` to satisfy inserts (trades/market_data/ohlcv/positions/open_orders/llm_calls/llm_traces).
- DAO APIs remain session-first and update the `sessions` row for side effects (`total_trades`, `total_fees`, `total_llm_cost`, `net_pnl`), so services and telemetry threads keep threading `session_id`.
- StrategyRunner/Resync/PlanMonitor/TradingContext wire `set_session` helpers and use session-based baselines (starting balance, duration, baseline positions), forcing session plumbing even when portfolio_id is present.
- Dashboard and MetricsDrift read/write via `session_id` (version → session lookup, session_stats_cache, equity snapshots), so UI/ops flows still depend on the legacy session row.
- We can assume a **fresh DB** (no backfill/migration needed); sessions only remain to let us split refactors while tests still pass mid-flight.

## Cleanup tasks (blockers before further portfolio work)

- [x] **Schema: make portfolio first-class, session optional**
  - [x] Update table definitions so `portfolio_id` is `NOT NULL` everywhere (fresh DB) and `session_id` is nullable; keep `sessions` table only for compatibility.
  - [x] Remove `session_stats_cache` creation and session-side fee/trade counters (stats come from `portfolio_stats_cache`); keep compatibility view for tests that still read sessions.
    - [x] Drop `session_stats_cache` creation/accessors in `database.py` and migrate callers to portfolio cache.
    - [x] Stop incrementing session totals (trades/fees/llm_cost) on writes; rely on portfolio cache and leave `sessions` read-only for legacy metadata.
    - [x] Update tests/fixtures to use portfolio stats cache and adjust assertions.
  - [x] Add a lightweight compatibility shim (view/trigger or DAO guard) to map legacy session writes to portfolio_id during the transition.
  - [x] Ensure legacy session constructors attach/issue a portfolio automatically so inserts satisfy the new `portfolio_id` NOT NULL constraints (used by older tests/helpers).
- [ ] **Database API: flip to portfolio-first signatures**
  - [ ] Add portfolio-scoped DAO entry points (`log_trade`, `log_market_data`, `log_llm_*`, `replace_positions/open_orders`, `get_recent_*`) that require `portfolio_id`; keep session-aware shims with deprecation warnings.
    - [ ] Add explicit portfolio-first helpers for trades/market data/OHLCV/positions/open orders/processed trades/trade plans/LLM logs/equity snapshots that do not accept `session_id`.
    - [ ] Wrap legacy session-first helpers to call portfolio helpers and emit deprecation warnings without changing behavior.
    - [ ] Cover the new portfolio-first helpers with unit tests and ensure shims preserve existing call patterns.
  - [ ] Add `ensure_active_portfolio` helper returning `(portfolio_id, run_id)` without creating sessions; make stats accessors use only `portfolio_stats_cache`.
    - [ ] Move stats accessors to portfolio-only entry points and mark session-based stats lookups as deprecated.
    - [ ] Add tests for `ensure_active_portfolio` and portfolio-only stats retrieval.
- [ ] **Runner/services wiring: remove session plumbing**
  - [ ] Drop `StrategyRunner.session_id/session` fields; thread `portfolio_id`+`run_id` into ResyncService, PlanMonitor, TradeActionHandler, MarketDataService, TradingContext, and telemetry without requiring `set_session`.
  - [ ] Remove session-specific baselines (starting_balance, session_started) from runner lifecycle and resync; rely on portfolio stats cache and equity snapshots instead.
- [ ] **Context and risk**
  - [ ] Rework `TradingContext` summaries to use portfolio lifetime (portfolio.created_at or first equity snapshot) and portfolio stats cache; replace `_net_quantity_for_session` with portfolio-level baseline handling for sandbox airdrops.
  - [ ] Replace `MetricsDrift` with a portfolio-scoped equity drift check (uses portfolio stats cache + latest equity snapshot) and retire session_id logging.
  - [ ] Update RiskManager baseline/telemetry hooks to use portfolio metadata and remove any session_id fields.
- [ ] **Dashboard/telemetry**
  - [ ] Switch dashboard loaders to select by portfolio (and optional bot_version filter) instead of `get_session_id_by_version`; read stats from portfolio cache and show run_id metadata.
  - [ ] Remove session_id from telemetry payloads/log lines; ensure run_id/portfolio_id are present everywhere a session_id is currently written.
- [ ] **Tests and fixtures**
  - [ ] Update fixtures/mocks to create portfolios instead of sessions; remove `set_session` expectations in service tests and cover portfolio-only flows.
  - [ ] Add regression tests ensuring no new session rows are created on restart and that portfolio-only writes/readbacks work across trades, plans, orders, market data, and telemetry.

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

- [x] Service initialization changes
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
  - [x] Add initialization tests covering portfolio/run wiring (session reuse, telemetry metadata) and fill any gaps found.
- [x] Add portfolio-aware DAO calls in services (positions/orders/trades/market data).
  - [x] `PlanMonitor` uses `portfolio_id` when reading/updating plans and logging trades.
  - [x] `ResyncService` persists and reloads positions/orders/trade sync markers by `portfolio_id`.
  - [x] `StrategyRunner` helper lookups (active/sync symbols, rebuild stats) scope DB reads by `portfolio_id`.
  - [x] `MarketDataService`/OHLCV capture uses `portfolio_id` for logging/pruning.
  - [x] `TradingContext` fetches trades/LLM traces/market data with `portfolio_id` scope.
  - [x] Adjust resync bootstrap to load prior positions/open orders for the portfolio without clearing across restarts.
    - [x] Load stored portfolio positions/open orders into risk manager at startup before exchange fetch.
    - [x] Add restart coverage tests for bootstrap loading without wiping DB snapshots.
  - [x] Remove session-centric flows (daily session creation, start_of_day reset, session_id logging) and replace with portfolio_id/run_id usage across services and loggers.
    - [x] Stop creating daily sessions in `StrategyRunner`; hydrate portfolio/run context once per start.
    - [x] Remove start_of_day equity reset logic tied to session creation; rely on portfolio stats cache instead.
    - [x] Thread portfolio_id/run_id into log/telemetry metadata where session_id was used.
    - [x] Add regression tests covering restart without new session creation and logging scope.

- [x] Accounting and risk
  - [x] Make `PortfolioTracker` portfolio-scoped: load/apply trades by portfolio_id; persist stats cache keyed by portfolio.
    - [x] Add unit coverage for portfolio-scoped rebuild/apply flows (stats cache read/write, restart restore).
  - [x] Compute PnL and exposure as portfolio-level metrics; remove daily reset logic and ensure portfolio-days derive mark-to-market deltas for reporting only.
    - [x] Remove start-of-day/daily-loss tracking from risk/runner/database and migrate caches to portfolio aggregates with tests.
    - [x] Ensure PortfolioTracker rebuild/apply flows stay portfolio-scoped (no per-day resets) and expose portfolio-level PnL/exposure hooks with restart coverage.
    - [x] Ensure `portfolio_days` rows derive deltas from mark-to-market snapshots only (reporting, not gating risk).
    - [x] Make stats consumers (db/dashboard/context) read from portfolio stats cache and surface exposure_notional totals.
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
  - [ ] Update risk documentation (AGENTS, architecture, research notes) to reflect portfolio-level risk and removal of daily-loss gates.
