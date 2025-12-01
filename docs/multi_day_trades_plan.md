# Multi-day Positional Trades Plan

Notes on the current session model:
- A new `sessions` row is created on each runner start with `date` (day-only) and `bot_version`; all state (trades, market data, positions, open orders, plans, indicators, OHLCV, equity snapshots) is keyed by `session_id`.
- Risk is anchored to `start_of_day_equity` stored in `risk_state`/`session_stats_cache`, assuming a single-day session; daily loss checks reset only when a new session baseline is written.
- Resync, portfolio tracking, and plan monitoring all scope to the active `session_id`, so restarting or rolling to a new calendar day drops visibility into prior-day positions/plans unless they are re-synced and reinserted for the new session.
- Dashboard lookups (history, open orders/plans, stats) are session-scoped and select the most recent session for the current `BOT_VERSION`.

## Work Plan

- [ ] Session ➜ Portfolio cleanup
  - [x] Rename `session_stats`/helpers to `portfolio_stats` across `StrategyRunner`, `PortfolioTracker`, `ResyncService`, `TradeActionHandler`, `TradingContext`, telemetry emitters, and tests; update log/telemetry strings to stop calling runs "sessions."
  - [ ] Update LLM context and dashboard consumers to emit/read a `portfolio` block (not `session`), drop `start_new_session` wiring, and ensure baseline math no longer assumes day resets.
  - [x] Replace session-scoped config flags (`LLM_MAX_SESSION_COST`, `MARKET_DATA_RETENTION_MINUTES` comments, etc.) with portfolio-level names/env vars and update docs/consumers.
  - [ ] Delete session-first DB APIs and shims (`get_or_create_session`, session_id params on CRUD/prune helpers, Deprecation warnings) in favor of portfolio/run-only methods; migrate all call sites and tests.
  - [ ] Design and apply a migration that removes `sessions` table dependencies and `session_id` columns/indexes (or formalizes them as optional `run_id` metadata), including `session_portfolios` view teardown/backfill strategy for legacy data.
  - [ ] Update docs (architecture, README/ops/AGENTS) to describe the portfolio + run_id lifecycle and remove remaining session terminology.
  - [ ] Add regression coverage for portfolio-only flows (stats rebuilds, trade sync, plan monitors, market data retention, LLM traces) with no session_id fallback.

- [x] Accounting and risk
  - [x] Make `PortfolioTracker` portfolio-scoped: load/apply trades by portfolio_id; persist stats cache keyed by portfolio.
    - [x] Add unit coverage for portfolio-scoped rebuild/apply flows (stats cache read/write, restart restore).
  - [x] Compute PnL and exposure as portfolio-level metrics; remove daily reset logic and ensure portfolio-days derive mark-to-market deltas for reporting only.
    - [x] Remove start-of-day/daily-loss tracking from risk/runner/database and migrate caches to portfolio aggregates with tests.
    - [x] Ensure PortfolioTracker rebuild/apply flows stay portfolio-scoped (no per-day resets) and expose portfolio-level PnL/exposure hooks with restart coverage.
    - [x] Ensure `portfolio_days` rows derive deltas from mark-to-market snapshots only (reporting, not gating risk).
    - [x] Make stats consumers (db/dashboard/context) read from portfolio stats cache and surface exposure_notional totals.
  - [x] Update `RiskManager` to enforce portfolio-wide caps (order value, exposure, position count) and drop daily-loss gating; honor configurable base currency.
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
