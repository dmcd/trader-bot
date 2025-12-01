# Multi-day Positional Trades Plan

Legacy session model (now removed):
- Prior versions created a `sessions` row on each runner start with `date`/`bot_version` and keyed all state off `session_id`.
- Risk and dashboards were scoped to the active session, so restarting or rolling days dropped visibility unless data was re-synced into a new session row.
- Portfolio/run scoping replaces session_id and should be the only path forward.

## Work Plan

- [x] Accounting and risk
  - [x] Make `PortfolioTracker` portfolio-scoped: load/apply trades by portfolio_id; persist stats cache keyed by portfolio.
    - [x] Add unit coverage for portfolio-scoped rebuild/apply flows (stats cache read/write, restart restore).
  - [x] Compute PnL and exposure as portfolio-level metrics; remove daily reset logic and ensure portfolio-days derive mark-to-market deltas for reporting only.
    - [x] Remove start-of-day/daily-loss tracking from risk/runner/database and migrate caches to portfolio aggregates with tests.
    - [x] Ensure PortfolioTracker rebuild/apply flows stay portfolio-scoped (no per-day resets) and expose portfolio-level PnL/exposure hooks with restart coverage.
    - [x] Ensure `portfolio_days` rows derive deltas from mark-to-market snapshots only (reporting, not gating risk).
    - [x] Make stats consumers (db/dashboard/context) read from portfolio stats cache and surface exposure_notional totals.
  - [x] Update `RiskManager` to enforce portfolio-wide caps (order value, exposure, position count) and drop daily-loss gating; honor configurable base currency.
  - [x] Add configurable timezone handling for portfolio-day reporting (AEST default) when generating `portfolio_days` rows.
  - [x] Add end-of-day snapshot writer that records equity + open positions/plans without flattening.
    - [x] Define schema/storage for EOD snapshots (equity, open positions, open plans, timestamp, timezone).
    - [x] Add DB helpers to write/read latest EOD snapshot per portfolio (use portfolio-day timezone).
    - [x] Hook strategy runner to emit EOD snapshot on shutdown/rollover; include tests for restart restore.
    - [x] Expose snapshot data to dashboard/context consumers for visibility (optional toggle).
  - [x] Remove day end flattening of open positions/plans.

- [ ] Trading and monitoring flows
  - [x] Update `PlanMonitor`/`TradeActionHandler`/`ResyncService` to use `portfolio_id` and preserve open plans/positions across date rollovers.
    - [x] Require portfolio_id to be set for plan monitoring/action/resync paths and thread it from runner/services on restart.
    - [x] Preserve open plans/positions/open orders across rollovers by seeding from DB snapshots instead of day-specific clears.
    - [x] Add restart coverage for plan monitor/action/resync flows to prove plans/positions persist across days.
  - [x] Implement overnight widening of stops/targets and auto-rearm of plan monitors on restart (configurable policy).
    - [x] Add config flags + policy fields for overnight widening and auto-rearm (percent/absolute deltas, max widen, enable toggles).
    - [x] Persist per-plan overnight state (last widened timestamp/version, widened prices) to avoid re-widening and support restart restores.
    - [x] Apply widening + auto-rearm in plan monitoring/resync/runner startup so monitors resume with widened stops/targets after rollovers.
    - [x] Add unit coverage for overnight widen + restart flows, including opt-out and single-application guards.
  - [x] Add portfolio aggregates (PnL, exposure, open positions, costs) to `TradingContext`.
  - [ ] Dedupe processed trades across days/runs using exchange trade/order ids; add DB uniqueness guard on (portfolio_id, trade_id/client_order_id).
    - [ ] Add DB uniqueness constraints/indexes on processed trades (portfolio_id, trade_id) and (portfolio_id, client_order_id) to block duplicates.
    - [ ] Ensure resync/runner dedupes via exchange trade/order ids across days/runs without session scoping (persist + hydrate).
    - [ ] Add restart/day-rollover coverage for deduping + constraint handling.

- [ ] Dashboard and telemetry
  - [ ] surface open positions/plans regardless of run.
  - [ ] Tag telemetry/log messages with `portfolio_id` and `run_id` for traceability.

- [ ] Testing and rollout
  - [ ] Add regression coverage for portfolio-only flows (stats rebuilds, trade sync, plan monitors, market data retention, LLM traces) with no session_id fallback.
  - [ ] Expand unit tests for `PortfolioTracker`, `RiskManager`, `ResyncService`, and dashboard loaders to cover portfolio scoping and timezone config.
  - [ ] Add integration tests for a position opened on day 1 and held on day 2 (trade sync, stop/target persistence, overnight widening, risk/exposure checks).
  - [ ] Add tests for trade/order dedupe across runs/days using exchange ids.

- [ ] Documentation and ops
  - [ ] Update `README`/`architecture.md` to describe the new portfolio/daily model and operational scripts.
