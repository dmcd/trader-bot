# Technical Review

## Architecture / Maintainability
- [ ] Strategy execution, risk, and DB concerns are tightly coupled: the runner calls DB methods directly from many paths, while `RiskManager` holds mutable state but depends on the runner to hydrate prices. A clearer boundary (e.g., an `ExposureService` fed by normalized market snapshots) would simplify tests and reduce duplicated price lookups.
- [ ] The LLM strategy builds prompts and tools inline but the runner still fetches its own market data and OHLCV separately. A single data provider interface would reduce duplication and make unit tests deterministic.
- [ ] SQLite access uses a shared connection with `check_same_thread=False` but no locking; multiple async tasks call into the DB concurrently (equity logging, trade syncing, plan updates) which risks intermittent failures and partial writes.

## Testing markers
- `integration`: long-running or cross-component flows (runner, DB round-trips, shadow runner).
- `contract`: compatibility/contract checks against venues or LLM providers.
- `llm`: tests that touch real or mocked LLM providers; exclude in fast suites unless explicitly targeted.
- Quick runs: `python -m pytest tests/unit` for unit-only, `python -m pytest -m "integration or contract"` for heavier suites, or combine with `-m "not integration and not contract"` to keep CI fast.
