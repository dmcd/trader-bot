# Test Suite Restructure Plan

Context: 48 test modules under `tests/`, mixing `unittest` and `pytest`, with no clear unit/integration split. Several files cover the same surface area (e.g., database and strategy) and a few modules carry little or no assertion value (e.g., `tests/test_server.py` defines only stubs).

## Target shape (idiomatic pytest layout)
- `tests/unit/trader_bot/...` mirrors `trader_bot/` modules (e.g., `tests/unit/trader_bot/test_database.py`, `.../strategy/test_strategy.py`).
- `tests/unit/trader_bot/services/...` mirrors `trader_bot/services/`.
- `tests/integration/...` for runner/LLM/DB flows that spin event loops or touch multiple components.
- `tests/contract/...` reserved for exchange/LLM contract checks or shadow-runner safety nets.
- Shared fixtures live in `tests/conftest.py` (global) and scoped `tests/unit/conftest.py` / `tests/integration/conftest.py` as needed; avoid duplicating helpers inside individual test files.

## Action plan
- [ ] Normalize layout and naming
  - [ ] Create `tests/unit/trader_bot/` and move current unit-leaning modules there; keep service tests in `tests/unit/trader_bot/services/` to mirror package structure.
  - [ ] Create `tests/integration/` and move higher-level flows (`test_runner_integration_stub.py`, `test_strategy_runner_control_paths.py`, `test_restart_recovery.py`, `test_tool_roundtrip_integration.py`, `test_server_shadow_runner.py`, `test_trade_sync.py`) into it; add `@pytest.mark.integration`.
  - [ ] Remove or fix `tests/test_server.py` (currently defines stubs with no assertions); if kept, rewrite as an integration/contract check that exercises the public server API.
- [ ] Consolidate overlapping suites
  - [ ] Merge `test_database_unit.py` and `test_database_additional.py` into a single `tests/unit/trader_bot/test_database.py` with a shared fixture, parameterized pruning cases, and explicit integration markers for any slow DB work.
  - [ ] Merge strategy-focused files (`test_strategy.py`, `test_strategy_additional_unit.py`, and portions of `test_strategy_llm_validation.py`) under `tests/unit/trader_bot/strategy/` using shared setup fixtures; split out any true LLM/plan-monitor flows into integration tests.
  - [ ] Group runner/risk/slippage spacing tests (`test_runner_sleep_spacing.py`, `test_sandbox_daily_loss.py`, `test_risk_exposure.py`, `test_multi_symbol_exposure.py`, `test_order_value_buffer.py`, `test_slippage_guard.py`) by concern (risk vs execution pacing) to reduce fixture duplication and make expectations easier to scan.
- [ ] Fixture and helper cleanup
  - [ ] Centralize fake logger/test DB/temp file fixtures in `tests/conftest.py`; remove ad-hoc temp DB setup from individual files in favor of shared fixtures/fixtures using `tmp_path`.
  - [ ] Prefer `pytest`-style tests over `unittest.TestCase` where possible (drop `setUp/tearDown` boilerplate, use fixtures and parametrization).
  - [ ] Add factories/builders for common payloads (market data points, strategy signals, trade plans) to eliminate repeated dict literals across strategy/runner tests.
- [ ] Markers and tooling
  - [ ] Extend `pytest.ini` with markers (`integration`, `contract`, `llm`) and add usage docs to `README.md` or `docs/technical_review.md`.
  - [ ] Add `tests/README.md` describing layout, how to run fast vs full suites, and conventions for new tests.
- [ ] CI/test performance
  - [ ] Split CI to run unit suite on every push and integration/contract nightly or on-demand.
  - [ ] Track duration of the slowest tests after the move; target <30s for unit suite by trimming sleeps and using fakes instead of real event-loop waits.

## Optional follow-ups
- [ ] Add coverage for gaps called out by component owners (e.g., risk manager edge cases, prompt budget enforcement under multi-symbol scenarios) once the layout is stable.
- [ ] Introduce property-based tests (Hypothesis) for numeric guards like exposure caps and stop/target clamping.
