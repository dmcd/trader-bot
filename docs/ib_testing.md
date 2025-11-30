# Interactive Brokers Testing Guide

This repo ships with IB-focused unit doubles plus an integration smoke that can run in playback (fixture) mode for CI or against a live Gateway/TWS session when credentials are available.

## Unit Coverage
- Shared fakes live in `tests/ib_fakes.py` (`FakeIB`, `FakeTicker`, `FakeTrade`, `FakeBar`, etc.) and a helper loader `load_ib_fixture_bundle` for JSON fixtures.
- Use the fakes directly in unit tests to normalize market data, trades, OHLCV, and order lifecycles without hitting IBKR.
- Quick run: `python -m pytest tests/unit/trader_bot/test_ib_trader.py`.

## Integration Smoke (Playback by Default)
- Test file: `tests/integration/test_ib_playback.py` (marked `@pytest.mark.integration`).
- Default mode (`IB_TEST_MODE=playback`) replays `tests/fixtures/ib/playback_bundle.json` via the shared FakeIB client. Override fixture path with `IB_PLAYBACK_FIXTURE`.
- Live mode: set `IB_TEST_MODE=live` and export `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, and `IB_ACCOUNT_ID`. The test will skip unless those are present; it is intended for manual/paper runs, not CI.
- Run playback smoke locally: `python -m pytest -m integration tests/integration/test_ib_playback.py`.

## Capturing/Refreshing Fixtures
- Script: `tests/integration/ib_fixture_capture.py` connects to IBKR and writes a bundle matching the FakeIB schema.
- Prereqs: Gateway/TWS running, API enabled, env vars set (`IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `IB_ACCOUNT_ID`).
- Usage: `python tests/integration/ib_fixture_capture.py --output tests/fixtures/ib/playback_bundle.json` (edit symbols inside the script if you want different instruments).
- Keep captures sanitized (no real account ids or credentials) before committing.

## Env Flags
- `IB_TEST_MODE`: `playback` (default) or `live`.
- `IB_PLAYBACK_FIXTURE`: path to a JSON fixture for playback mode.
- `IB_HOST`/`IB_PORT`/`IB_CLIENT_ID`/`IB_ACCOUNT_ID`: required for live capture or live integration runs.
- Optional: `IB_INTEGRATION_HOST`, `IB_INTEGRATION_PORT`, `IB_INTEGRATION_CLIENT_ID`, `IB_INTEGRATION_ACCOUNT` mirror values for test-only configs; see `.env.example` for defaults.
