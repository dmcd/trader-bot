# Tests

- Layout: unit tests live in `tests/unit/trader_bot/...` mirroring the code packages, integration flows live in `tests/integration`, and contract suites can live under `tests/contract` for exchange/LLM compatibility checks.
- Running fast suite: `python -m pytest tests/unit`
- Running integration-only: `python -m pytest -m integration`
- Full run with all markers: `python -m pytest`
- Markers: `integration` for cross-component or event-loop flows, `contract` for venue/LLM compatibility, `llm` for tests that hit real or mocked providers; combine with `-m` to include/exclude.
