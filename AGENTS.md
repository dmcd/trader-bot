# Repository Guidelines

## Project Structure & Module Organization
- Core loop and orchestration: `strategy_runner.py` (LLM-driven decisions), `strategy.py` (LLMStrategy), `shadow_runner.py` (dry-run), `risk_manager.py` and `cost_tracker.py` (controls), `trading_context.py` and `database.py` (state), `technical_analysis.py` (indicators).
- Venue adapters: `gemini_trader.py` (ccxt-based) and `ib_trader.py` (ib_insync) with shared `config.py` for limits and API keys.
- Interfaces: `dashboard.py` (Streamlit monitor), `server.py` (FastMCP tools), `run.sh` (one-shot loop + dashboard).
- Data and logs: `trading.db` (SQLite), `bot.log` (human-readable), `console.log` (debug).
- Docs and tests: `docs/strategy.md` plus `test_*.py` modules for unit coverage.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`.
- Full test suite: `python -m pytest` (use `python -m pytest test_technical_analysis_unit.py` for a targeted file).
- Run strategy loop only: `python strategy_runner.py`.
- Run dashboard only: `streamlit run dashboard.py`.
- MCP server tools (IB): `python server.py`.
- Combined launcher: `chmod +x run.sh && ./run.sh` (loop + dashboard).

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; prefer type hints where practical.
- Keep functions small with explicit logging instead of print; reuse `logger` from `logger_config.py`.
- Use descriptive snake_case for variables/functions, CapWords for classes, and align new config flags with existing names in `config.py`.

## Testing Guidelines
- Framework: pytest; place new tests in `test_*.py` alongside existing unit files.
- Name tests for behavior (e.g., `test_stop_target_clamping_when_price_moves`).
- When adding risk or sizing logic, include regression tests and run `python -m pytest` before pushing.

## Commit & Pull Request Guidelines
- Commits: concise, present-tense summaries (e.g., "Clamp stop/target band"), scoped to a single concern when possible.
- PRs: include a short description, key risk considerations, tests run (`pytest` output), and screenshots/GIFs for dashboard changes.

## Security & Configuration Tips
- Secrets live in `.env`; never commit keys. Review `config.py` for all tunables (caps, fee overrides, cadence).
- Verify venue choice via `ACTIVE_EXCHANGE` and `TRADING_MODE`; paper/live settings change API endpoints and limits.
