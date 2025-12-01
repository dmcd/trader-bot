# Repository Guidelines

## Project Structure & Module Organization
- Core logic in `trader_bot/`: `strategy_runner.py` orchestrates the loop; `strategy.py` holds the LLM strategy; `risk_manager.py`, `cost_tracker.py`, and `trading_context.py` cover guardrails, fee usage, and prompt context; `database.py` persists state; `technical_analysis.py` provides indicators.
- Venue adapters: `gemini_trader.py` for crypto (ccxt), `ib_trader.py`/`ib_contracts.py` for Interactive Brokers routing; `config.py` centralizes tunables and env parsing.
- Interfaces: `dashboard.py` (Streamlit UI), `run.sh` (dashboard + loop launcher).
- Data/logs live in the repo root: `trading.db`, `bot.log`, `telemetry.log`, `console.log`. Tests sit in `tests/`, docs in `docs/`.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Run all tests: `python -m pytest`
- Run a focused test: `python -m pytest tests/test_technical_analysis_unit.py`
- Start strategy loop: `python -m trader_bot.strategy_runner`
- Launch dashboard: `python -m streamlit run trader_bot/dashboard.py`
- Launch loop + dashboard together: `./run.sh` (make executable with `chmod +x run.sh`)

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, descriptive `snake_case` functions/variables, `CapWords` classes.
- Prefer type hints where practical.
- Logging via `logger` from `logger_config.py`; avoid `print`.
- Configuration flags follow patterns in `config.py`; prefer portfolio/run identifiers over session ids.
- Keep comments succinct and purposeful; default to ASCII.

## Testing Guidelines
- Framework: `pytest`. Place tests in `tests/test_*.py`; name by behavior (e.g., `test_stop_target_clamping_when_price_moves`).
- Run `python -m pytest` before pushing, especially after changes to risk, sizing, or exchange logic.
- Coverage is reported via `coverage.xml`; keep regressions in check when touching core logic.

## Commit & Pull Request Guidelines
- Commits: concise, present tense, single concern (e.g., “Clamp stop/target band”).
- PRs: include a short description, key risk considerations, and test output; add screenshots for UI changes.
- Link issues when applicable; note configuration changes or new env vars in the description.

## Security & Configuration Tips
- Store secrets in `.env`; never commit API keys. Sample envs live in `.env.example`, `.env-ib`, `.env-gemini`, `.env-live`.
- Verify venue/env settings: `ACTIVE_EXCHANGE` (`GEMINI` or `IB`), `TRADING_MODE` (`PAPER`/`LIVE`), and `IB_ALLOWED_INSTRUMENT_TYPES` when enabling FX.
- Review caps in `config.py` (loss limits, max order value, exposure, spread/tick guards) before running live. 
