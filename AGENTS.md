# Repository Guidelines

## Project Structure & Module Organization

### Core Components
- **`strategy_runner.py`**: The main orchestration loop. Handles data fetching, risk checks, order placement, and logging.
- **`strategy.py`**: Contains the `LLMStrategy` class, which interfaces with Gemini 2.5 Flash for decision making.
- **`risk_manager.py`**: Enforces safety limits (daily loss, max order value, exposure caps).
- **`cost_tracker.py`**: Tracks trading fees and LLM token usage.
- **`trading_context.py`**: Manages context for the LLM (open orders, market data summary).
- **`database.py`**: SQLite database interface for persisting state and logs.
- **`technical_analysis.py`**: Calculates indicators (RSI, MACD, Bollinger Bands).

### Venue Adapters
- **`gemini_trader.py`**: Adapter for Gemini Exchange (using `ccxt`). Handles sandbox precision fixes.
- **`config.py`**: Central configuration for API keys, limits, and trading mode.

### Interfaces
- **`dashboard.py`**: Streamlit dashboard for monitoring and manual control.
- **`run.sh`**: Helper script to launch the strategy loop and dashboard together.

### Data and Logs
- **`trading.db`**: SQLite database file.
- **`bot.log`**: Human-readable log of high-level actions and decisions.
- **`telemetry.log`**: Structured JSON logs for analysis, including full LLM prompts.
- **`console.log`**: Detailed debug logs capturing stdout/stderr.

## Build, Test, and Development Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `python -m pytest`
- **Run specific test**: `python -m pytest test_technical_analysis_unit.py`
- **Run strategy loop**: `python strategy_runner.py`
- **Run dashboard**: `streamlit run dashboard.py`
- **Launch everything**: `./run.sh`

## Coding Style & Naming Conventions

- **Python**: 3.10+
- **Indentation**: 4 spaces
- **Type Hints**: Preferred where practical.
- **Logging**: Use `logger` from `logger_config.py`. Avoid `print`.
- **Naming**: Descriptive `snake_case` for variables/functions, `CapWords` for classes.
- **Config**: Align new configuration flags with existing patterns in `config.py`.

## Testing Guidelines

- **Framework**: `pytest`.
- **Location**: Place new tests in `test_*.py` files.
- **Scope**: Name tests for behavior (e.g., `test_stop_target_clamping_when_price_moves`).
- **Regression**: Always run the full suite (`python -m pytest`) before pushing changes, especially for risk or sizing logic.

## Commit & Pull Request Guidelines

- **Commits**: Concise, present-tense summaries (e.g., "Clamp stop/target band"). Scope to a single concern.
- **PRs**: Include a short description, key risk considerations, test output, and screenshots for UI changes.

## Security & Configuration Tips

- **Secrets**: Store in `.env`. Never commit API keys.
- **Tunables**: Review `config.py` for all adjustable parameters (caps, fees, cadence).
- **Venues**: Verify `ACTIVE_EXCHANGE` and `TRADING_MODE` in `.env`. Paper and Live modes use different endpoints and limits.
