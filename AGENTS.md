# Repository Guidelines

## Project Structure & Module Organization

### Core Components
- **`trader_bot/strategy_runner.py`**: The main orchestration loop. Handles data fetching, risk checks, order placement, and logging.
- **`trader_bot/strategy.py`**: Contains the `LLMStrategy` class, which interfaces with Gemini 2.5 Flash for decision making.
- **`trader_bot/risk_manager.py`**: Enforces safety limits (daily loss, max order value, exposure caps).
- **`trader_bot/cost_tracker.py`**: Tracks trading fees and LLM token usage.
- **`trader_bot/trading_context.py`**: Manages context for the LLM (open orders, market data summary).
- **`trader_bot/database.py`**: SQLite database interface for persisting state and logs.
- **`trader_bot/technical_analysis.py`**: Calculates indicators (RSI, MACD, Bollinger Bands).

### Venue Adapters
- **`trader_bot/gemini_trader.py`**: Adapter for Gemini Exchange (using `ccxt`). Handles sandbox precision fixes.
- **`trader_bot/config.py`**: Central configuration for API keys, limits, and trading mode.

### Interfaces
- **`trader_bot/dashboard.py`**: Streamlit dashboard for monitoring and manual control.
- **`run.sh`**: Helper script to launch the strategy loop and dashboard together.

### Data and Logs
- **`trading.db`**: SQLite database file.
- **`bot.log`**: Human-readable log of high-level actions and decisions.
- **`telemetry.log`**: Structured JSON logs for analysis, including full LLM prompts.
- **`console.log`**: Detailed debug logs capturing stdout/stderr.

## Build, Test, and Development Commands

- **Install dependencies**: `pip install -r requirements.txt`
- **Run tests**: `python -m pytest`
- **Run specific test**: `python -m pytest tests/test_technical_analysis_unit.py`
- **Run strategy loop**: `python -m trader_bot.strategy_runner`
- **Run dashboard**: `python -m streamlit run trader_bot/dashboard.py`
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
- **Location**: Place new tests in `tests/test_*.py` files.
- **Scope**: Name tests for behavior (e.g., `test_stop_target_clamping_when_price_moves`).
- **Regression**: Always run the full suite (`python -m pytest`) before pushing changes, especially for risk or sizing logic.

## Commit & Pull Request Guidelines

- **Commits**: Concise, present-tense summaries (e.g., "Clamp stop/target band"). Scope to a single concern.
- **PRs**: Include a short description, key risk considerations, test output, and screenshots for UI changes.

## Security & Configuration Tips

- **Secrets**: Store in `.env`. Never commit API keys.
- **Tunables**: Review `config.py` for all adjustable parameters (caps, fees, cadence).
- **Venues**: Verify `ACTIVE_EXCHANGE` and `TRADING_MODE` in `.env`. Paper and Live modes use different endpoints and limits.

## Venues & Setup

- **Gemini**: Default ccxt adapter. Requires `GEMINI_EXCHANGE_API_KEY`/`GEMINI_EXCHANGE_SECRET` (or sandbox keys) and `ACTIVE_EXCHANGE=GEMINI`.
- **Interactive Brokers (IB)**: Runs against TWS or IB Gateway with API sockets enabled (Java needed). Typical socket `127.0.0.1:7497` for paper; ensure the account id is authorized. Key env vars: `ACTIVE_EXCHANGE=IB`, `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `IB_ACCOUNT_ID`, `IB_PAPER`, `IB_BASE_CURRENCY`, `IB_EXCHANGE`, `IB_PRIMARY_EXCHANGE`, `IB_ALLOWED_INSTRUMENT_TYPES`, `IB_STOCK_COMMISSION_PER_SHARE`, `IB_STOCK_MIN_COMMISSION`, `IB_FX_COMMISSION_PCT`.
