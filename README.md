# Dennis-Day Trading Bot

A laptop-run trading bot that aims to replicate a disciplined professional day trader. Decisions come from a configurable LLM using only fresh market/context data (no back-testing) and heuristic intuition. It includes a risk engine, SQLite storage, structured logging, and a Streamlit dashboard for monitoring and control.

## 🚀 Getting Started

Follow these steps to get the bot up and running in minutes.

### 1. Prerequisites

*   **Python 3.10+** installed.
*   **API Keys**:
    *   **LLM**: Gemini 2.5 Flash API key (default) or OpenAI API key when using `LLM_PROVIDER=OPENAI`.
    *   **Exchange**:
        *   **Gemini Exchange Account**: API Key and Secret (Sandbox or Live).
        *   **Interactive Brokers**: TWS or IB Gateway running with API access enabled, Java installed, and the socket host/port open (default `127.0.0.1:7497` for paper). Make sure the account id you plan to trade with is authorized for API connections.

### 2. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory. You can copy the example below and fill in your details.

**`.env` Example:**

```env
# --- Core Settings ---
TRADING_MODE=PAPER            # 'PAPER' or 'LIVE'
ACTIVE_EXCHANGE=GEMINI        # 'GEMINI'
LLM_PROVIDER=GEMINI           # 'GEMINI' (default) or 'OPENAI'
LLM_MODEL=gemini-2.5-flash    # Override to switch Gemini or OpenAI model ids

# --- Gemini (Google) LLM ---
GEMINI_API_KEY=your_google_gemini_api_key

# --- OpenAI LLM (optional) ---
# Set when LLM_PROVIDER=OPENAI
OPENAI_API_KEY=your_openai_api_key

# --- Exchange: Gemini ---
GEMINI_EXCHANGE_API_KEY=your_gemini_exchange_key
GEMINI_EXCHANGE_SECRET=your_gemini_exchange_secret
GEMINI_SANDBOX_API_KEY=your_gemini_sandbox_key
GEMINI_SANDBOX_SECRET=your_gemini_sandbox_secret

# --- Exchange: Interactive Brokers ---
IB_HOST=127.0.0.1
IB_PORT=7497                  # 7497 (paper) / 7496 (live) unless you changed TWS/Gateway config
IB_CLIENT_ID=1                # Unique per TWS/Gateway API client
IB_ACCOUNT_ID=your_ib_account # e.g., DU1234567
IB_PAPER=true
IB_BASE_CURRENCY=AUD
IB_EXCHANGE=SMART             # Default routing exchange
IB_PRIMARY_EXCHANGE=ASX       # Primary listing venue for ASX equities
IB_ALLOWED_INSTRUMENT_TYPES=STK,FX
IB_STOCK_COMMISSION_PER_SHARE=0.005
IB_STOCK_MIN_COMMISSION=1.0
IB_FX_COMMISSION_PCT=0.0

# --- Risk Management (Adjust to your preference) ---
MAX_DAILY_LOSS=500.0          # Stop trading if loss exceeds this amount
MAX_DAILY_LOSS_PERCENT=3.0    # Stop trading if loss exceeds this % of equity
MAX_ORDER_VALUE=500.0         # Max value per order
MAX_TOTAL_EXPOSURE=1000.0     # Max total portfolio exposure
PORTFOLIO_NAME=default        # Portfolio/run scope for multi-day trades (keeps positions/plans across restarts)
PORTFOLIO_BASE_CURRENCY=USD   # Base currency for risk/pnl
PORTFOLIO_DAY_TIMEZONE=Australia/Sydney  # Day boundary for reporting + overnight widening
```

*See `config.py` for a full list of configurable options.*

### 4. Running the Bot

The easiest way to start is using the helper script, which launches both the trading loop and the dashboard:

```bash
chmod +x run.sh
./run.sh
```

**Manual Startup:**

If you prefer to run components individually:

1.  **Strategy Loop** (The brain):
    ```bash
    python -m trader_bot.strategy_runner
    ```
2.  **Dashboard** (The UI):
    ```bash
    python -m streamlit run trader_bot/dashboard.py
    ```

### Project Layout

- `trader_bot/`: Core package (strategy runner, LLM strategy, risk manager, exchange adapter, dashboard, prompts). Portfolio-aware services persist positions/plans/trades to SQLite for multi-day continuity.
- `tests/`: Pytest suite and shared fixtures.
- `docs/`: Single architecture overview.
- `run.sh`: Helper launcher that starts the runner and dashboard together.
- Runtime artifacts: `trading.db`, `bot.log`, `telemetry.log`, `console.log` in the repo root.

### Testing & Coverage

Run the full suite (with coverage) using:

```bash
python -m pytest
```

This will emit a terminal coverage summary and write `coverage.xml` for tooling.

## 🗂️ Portfolio & Multi-day Ops

- **Portfolios are the unit of state**: trades, plans, positions, telemetry, and processed trade ids are keyed by `PORTFOLIO_NAME` (or the `bot_version` fallback). Keep this stable across restarts to continue managing open positions/plans.
- **Day boundaries are configurable**: `PORTFOLIO_DAY_TIMEZONE` drives reporting rows (`portfolio_days`) and overnight stop/target widening rules.
- **Restart continuity**: the runner writes end-of-day snapshots (equity, open positions, open plans) and resyncs on startup so open inventory persists across days. No daily flattening is performed.
- **Deduping**: trades are deduped per-portfolio by exchange trade id and client order id; processed ids are cached in DB to avoid double-booking across runs/days.
- **Base currency**: `PORTFOLIO_BASE_CURRENCY` feeds risk sizing and exposure calculations (with optional FX provider when trading cross-currency symbols).

## 📚 Documentation

For a deeper dive into how the bot works, check out the documentation:

*   [**Repository Guidelines**](AGENTS.md): Structure, commands, and conventions.
*   [**Architecture Overview**](docs/architecture.md): Single-page map with diagrams and one-line component summaries.
*   [**IB Testing Guide**](docs/ib_testing.md): How to run IB unit/playback tests and refresh fixtures.
