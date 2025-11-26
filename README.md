# Dennis-Day Trading Bot

An autonomous trading bot that trades on Interactive Brokers or Gemini using Gemini 2.5 Flash for decision making. It features a robust risk engine, SQLite logging, and a Streamlit dashboard for monitoring.

## 🚀 Getting Started

Follow these steps to get the bot up and running in minutes.

### 1. Prerequisites

*   **Python 3.10+** installed.
*   **API Keys**:
    *   **Gemini (Google)**: An API key for Gemini 2.5 Flash (for the LLM strategy).
    *   **Exchange**:
        *   **Gemini Exchange**: API Key and Secret (Sandbox recommended for testing).
        *   **Interactive Brokers**: IB Gateway or TWS installed and running.

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
ACTIVE_EXCHANGE=GEMINI        # 'GEMINI' or 'IB'

# --- Gemini (Google) LLM ---
GEMINI_API_KEY=your_google_gemini_api_key

# --- Exchange: Gemini ---
GEMINI_EXCHANGE_API_KEY=your_gemini_exchange_key
GEMINI_EXCHANGE_SECRET=your_gemini_exchange_secret
GEMINI_SANDBOX_API_KEY=your_gemini_sandbox_key
GEMINI_SANDBOX_SECRET=your_gemini_sandbox_secret

# --- Exchange: Interactive Brokers ---
IB_HOST=127.0.0.1
IB_PORT=4002                  # 4002 for Paper, 7497 for Live
IB_CLIENT_ID=1

# --- Risk Management (Adjust to your preference) ---
MAX_DAILY_LOSS=500.0          # Stop trading if loss exceeds this amount
MAX_DAILY_LOSS_PERCENT=3.0    # Stop trading if loss exceeds this % of equity
MAX_ORDER_VALUE=500.0         # Max value per order
MAX_TOTAL_EXPOSURE=1000.0     # Max total portfolio exposure
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
    python strategy_runner.py
    ```
2.  **Dashboard** (The UI):
    ```bash
    streamlit run dashboard.py
    ```

## 📚 Documentation

For a deeper dive into how the bot works, check out the documentation:

*   [**Strategy Architecture**](docs/strategy.md): Learn about the `LLMStrategy` and decision flow.
*   [**Repository Guidelines**](AGENTS.md): Structure and conventions.
