# Dennis-Day Trading Bot

A laptop-run trading co-pilot that aims to replicate a disciplined professional day trader on Gemini. Decisions come from Gemini 2.5 Flash using only fresh market/context data (no pretraining on your history) plus heuristic intuition from the runner’s guards. It runs manually from your machine, with a robust risk engine, SQLite logging, and a Streamlit dashboard for monitoring and control.

## 🚀 Getting Started

Follow these steps to get the bot up and running in minutes.

### 1. Prerequisites

*   **Python 3.10+** installed.
*   **API Keys**:
    *   **Gemini (Google)**: An API key for Gemini 2.5 Flash (for the LLM strategy).
    *   **Exchange**:
        *   **Gemini Exchange Account**: API Key and Secret (Sandbox or Live).

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

# --- Gemini (Google) LLM ---
GEMINI_API_KEY=your_google_gemini_api_key

# --- Exchange: Gemini ---
GEMINI_EXCHANGE_API_KEY=your_gemini_exchange_key
GEMINI_EXCHANGE_SECRET=your_gemini_exchange_secret
GEMINI_SANDBOX_API_KEY=your_gemini_sandbox_key
GEMINI_SANDBOX_SECRET=your_gemini_sandbox_secret

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
