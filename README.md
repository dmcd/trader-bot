# AI Trading Bot & MCP Server

This project implements an autonomous trading bot for Interactive Brokers (IB) using `ib_insync`, exposed via a Model Context Protocol (MCP) server, and controlled by a Gemini-powered strategy loop.

## Project Overview

The system consists of three main layers:
1.  **Core Trading Layer (`trader.py`)**: Handles connection to IB Gateway/TWS, market data subscriptions, and order placement.
2.  **Risk Management Layer (`risk_manager.py`)**: Acts as a safety guard, enforcing limits on daily losses and maximum order sizes before any trade is executed.
3.  **Autonomous Strategy Layer (`strategy_runner.py`)**: The "Brain" that fetches data, prompts the Gemini LLM for trading decisions, and executes them via the Risk Manager.
4.  **MCP Server (`server.py`)**: Exposes the bot's capabilities (Get Account, Get Price, Buy, Sell) as standard MCP tools, allowing external LLM agents to interact with the trading engine.

## Key Files

-   `trader.py`: `TraderBot` class. Wraps `ib_insync` for async trading operations.
-   `risk_manager.py`: `RiskManager` class. Checks `MAX_DAILY_LOSS` and `MAX_ORDER_VALUE`.
-   `strategy_runner.py`: Main entry point for the autonomous loop.
-   `server.py`: FastMCP server implementation.
-   `config.py`: Configuration loader (Env vars).
-   `requirements.txt`: Python dependencies.

## Setup & Bootstrap

### 1. Prerequisites
-   **Interactive Brokers Account**: Paper Trading account recommended.
-   **IB Gateway or TWS**: Must be running and listening on port `4002` (Paper) or `7497` (Live).
-   **Python 3.10+**

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```env
# Trading Mode
TRADING_MODE=PAPER

# IB Connection
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1

# Risk Limits (AUD)
MAX_DAILY_LOSS=50.0
MAX_ORDER_VALUE=100.0
MAX_POSITIONS=3

# Gemini API Key (Required for Strategy Runner)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Running the Bot

**The Easy Way (Single Command):**
```bash
chmod +x run.sh
./run.sh
```
This will start the autonomous bot in the background and launch the dashboard in your browser.

**Manual Mode:**

**Autonomous Mode (Strategy Loop):**
```bash
python strategy_runner.py
```

**MCP Server Mode (Tool Exposure):**
```bash
python server.py
```

**Dashboard:**
```bash
streamlit run dashboard.py
```

**Verification:**
-   Run `python test_connect.py` to test IB connection.
-   Run `python test_server.py` to test MCP server tools.
