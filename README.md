# AI Trading Bot (IB or Gemini) + MCP + Dashboard

An autonomous trading loop that talks to Interactive Brokers or Gemini via async adapters, guards orders with a risk engine, logs everything to SQLite, and exposes tooling through both a Streamlit dashboard and an MCP server.

## What’s Inside
- **Strategy loop (`strategy_runner.py`)** – orchestrates data fetch, Gemini 2.5 Flash decisioning, risk checks, order placement, logging, and cost tracking.
- **Brokers** – `ib_trader.py` (IB via `ib_insync`) and `gemini_trader.py` (Gemini via `ccxt`, sandbox-aware precision fixes).
- **Risk & cost controls** – `risk_manager.py` for order/daily/exposure caps, `cost_tracker.py` for fees and LLM usage.
- **State & context** – `database.py` (SQLite), `trading_context.py` for LLM context, `technical_analysis.py` for RSI/MACD/Bollinger/SMA.
- **Interfaces** – `dashboard.py` (Streamlit control/monitor), `server.py` (FastMCP tools for IB), `run.sh` helper to start loop + dashboard, logs in `bot.log` (human) and `console.log` (debug).

## Documentation
-   [**Strategy Architecture**](docs/strategy.md): How the `LLMStrategy` works and how to build your own.
-   [**Repository Guidelines**](AGENTS.md): Contributor guide, structure, commands, and conventions.

## Requirements
- Python 3.10+
- For IB: IB Gateway/TWS running (default paper port 4002; live 7497) and an IB account.
- For Gemini: API keys (live and/or sandbox). The trading loop always uses a single active venue: `IB` **or** `GEMINI`.

## Setup
```bash
pip install -r requirements.txt
```

Create a `.env` in the repo root (fill the secrets you use):
```env
# Core routing
TRADING_MODE=PAPER            # PAPER or LIVE
ACTIVE_EXCHANGE=GEMINI        # IB or GEMINI

# IB
IB_HOST=127.0.0.1
IB_PORT=4002
IB_CLIENT_ID=1

# Gemini (LLM and exchange)
GEMINI_API_KEY=your_gemini_llm_key
GEMINI_EXCHANGE_API_KEY=your_gemini_key
GEMINI_EXCHANGE_SECRET=your_gemini_secret
GEMINI_SANDBOX_API_KEY=your_sandbox_key
GEMINI_SANDBOX_SECRET=your_sandbox_secret

# Risk + sizing (set to your appetite)
MAX_DAILY_LOSS=500.0
MAX_DAILY_LOSS_PERCENT=3.0
MAX_ORDER_VALUE=500.0
ORDER_VALUE_BUFFER=1.0        # trim sizes to stay just under the cap (helps avoid rejections)
MAX_TOTAL_EXPOSURE=1000.0

# Loop cadence & safety
LOOP_INTERVAL_SECONDS=300
MIN_TRADE_INTERVAL_SECONDS=300
FEE_RATIO_COOLDOWN=50.0
PRIORITY_MOVE_PCT=1.5
PRIORITY_LOOKBACK_MIN=5
BREAK_GLASS_COOLDOWN_MIN=60
BREAK_GLASS_SIZE_FACTOR=0.6
```
See `config.py` for every configurable option (fee overrides, sizing caps, etc.).

## Running
- **Quick start (loop + dashboard)**: `chmod +x run.sh && ./run.sh`
- **Strategy loop only**: `python strategy_runner.py`
- **Dashboard only**: `streamlit run dashboard.py`
- **MCP server (IB tools)**: `python server.py`
- **Smoke the MCP wiring**: `python test_server.py`

The loop stores state in `trading.db`, writes human-readable events to `bot.log`, and detailed diagnostics to `console.log`.

## Testing
- Install deps (includes pytest): `pip install -r requirements.txt`
- Run all tests: `python -m pytest`
- Run a specific file while iterating: `python -m pytest test_technical_analysis_unit.py`

## Notes & limitations
- One active venue at a time (`ACTIVE_EXCHANGE`), and the loop currently fetches/trades a single symbol per venue (`BHP` for IB, `BTC/USD` for Gemini).
- Paper mode on Gemini uses sandbox URLs and backfills missing precision metadata so orders format correctly.
- Fees/LLM costs are tracked per session; the dashboard shows realized/unrealized PnL, exposure, costs, and recent trades.
