# Architecture

Concise view of how the bot fits together, with a sentence per component and two quick diagrams to keep it maintainable.

## System Diagram

```mermaid
flowchart TD
    subgraph Exchange["Gemini via ccxt"]
        Ticker["Ticker/Order Book"]
        Orders["Orders/Fills"]
        Trades["Trade History"]
    end

    Runner["strategy_runner.py"] -->|market data| TA["technical_analysis.py"]
    Runner -->|decisions| Strategy["strategy.py (LLMStrategy)"]
    Runner -->|risk checks| Risk["risk_manager.py"]
    Runner -->|fees| Costs["cost_tracker.py"]
    Runner -->|orders| Trader["gemini_trader.py"]
    Runner -->|persist| DB["database.py (SQLite)"]
    Runner -->|context| Context["trading_context.py"]
    Runner -->|ui data| Dashboard["dashboard.py (Streamlit)"]
    Strategy -->|LLM calls| LLM["Gemini/OpenAI"]
    Trader --> Exchange
    Trader <-->|fills| Runner
    Exchange -->|trades/tickers| Runner
```

## Loop Sequence (happy path)

```mermaid
sequenceDiagram
    participant Ex as Exchange
    participant R as StrategyRunner
    participant S as LLMStrategy
    participant Risk as RiskManager
    participant T as GeminiTrader
    participant DB as TradingDatabase

    R->>Ex: fetch ticker/order book
    R->>DB: log market snapshot
    R->>S: ask for decision with context
    S->>LLM: planner + decision prompts
    LLM-->>S: JSON decision
    S-->>R: StrategySignal (action/size/targets)
    R->>Risk: check limits/spacing/liquidity
    alt allowed
        R->>T: place_order_async
        T-->>R: order result + liquidity
        R->>DB: log_trade + update stats
    else blocked
        R-->>S: on_trade_rejected
    end
    R->>Dashboard: updated metrics/logs
```

## Components (one-liners)
- `strategy_runner.py`: Orchestrates the loop, wires services, enforces spacing/slippage/liquidity/risk, and logs everything.
- `strategy.py` (`LLMStrategy`): Builds planner/decision prompts, normalizes tool hints, sizes trades, and handles cooldowns.
- `trading_context.py`: Packages positions, orders, summaries, and regime flags for the LLM.
- `technical_analysis.py`: Computes RSI, MACD, Bollinger Bands, SMAs, and simple signal summaries.
- `risk_manager.py`: Enforces order value, exposure caps, position count, and daily loss guardrails.
- `gemini_trader.py`: ccxt adapter for Gemini with precision fixes, post-only handling, and order/trade sync.
- `cost_tracker.py`: Estimates exchange fees and LLM token costs for net PnL.
- `database.py`: SQLite schema/helpers for sessions, trades, prompts/traces, OHLCV, equity, positions, open orders, commands, and trade plans.
- `dashboard.py`: Streamlit UI for performance, costs, health, history, logs, and control commands.
- `config.py`: Central tunables for API keys, limits, cadence, and modes.
