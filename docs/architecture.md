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
    subgraph Services
        Orchestrator["strategy_orchestrator.py"]
        Health["health_manager.py"]
        Plans["plan_monitor.py"]
        Commands["command_processor.py"]
        DataSvc["market_data_service.py"]
        Portfolio["portfolio_tracker.py"]
        Actions["trade_action_handler.py"]
        Resync["resync_service.py"]
    end

    Runner["strategy_runner.py"] --> Orchestrator
    Orchestrator --> Commands
    Orchestrator --> Plans
    Orchestrator --> Health
    Orchestrator --> Risk["risk_manager.py"]
    Runner --> DataSvc
    Runner --> Portfolio
    Runner --> Actions
    Runner --> Resync
    Runner -->|market data| TA["technical_analysis.py"]
    Runner -->|decisions| Strategy["strategy.py (LLMStrategy)"]
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
    participant Orch as StrategyOrchestrator
    participant T as GeminiTrader
    participant DB as TradingDatabase

    R->>Orch: process dashboard commands + budget gates
    Orch-->>R: continue or stop
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

## Delegation Boundaries
- `StrategyRunner` owns the loop clock and session lifecycle, but defers command handling and stop logic to `StrategyOrchestrator`.
- `StrategyOrchestrator` fans out to `CommandProcessor`, `HealthCircuitManager`, `PlanMonitor`, and `RiskManager` so the loop body stays thin.
- Market capture and pruning run through `MarketDataService`; trade executions/plan updates route via `TradeActionHandler`.
- Holdings, PnL, and fee/LLM cost accounting live in `PortfolioTracker`, while `ResyncService` handles state reconciliation on startup and trade sync inside the loop.

## Components (one-liners)
- `strategy_runner.py`: Thin orchestrator loop that wires services, hands telemetry/risk gating to `StrategyOrchestrator`, and owns the trading lifecycle.
- `services/strategy_orchestrator.py`: Lifecycle harness for start/stop/cleanup, command processing, budget gates, plan monitor coordination, and market-data health checks.
- `services/command_processor.py`: Executes dashboard-issued commands (stop bot, close positions) ahead of each loop.
- `services/plan_monitor.py`: Manages open plan stops/targets, trailing rules, and auto-flatten windows.
- `services/health_manager.py`: Circuit breaker for exchange/tool streaks plus market-data staleness/latency gating.
- `services/market_data_service.py`: OHLCV capture/pruning helpers and timeframe parsing for cadence guards.
- `services/portfolio_tracker.py`: Tracks holdings and session stats, rebuilds from exchange trades, and persists caches.
- `services/trade_action_handler.py`: Executes plan actions (update/partial/close/pause), RR filters, slippage checks, and liquidity guards.
- `services/resync_service.py`: Reconciles DB snapshots with exchange state and syncs recent trades on startup and during loops.
- `strategy.py` (`LLMStrategy`): Builds planner/decision prompts, normalizes tool hints, sizes trades, and handles cooldowns.
- `trading_context.py`: Packages positions, orders, summaries, and regime flags for the LLM.
- `technical_analysis.py`: Computes RSI, MACD, Bollinger Bands, SMAs, and simple signal summaries.
- `risk_manager.py`: Enforces order value, exposure caps, position count, and daily loss guardrails.
- `gemini_trader.py`: ccxt adapter for Gemini with precision fixes, post-only handling, and order/trade sync.
- `cost_tracker.py`: Estimates exchange fees and LLM token costs for net PnL.
- `database.py`: SQLite schema/helpers for sessions, trades, prompts/traces, OHLCV, equity, positions, open orders, commands, and trade plans.
- `dashboard.py`: Streamlit UI for performance, costs, health, history, logs, and control commands.
- `config.py`: Central tunables for API keys, limits, cadence, and modes.
