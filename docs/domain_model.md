# Domain Model

The bot persists trading state in SQLite for restart resilience, monitoring, and plan enforcement. Below is the core domain and how entities relate.

## Entity Diagram

```mermaid
classDiagram
    direction LR
    Session "sessions" --> Trade "trades" : 1-to-many
    Session --> TradePlan "trade_plans" : 1-to-many
    Session --> Position "positions" : snapshot
    Session --> OpenOrder "open_orders" : snapshot
    Session --> MarketData "market_data" : 1-to-many
    Session --> OHLCV "ohlcv" : 1-to-many
    Session --> Command "commands" : dashboard→bot
    Session --> LLMTrace "llm_traces" : 1-to-many
    Session --> SessionStatsCache "session_stats_cache" : 1-to-1

    class Session {
      id:int
      date:iso
      starting_balance:float
      net_pnl:float
      total_fees:float
      total_llm_cost:float
      total_trades:int
      created_at
    }

    class Trade {
      id:int
      session_id:int
      timestamp
      symbol
      action:BUY/SELL
      quantity:float
      price:float
      fee:float
      liquidity
      realized_pnl:float
      reason
      trade_id (venue)
    }

    class TradePlan {
      id:int
      session_id:int
      symbol
      side:BUY/SELL
      entry_price:float
      stop_price:float
      target_price:float
      size:float
      status:open/closed
      reason
      entry_order_id
      entry_client_order_id
      version:int
      opened_at
      closed_at
    }

    class Position {
      id:int
      session_id:int
      symbol
      quantity:float
      avg_price:float
      exchange_timestamp
    }

    class OpenOrder {
      id:int
      session_id:int
      order_id
      symbol
      side
      price:float
      amount:float
      remaining:float
      status
      exchange_timestamp
    }

    class MarketData {
      id:int
      session_id:int
      symbol
      timestamp
      price:float
      bid:float
      ask:float
      volume:float
      spread_pct:float
      bid_size:float
      ask_size:float
      ob_imbalance:float
    }

    class OHLCV {
      id:int
      session_id:int
      symbol
      timeframe
      timestamp
      open
      high
      low
      close
      volume
    }

    class Command {
      id:int
      command
      status:pending/done
      created_at
      executed_at
    }

    class LLMTrace {
      id:int
      session_id:int
      prompt
      decision_json
      execution_result
      created_at
    }

    class SessionStatsCache {
      session_id:int (pk)
      total_trades:int
      total_fees:float
      gross_pnl:float
      total_llm_cost:float
      start_of_day_equity:float
    }
```

## Lifecycle & Responsibilities

- **Session**: One per trading day. Seeds `starting_balance`, holds aggregates (`net_pnl`, `total_fees`, `total_llm_cost`, `total_trades`).
- **Market snapshots**: `market_data` stores ticker/ob data each loop; `ohlcv` caches per-timeframe candles for technical analysis and LLM tools.
- **Orders & Plans**: Live open orders are snapshotted in `open_orders` (filtered to our client IDs). `trade_plans` track stop/target intent and plan versions; `entry_client_order_id` links plans to venue orders when available.
- **Trades**: Each fill is logged in `trades` with venue trade_id, fee, and realized PnL. Trade reasons prefer cached order reasons or plan reasons; unattributed trades are skipped.
- **Positions**: Snapshots of current holdings for exposure tracking and context.
- **Commands**: Dashboard-to-bot control messages (e.g., stop/close-all).
- **LLM traces**: Stored prompts/decisions and execution outcomes for audit/telemetry.
- **Session stats cache**: Fast restart of aggregates; rebuilt from trades when stale.

## Data Flow (overview)

1. Loop fetches live market data → `market_data` + `ohlcv` (cached).
2. Open orders/positions refreshed from exchange → filtered to our client IDs → stored in `open_orders`/`positions`.
3. Trades synced from exchange (client ID prefix, cutoff time) → `trades` + stat cache update + plan monitoring.
4. Decisions: LLM context built from `positions`, `open_orders`, `trade_plans`, `recent_trades`, `session` stats.
5. Commands polled from `commands` to allow dashboard control.

## Config knobs

- `CLIENT_ORDER_PREFIX`: ensures only our orders/trades are considered.
- `TRADE_SYNC_CUTOFF_MINUTES`: ignore stale fills when syncing.
- Risk/plan caps in `config.py` drive sizing and plan lifecycle logic.
