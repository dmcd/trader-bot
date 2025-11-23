# Strategy Architecture

The trading bot uses a **Strategy Pattern** to decouple trading logic from execution. This allows for easy testing, swapping of strategies, and clear separation of concerns.

## Core Components

### 1. `BaseStrategy` (Abstract Base Class)
Located in `strategy.py`, this defines the interface that all strategies must implement.

-   **`generate_signal(...)`**: The main method called by the runner. It receives market data and context, and returns a `StrategySignal` (BUY/SELL/HOLD) or `None`.
-   **`on_trade_executed(...)`**: Callback to update internal state (e.g., last trade time) after a successful order.
-   **`on_trade_rejected(...)`**: Callback to handle rejected orders.

### 2. `LLMStrategy` (Concrete Implementation)
The default strategy that uses Google's Gemini 2.5 Flash model to make decisions.

-   **Process**:
    1.  **Filters**: Checks for "chop" (low volatility) and high fees.
    2.  **Context**: Gathers market data, technical indicators (RSI, MACD, BB), and account stats.
    3.  **Prompt**: Constructs a detailed prompt for the LLM.
    4.  **Decision**: Parses the LLM's JSON response into a signal.
-   **State**: Tracks `last_trade_ts` to enforce cooldowns and `_last_break_glass` for priority overrides.

## Extending the Bot

To create a new strategy (e.g., `MeanReversionStrategy`):

1.  Create a new class inheriting from `BaseStrategy`.
2.  Implement `generate_signal`.
3.  In `strategy_runner.py`, swap `LLMStrategy` with your new class.

```python
class MeanReversionStrategy(BaseStrategy):
    async def generate_signal(self, session_id, market_data, current_pnl, context=None):
        # Your logic here
        return StrategySignal(action="BUY", quantity=0.1, reason="RSI < 30")
```
