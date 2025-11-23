# Backtesting Engine

The Backtesting Engine allows you to simulate strategy performance using historical data stored in `trading.db`. This is crucial for validating changes and tuning prompts without risking real capital.

## How It Works

The `BacktestEngine` (`backtester.py`) replays a past trading session event-by-event.

1.  **Data Loading**: Fetches all `market_data` for a specific `session_id` from the database.
2.  **Simulation Loop**: Iterates through the data chronologically.
    -   Updates the "current time" in the context.
    -   Feeds the data to the Strategy's `generate_signal` method.
    -   Simulates order execution, fees, and PnL updates.
3.  **Reporting**: Generates a report with Initial/Final Balance, Total Return, and a Trade Log.

## Running a Backtest

### Via Dashboard (Recommended)
1.  Start the dashboard: `streamlit run dashboard.py`
2.  Navigate to the **🧪 Backtesting** tab.
3.  Select a historical session from the dropdown.
4.  Click **▶️ Run Backtest**.
5.  View the results, including the Equity Curve and Trade Log.

### Via Command Line
You can use the `run_backtest.py` script as a template:

```bash
python run_backtest.py
```

## Key Features
-   **Time Travel**: The `database.py` and `strategy.py` have been updated to respect a `before_timestamp` or `current_time` context, ensuring the strategy only sees data "available" at that simulated moment.
-   **Fee Simulation**: Uses `CostTracker` to apply realistic exchange fees (e.g., Gemini Taker fees).
