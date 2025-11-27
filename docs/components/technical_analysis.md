# Technical Analysis

`technical_analysis.py` computes indicators used by the strategy.

## Indicators
- RSI, MACD, Bollinger Bands, moving averages, and other helpers the LLM can request.

## Usage
- Called by `trading_context` to include indicator snapshots in the LLM prompt.
- Respects tool limits for bar counts to keep payloads small.

## Tips
- When adding indicators, keep return shapes consistent and document units/periods.
- Avoid heavy computations in the main loop; precompute or cap lengths where possible.
