# Trading Context

`trading_context.py` builds the snapshot passed into the LLM.

## Contents
- Positions and open orders (sizes, prices, exposure estimates).
- Recent market data: prices, depth, indicators from `technical_analysis.py`.
- Session metrics: PnL, fees, cost ratios, cooldown flags.
- Policy flags: allowed symbols/timeframes, caps, loop interval.

## Responsibilities
- Normalize/limit data sizes (bars, trades, depth) per tool limits in `config.py`.
- Provide concise summaries to keep LLM token use low.

## Tips
- Add new fields carefully: keep shapes stable to avoid breaking prompt validation.
- When expanding tool outputs, respect `TOOL_MAX_*` limits to avoid oversized prompts.
