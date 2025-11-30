# Research Findings and Roadmap

## Current Architecture Snapshot
- **Orchestration:** `StrategyRunner` wires exchange access, risk, plan monitoring, portfolio tracking, telemetry, and the LLM strategy into a single loop that can run with or without live order execution. It seeds health circuits, market data capture, resync, and action handling so the bot can behave like a disciplined discretionary trader with strong guardrails. 【F:trader_bot/strategy_runner.py†L88-L206】
- **LLM-driven decisions:** `LLMStrategy` builds prompts from live market data, exposure stats, open orders, and plan caps, while enforcing LLM budget/cooldown rules and simple regime filters (chop detection, breakout priority signals). The decision schema supports actions like BUY/SELL, plan updates, partial closes, or pausing trading, using either Gemini or OpenAI. 【F:trader_bot/strategy.py†L102-L167】【F:trader_bot/strategy.py†L200-L520】
- **Risk controls:** `RiskManager` tracks daily loss (absolute and %), order value/min size, gross exposure, distinct position counts, and pending order stacking, trimming quantities when caps are hit. Baseline positions can be ignored in paper mode to avoid double-counting sandbox inventory. 【F:trader_bot/risk_manager.py†L23-L200】

## Honest Take
- The scaffold is solid for a discretionary-style bot: live context feeds the LLM, orders are wrapped in risk/health gates, and telemetry persists rich traces. That said, profitability will hinge on disciplined playbooks and faster information—LLM intuition alone may lag intraday moves without stronger structure and richer data.
- Operational resilience (health circuits, resync, exposure accounting) is in place, but edge awareness could improve: venue microstructure cues, multi-symbol prioritization, and playbook-specific sizing/targets aren’t first-class yet.

## Recommended Next Steps
- [ ] **Richer live data — order book + microstructure:** Extend `data_fetch_coordinator`/`market_data_service` to ingest depth imbalance, queue position hints, and low-latency top-of-book snapshots, then surface these signals in `TradingContext` for LLM prompts.
- [ ] **Richer live data — derivatives/flow:** Pull liquidation prints, funding rates, and volume/volatility bursts so prompts can reason about leverage stress and momentum follow-through.
- [ ] **Live news ingestion:** Integrate a streaming headline source (e.g., RSS firehose or websocket) with lightweight relevance scoring so the LLM can react to market-moving news alongside price/flow.
- [ ] **Plan lifecycle polish:** Add per-playbook trailing/auto-breakeven rules in `PlanMonitor` and better stacking guards in `TradeActionHandler` so plans evolve automatically when volatility shifts.
- [ ] **Budget-aware cadence:** Tighten LLM call strategy (e.g., heavier tool use, cached TA summaries) when burn is high or spreads widen, to preserve budget without missing inflections.
- [ ] **Multi-symbol focus:** Enhance symbol selection/prioritization (volatility/volume scores, correlation buckets) so the bot can rotate attention instead of defaulting to BTC/USD when state is thin.
