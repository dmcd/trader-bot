# Component Docs

This folder provides focused references for the main bot components. Each file covers responsibilities, key flows, and integration points.

- `strategy_runner.md` – orchestration loop, order/plan lifecycle, telemetry.
- `strategy.md` – LLM strategy inputs/outputs and signal contract.
- `risk_manager.md` – exposure guards, spacing, stop conditions.
- `trading_context.md` – state fed to the LLM (positions, prices, sentiment knobs).
- `gemini_trader.md` – exchange adapter, order routing, client order IDs.
- `cost_tracker.md` – fee and LLM token accounting.
- `database.md` – schema highlights and persistence flows.
- `technical_analysis.md` – indicators provided to the strategy.
- `dashboard.md` – Streamlit UI, what each panel shows, filtering rules.

Configuration knobs that affect multiple components live in `config.py`; see in-file comments for defaults and env overrides.
