from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trader_bot.strategy import LLMStrategy


@pytest.fixture
def strategy_env():
    for attr in ("_prompt_template_cache", "_system_prompt_cache"):
        if hasattr(LLMStrategy, attr):
            delattr(LLMStrategy, attr)

    db = MagicMock()
    db.log_llm_trace.return_value = 42
    ta = MagicMock()
    cost = MagicMock()
    cost.calculate_llm_burn.return_value = {
        "total_llm_cost": 0.0,
        "budget": 1.0,
        "pct_of_budget": 0.0,
        "burn_rate_per_hour": 0.0,
        "remaining_budget": 1.0,
        "hours_to_cap": None,
    }
    cost.calculate_llm_cost.return_value = 0.0
    strategy = LLMStrategy(db, ta, cost)
    strategy.model = MagicMock()

    yield SimpleNamespace(strategy=strategy, db=db, ta=ta, cost=cost)

    for attr in ("_prompt_template_cache", "_system_prompt_cache"):
        if hasattr(LLMStrategy, attr):
            delattr(LLMStrategy, attr)


@pytest.fixture
def set_loop_time(monkeypatch):
    def _set(ts: float):
        loop = MagicMock()
        loop.time.return_value = ts
        monkeypatch.setattr("trader_bot.strategy.asyncio.get_event_loop", lambda: loop)
        return loop

    return _set
