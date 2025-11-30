import re

from trader_bot.strategy import LLMStrategy


class Dummy:
    def __getattr__(self, _):
        return None


def test_prompt_budget_trims_memory_and_context(monkeypatch):
    # Small budget to force trimming
    monkeypatch.setenv("LLM_DECISION_BYTE_BUDGET", "200")
    strat = LLMStrategy(Dummy(), Dummy(), Dummy())

    prompt = (
        "HEADER\n"
        "MEMORY (recent plans/decisions):\n"
        + "X" * 300
        + "\n\nCONTEXT:\n"
        + "Y" * 300
        + "\nRULES:\n- do this\n"
    )
    trimmed = strat._enforce_prompt_budget(prompt, budget=200)
    assert len(trimmed.encode("utf-8")) <= 200
    assert "MEMORY: trimmed" in trimmed or "MEMORY" not in trimmed
    # Context should be removed or replaced
    assert "Y" * 10 not in trimmed
    # Rules should remain
    assert "RULES" in trimmed
