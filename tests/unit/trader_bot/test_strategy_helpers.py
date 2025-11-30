import os
from unittest.mock import MagicMock

import pytest

from trader_bot.strategy import LLMStrategy


class DummyTA:
    def __init__(self, indicators):
        self.indicators = indicators

    def calculate_indicators(self, data):
        return self.indicators


def make_strategy(monkeypatch, provider="GEMINI", openai_key=""):
    monkeypatch.setattr("trader_bot.strategy.LLM_PROVIDER", provider)
    monkeypatch.setattr("trader_bot.strategy.OPENAI_API_KEY", openai_key)
    monkeypatch.setattr("trader_bot.strategy.GEMINI_API_KEY", "")
    monkeypatch.setattr(LLMStrategy, "_load_system_prompt", lambda self: "SYS")
    monkeypatch.setattr(LLMStrategy, "_load_prompt_template", lambda self: "PROMPT {body}")
    return LLMStrategy(
        db=MagicMock(),
        technical_analysis=DummyTA({"bb_width": 0.5, "rsi": 52}),
        cost_tracker=MagicMock(),
    )


def test_extract_json_payload_handles_fences_and_malformed(monkeypatch):
    strategy = make_strategy(monkeypatch)
    fenced = "before ```json {\"a\": 1, \"b\": 2} ``` after"
    assert strategy._extract_json_payload(fenced) == '{"a": 1, "b": 2}'

    multi = "noise {\"ok\": true} trailing }"
    assert strategy._extract_json_payload(multi) == '{"ok": true}'


def test_is_choppy_flags_tight_range(monkeypatch):
    strategy = make_strategy(monkeypatch)
    recent = list(range(25))
    assert strategy._is_choppy("BTC/USD", market_data_point=None, recent_data=recent) is True


def test_openai_provider_requires_key(monkeypatch):
    strategy = make_strategy(monkeypatch, provider="OPENAI", openai_key="")
    assert strategy._llm_ready is False
    assert strategy._openai_client is None


def test_openai_provider_uses_client_when_key_present(monkeypatch):
    created = []

    class DummyOpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr("trader_bot.strategy.OpenAI", DummyOpenAI)
    strategy = make_strategy(monkeypatch, provider="OPENAI", openai_key="sk-test")

    assert strategy._llm_ready is True
    assert isinstance(strategy._openai_client, DummyOpenAI)
    assert created and created[0]["api_key"] == "sk-test"
