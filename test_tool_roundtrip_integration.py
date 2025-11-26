import asyncio
from types import SimpleNamespace

import pytest

from data_fetch_coordinator import DataFetchCoordinator
from llm_tools import ToolName
from strategy import LLMStrategy, StrategySignal


class StubExchange:
    def __init__(self):
        self.ohlcv_calls = 0
        self.ob_calls = 0

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.ohlcv_calls += 1
        return [
            [1, 10, 11, 9, 10.5, 100],
            [2, 10.5, 12, 10, 11, 120],
        ]

    async def fetch_order_book(self, symbol, limit):
        self.ob_calls += 1
        return {"bids": [[99, 1]], "asks": [[101, 2]], "timestamp": 123}

    async def fetch_trades(self, symbol, limit):
        return []


class StubDB:
    def __init__(self):
        self.traces = []
        self.calls = []

    def get_recent_ohlcv(self, session_id, symbol, timeframe, limit=50):
        return [{"close": 10, "volume": 1}, {"close": 11, "volume": 2}]

    def get_recent_market_data(self, session_id, symbol, limit=50, before_timestamp=None):
        return [{"price": 10}]

    def log_llm_call(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def log_llm_trace(self, session_id, prompt, response_text, decision_json=None, market_context=None):
        self.traces.append(
            {
                "session_id": session_id,
                "prompt": prompt,
                "response": response_text,
                "decision": decision_json,
                "context": market_context,
            }
        )
        return len(self.traces)


class StubTA:
    def calculate_indicators(self, recent_data):
        return {"rsi": 50}

    def format_indicators_for_llm(self, indicators, price):
        return "RSI neutral"


class StubCost:
    def calculate_llm_cost(self, *args, **kwargs):
        return 0.0


def _fake_response(text):
    usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=1)
    return SimpleNamespace(text=text, usage_metadata=usage)


@pytest.mark.asyncio
async def test_tool_roundtrip_executes_and_returns_decision(monkeypatch):
    db = StubDB()
    ta = StubTA()
    cost = StubCost()
    coordinator = DataFetchCoordinator(StubExchange())
    strategy = LLMStrategy(db, ta, cost, tool_coordinator=coordinator)

    calls = []

    async def fake_invoke(prompt, timeout=30):
        # First call (planner) returns tool requests; second returns final action
        if not calls:
            calls.append("planner")
            return _fake_response(
                '{"tool_requests":[{"id":"req1","tool":"get_market_data","params":{"symbol":"BTC/USD","timeframes":["1m"],"limit":2}},{"id":"req2","tool":"get_order_book","params":{"symbol":"BTC/USD","depth":1}}]}'
            )
        calls.append("decision")
        return _fake_response('{"action":"HOLD","symbol":"BTC/USD","reason":"tool test"}')

    monkeypatch.setattr(strategy, "_invoke_llm", fake_invoke)

    decision_json, trace_id = await strategy._get_llm_decision(
        session_id=1,
        market_data={"BTC/USD": {"price": 100, "bid": 99, "ask": 101}},
        current_equity=1000.0,
        prompt_context="",
        trading_context=None,
        open_orders=[],
        headroom=5000.0,
        pending_buy_exposure=0.0,
    )

    assert decision_json is not None
    assert '"HOLD"' in decision_json
    assert calls == ["planner", "decision"]
    # Trace should include tool metadata
    assert db.traces
    trace = db.traces[0]
    assert trace["context"]["tool_requests"][0]["tool"] == ToolName.GET_MARKET_DATA
    assert trace["context"]["tool_responses"][1]["tool"] == ToolName.GET_ORDER_BOOK
