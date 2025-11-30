import json
import re
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trader_bot.llm_tools import ToolName, ToolResponse
from trader_bot.strategy import (
    LLMStrategy,
    MAX_ORDER_VALUE,
    PLAN_MAX_PER_SYMBOL,
    _LLMResponse,
)


def _fake_response(text):
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MagicMock(prompt_token_count=1, candidates_token_count=1)
    return resp


def _reset_prompt_cache():
    for attr in ("_prompt_template_cache", "_system_prompt_cache"):
        if hasattr(LLMStrategy, attr):
            delattr(LLMStrategy, attr)


def test_fees_too_high(strategy_env):
    strategy = strategy_env.strategy
    stats_high = {"gross_pnl": 100, "total_fees": 60}
    stats_low = {"gross_pnl": 100, "total_fees": 10}

    assert strategy._fees_too_high(stats_high)
    assert strategy._fees_too_high(stats_low) is False


def test_chop_filter(strategy_env):
    strategy = strategy_env.strategy
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 0.5, "rsi": 52}
    assert strategy._is_choppy("BTC/USD", {}, [{"price": 100}] * 20)

    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 70}
    assert strategy._is_choppy("BTC/USD", {}, [{"price": 100}] * 20) is False


def test_chop_filter_handles_short_series_and_errors(strategy_env):
    strategy = strategy_env.strategy
    assert strategy._is_choppy("BTC/USD", {}, [{"price": 100}] * 5) is False
    strategy_env.ta.calculate_indicators.side_effect = RuntimeError("boom")
    assert strategy._is_choppy("BTC/USD", {}, [{"price": 100}] * 25) is False


@pytest.mark.asyncio
async def test_generate_signal_cooldown(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy.last_trade_ts = 990
    strategy_env.db.get_session_stats.return_value = {"gross_pnl": 100, "total_fees": 0}
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}

    signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert signal is None


@pytest.mark.asyncio
async def test_generate_signal_success(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy.last_trade_ts = 0
    strategy_env.db.get_session_stats.return_value = {"gross_pnl": 100, "total_fees": 0}
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    low_fee_stats = {"gross_pnl": 100, "total_fees": 10}

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = '{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test"}'
        signal = await strategy.generate_signal(
            1, {"BTC/USD": {"price": 100}}, 1000, 0, session_stats=low_fee_stats
        )

    assert signal is not None
    assert signal.action == "BUY"
    assert signal.quantity == 0.1
    assert hasattr(signal, "stop_price")


@pytest.mark.asyncio
async def test_generate_signal_blocks_on_fee_ratio(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy.last_trade_ts = 0
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}

    high_fee_stats = {"gross_pnl": 100, "total_fees": 60}
    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = '{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test"}'
        signal = await strategy.generate_signal(
            1, {"BTC/USD": {"price": 100}}, 1000, 0, session_stats=high_fee_stats
        )
        mock_llm.assert_not_awaited()

    assert signal is None


@pytest.mark.asyncio
async def test_clamps_stops_targets(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy.last_trade_ts = 0
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    low_fee_stats = {"gross_pnl": 100, "total_fees": 10}

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(
            {"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test", "stop_price": 150, "target_price": 300}
        )
        signal = await strategy.generate_signal(
            1, {"BTC/USD": {"price": 100}}, 1000, 0, session_stats=low_fee_stats
        )

    assert signal is not None
    assert signal.stop_price <= 100
    assert signal.target_price <= 102


def test_prompt_template_loaded_and_rendered(strategy_env):
    template_body = "TEMPLATE {asset_class} {available_symbols} {prompt_context_block}"
    for attr in ("_prompt_template_cache", "_system_prompt_cache"):
        if hasattr(LLMStrategy, attr):
            delattr(LLMStrategy, attr)

    with patch("trader_bot.strategy.Path.read_text", return_value=template_body):
        strategy = LLMStrategy(strategy_env.db, strategy_env.ta, strategy_env.cost)

    rendered = strategy._build_prompt(asset_class="crypto", available_symbols="BTC/USD", prompt_context_block="CTX")
    assert "TEMPLATE crypto BTC/USD CTX" in rendered


def test_venue_note_for_ib(monkeypatch, strategy_env):
    _reset_prompt_cache()
    monkeypatch.setattr("trader_bot.strategy.ACTIVE_EXCHANGE", "IB")
    monkeypatch.setattr("trader_bot.strategy.IB_BASE_CURRENCY", "AUD")

    with patch("trader_bot.strategy.Path.read_text", return_value="PROMPT {venue_note}"):
        strategy = LLMStrategy(strategy_env.db, strategy_env.ta, strategy_env.cost)

    note = strategy._venue_note()
    rendered = strategy._build_prompt(venue_note=note)

    assert "Interactive Brokers" in note
    assert "AUD" in note
    assert "Interactive Brokers" in rendered


def test_priority_signal_respects_context_and_move(strategy_env):
    strategy = strategy_env.strategy
    context = SimpleNamespace(current_iso_time="2024-01-01T00:00:00Z")
    strategy_env.db.get_recent_market_data.return_value = [
        {"price": 105},
        {"price": 103},
        {"price": 99},
        {"price": 98},
    ]

    triggered = strategy._priority_signal(1, "BTC/USD", context)
    _, kwargs = strategy_env.db.get_recent_market_data.call_args
    assert kwargs.get("before_timestamp") == context.current_iso_time
    assert triggered

    strategy_env.db.get_recent_market_data.reset_mock()
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}]
    assert strategy._priority_signal(1, "BTC/USD") is False


def test_build_timeframe_summary_includes_vol_and_volume(strategy_env):
    strategy = strategy_env.strategy
    bars = []
    for i in range(6):
        price = 100 + i
        bars.append(
            {
                "timestamp": i,
                "open": price - 0.5,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 10 + i,
            }
        )
    strategy_env.db.get_recent_ohlcv.side_effect = lambda session_id, symbol, tf, limit=50: list(bars)

    summary = strategy._build_timeframe_summary(1, "BTC/USD")

    assert summary.startswith("Multi-timeframe")
    assert "vol" in summary
    assert "avg vol" in summary
    assert "1m:" in summary
    assert "1d:" in summary


def test_compute_regime_flags_buckets(strategy_env):
    strategy = strategy_env.strategy
    one_h = [{"close": 100 + (i * 0.05)} for i in range(10)]
    market_point = {"spread_pct": 0.1, "bid_size": 2, "ask_size": 3, "bid": 99, "ask": 101}

    flags = strategy._compute_regime_flags(
        session_id=1,
        symbol="BTC/USD",
        market_data_point=market_point,
        recent_bars={"1h": one_h},
    )

    assert "low" in flags.get("volatility", "")
    assert flags.get("trend", "").startswith("up")
    assert "ok_spread" in flags.get("liquidity", "")
    assert "depth" in flags


def test_prompt_template_cache_single_read(strategy_env):
    calls = Counter()

    def fake_read(self):
        calls[self.name] += 1
        return "TEMPLATE {prompt_context_block}"

    for attr in ("_prompt_template_cache", "_system_prompt_cache"):
        if hasattr(LLMStrategy, attr):
            delattr(LLMStrategy, attr)

    with patch("trader_bot.strategy.Path.read_text", fake_read):
        LLMStrategy(strategy_env.db, strategy_env.ta, strategy_env.cost)
        LLMStrategy(strategy_env.db, strategy_env.ta, strategy_env.cost)

    assert calls["llm_prompt_template.txt"] == 1
    assert calls["llm_system_prompt.txt"] == 1


def test_enforce_prompt_budget_hard_clamp(strategy_env):
    strategy = strategy_env.strategy
    prompt = "HEADER\n" + ("A" * 800)

    trimmed = strategy._enforce_prompt_budget(prompt, budget=120)

    assert len(trimmed.encode("utf-8")) <= 120
    assert "TRIMMED" in trimmed


def test_parse_tool_requests_coercion_and_errors(strategy_env):
    strategy = strategy_env.strategy
    assert strategy._parse_tool_requests("not-json") == []

    payload = json.dumps(
        {
            "tool_requests": [
                {
                    "id": "r1",
                    "tool": "get_order_book",
                    "params": {"symbol": "BTC/USD", "depth": {"fast": 5, "slow": 10}},
                },
                {
                    "id": "r2",
                    "tool": "get_recent_trades",
                    "params": {"symbol": "BTC/USD", "limit": "25"},
                },
            ]
        }
    )
    requests = strategy._parse_tool_requests(payload)
    assert len(requests) == 2
    ob_req = next(r for r in requests if r.tool == ToolName.GET_ORDER_BOOK)
    trades_req = next(r for r in requests if r.tool == ToolName.GET_RECENT_TRADES)
    assert ob_req.params.depth == 10
    assert trades_req.params.limit == 25

    bad_payload = json.dumps({"tool_requests": [{"id": "bad", "tool": "invalid", "params": {}}]})
    assert strategy._parse_tool_requests(bad_payload) == []


def test_clamp_quantity_respects_caps(strategy_env):
    strategy = strategy_env.strategy
    assert strategy._clamp_quantity(quantity=10, price=200, headroom=1000) == 5.0
    assert strategy._clamp_quantity(quantity=1000, price=10, headroom=None) == MAX_ORDER_VALUE / 10
    assert strategy._clamp_quantity(quantity=1, price=0, headroom=1000) == 0.0


@pytest.mark.asyncio
async def test_get_llm_decision_tool_flow_and_constraints(strategy_env):
    strategy = strategy_env.strategy
    strategy._llm_ready = True
    strategy.last_rejection_reason = "recent rejection"
    tool_coordinator = SimpleNamespace()
    tool_coordinator.handle_requests = AsyncMock(
        return_value=[
            ToolResponse(
                id="r1",
                tool=ToolName.GET_RECENT_TRADES,
                data={"trades": []},
            )
        ]
    )
    strategy.tool_coordinator = tool_coordinator

    prompts: list[str] = []
    responses = [
        _LLMResponse(
            '{"tool_requests":[{"id":"r1","tool":"get_recent_trades","params":{"symbol":"BTC/USD","limit":"5"}}]}',
            prompt_tokens=2,
            completion_tokens=2,
        ),
        _LLMResponse(
            '{"action":"HOLD","symbol":"BTC/USD","quantity":0,"reason":"cooldown"}',
            prompt_tokens=3,
            completion_tokens=1,
        ),
    ]

    async def fake_invoke(prompt, timeout=30):
        prompts.append(prompt)
        return responses.pop(0)

    strategy_env.db.get_recent_ohlcv.return_value = [{"close": 100}] * 12
    strategy_env.db.get_open_orders.return_value = []
    strategy_env.db.get_open_trade_plans.return_value = [{"symbol": "BTC/USD"}] * (PLAN_MAX_PER_SYMBOL + 1)
    trading_context = SimpleNamespace(
        get_context_summary=lambda symbol, open_orders=None: "ctx",
        get_memory_snapshot=lambda: "mem",
    )

    with patch.object(strategy, "_invoke_llm", fake_invoke):
        decision_json, trace_id = await strategy._get_llm_decision(
            session_id=1,
            market_data={
                "BTC/USD": {
                    "price": 100,
                    "spread_pct": 0.1,
                    "bid": 99,
                    "ask": 101,
                    "bid_size": 1,
                    "ask_size": 1,
                }
            },
            current_equity=1000,
            prompt_context="CTX",
            trading_context=trading_context,
            open_orders=[],
            headroom=50.0,
            pending_buy_exposure=0.0,
            can_trade=False,
            spacing_flag="cooldown 10s",
            plan_counts={"BTC/USD": PLAN_MAX_PER_SYMBOL},
            burn_stats={
                "total_llm_cost": 0.1,
                "budget": 1.0,
                "pct_of_budget": 0.1,
                "burn_rate_per_hour": 0.01,
                "remaining_budget": 0.9,
                "hours_to_cap": 10,
            },
        )

    assert decision_json == '{"action":"HOLD","symbol":"BTC/USD","quantity":0,"reason":"cooldown"}'
    assert trace_id == 42
    tool_coordinator.handle_requests.assert_awaited_once()
    assert len(prompts) == 2
    assert "Return ONLY a JSON object with this shape" in prompts[0]
    assert "TOOL RESPONSES (JSON)" in prompts[1]
    assert "cooldown active" in prompts[1]
    assert "plan cap reached" in prompts[1]


@pytest.mark.asyncio
async def test_trade_callbacks_update_cooldown(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}

    strategy.on_trade_executed(990)
    assert strategy.last_trade_ts == 990

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert signal is None
    strategy.on_trade_rejected("too risky")
    assert strategy.last_rejection_reason == "too risky"


@pytest.mark.asyncio
async def test_schema_validation_failure(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0
    bad_json = '{"action": "BUY"}'

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = bad_json
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert signal is None


@pytest.mark.asyncio
async def test_stop_target_clamping(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0
    raw_decision = json.dumps(
        {
            "action": "BUY",
            "symbol": "BTC/USD",
            "quantity": 0.1,
            "reason": "test",
            "stop_price": 150,
            "target_price": 300,
        }
    )

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = raw_decision
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert signal is not None
    assert signal.stop_price <= 100
    assert signal.target_price <= 102


@pytest.mark.asyncio
async def test_null_quantity_validation(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0
    raw_decision = json.dumps(
        {"action": "HOLD", "symbol": "BTC/USD", "quantity": None, "reason": "test null quantity"}
    )

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = raw_decision
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert signal is not None
    assert signal.action == "HOLD"
    assert signal.quantity == 0


@pytest.mark.asyncio
async def test_null_symbol_validation(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0
    raw_decision = json.dumps(
        {
            "action": "PAUSE_TRADING",
            "symbol": None,
            "quantity": 0,
            "reason": "Market closed",
            "duration_minutes": 60,
        }
    )

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = raw_decision
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert signal is not None
    assert signal.action == "PAUSE_TRADING"
    assert signal.symbol is None


@pytest.mark.asyncio
async def test_generate_signal_plan_management_actions(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1500)
    strategy.last_trade_ts = 0
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    update_decision = json.dumps(
        {
            "action": "UPDATE_PLAN",
            "symbol": "BTC/USD",
            "quantity": 0,
            "reason": "tighten",
            "plan_id": 5,
            "stop_price": 95,
            "target_price": 105,
            "size_factor": 0.8,
        }
    )
    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (update_decision, 99)
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
    assert signal.action == "UPDATE_PLAN"
    assert signal.plan_id == 5
    assert signal.stop_price == 95
    assert signal.target_price == 105
    assert signal.trace_id == 99
    assert signal.size_factor == 0.8


@pytest.mark.asyncio
async def test_generate_signal_partial_and_close_actions(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1500)
    strategy.last_trade_ts = 0
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    partial_json = json.dumps(
        {
            "action": "PARTIAL_CLOSE",
            "symbol": "BTC/USD",
            "quantity": 0,
            "reason": "trim",
            "plan_id": 7,
            "close_fraction": 0.25,
        }
    )
    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (partial_json, 101)
        partial_signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
    assert partial_signal.action == "PARTIAL_CLOSE"
    assert partial_signal.plan_id == 7
    assert partial_signal.close_fraction == 0.25
    assert partial_signal.trace_id == 101

    set_loop_time(1510)
    close_json = json.dumps(
        {"action": "CLOSE_POSITION", "symbol": "BTC/USD", "quantity": 0, "reason": "exit now"}
    )
    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (close_json, 202)
        close_signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
    assert close_signal.action == "CLOSE_POSITION"
    assert close_signal.symbol == "BTC/USD"
    assert close_signal.trace_id == 202


@pytest.mark.asyncio
async def test_break_glass_allows_trade_during_cooldown(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(4000)
    strategy.last_trade_ts = 3985
    strategy._last_break_glass = 0
    strategy._priority_signal = MagicMock(return_value=True)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    buy_decision = json.dumps(
        {"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "break glass", "stop_price": 98, "target_price": 102}
    )
    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (buy_decision, 303)
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
    assert signal.action == "BUY"
    assert strategy._last_break_glass == 4000
    assert signal.trace_id == 303


@pytest.mark.asyncio
async def test_generate_signal_returns_none_when_llm_not_ready(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(2000)
    strategy._llm_ready = False
    strategy.last_trade_ts = 0
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}

    signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
    assert signal is None


def test_extract_json_payload_handles_chatter(strategy_env):
    noisy = (
        "Some analysis text that should be ignored before the JSON.\n\n"
        "```json\n"
        "{\n"
        '  "action": "HOLD",\n'
        '  "symbol": "BTC/USD",\n'
        '  "quantity": 0.0,\n'
        '  "reason": "Conflicting indicators"\n'
        "}\n"
        "```"
    )

    payload = strategy_env.strategy._extract_json_payload(noisy)
    decision = json.loads(payload)
    assert decision["action"] == "HOLD"


def test_extract_json_payload_picks_first_valid_object(strategy_env):
    text = "prelude {\"action\":\"HOLD\"}{\"action\":\"BUY\"}"
    payload = strategy_env.strategy._extract_json_payload(text)
    assert json.loads(payload)["action"] == "HOLD"


def test_extract_json_payload_handles_incomplete_fence(strategy_env):
    text = "```json\n{\"action\":\"SELL\",\"reason\":\"done\"}\n```\ntrailing } noise"
    payload = strategy_env.strategy._extract_json_payload(text)
    assert json.loads(payload)["action"] == "SELL"


def test_tool_request_filtering_and_alias(strategy_env):
    strategy = strategy_env.strategy
    payload = json.dumps(
        {
            "tool_requests": [
                {
                    "id": "req1",
                    "tool": "get_recent_trades",
                    "params": {"symbol": "BTC/USD", "limit": 10, "extra_field": "should_be_removed"},
                }
            ]
        }
    )
    requests = strategy._parse_tool_requests(payload)
    assert len(requests) == 1
    assert requests[0].tool == "get_recent_trades"
    assert requests[0].params.limit == 10

    payload_alias = json.dumps(
        {"tool_requests": [{"id": "req2", "tool": "get_market_data", "params": {"symbol": "BTC/USD", "timeframes": ["4h", "1m"]}}]}
    )
    requests_alias = strategy._parse_tool_requests(payload_alias)
    assert len(requests_alias) == 1
    assert "6h" in requests_alias[0].params.timeframes
    assert "1m" in requests_alias[0].params.timeframes
    assert "4h" not in requests_alias[0].params.timeframes


@pytest.mark.asyncio
async def test_prompt_includes_plan_cap_note(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.db.get_open_orders.return_value = []
    strategy_env.db.get_open_trade_plans.return_value = [{"symbol": "BTC/USD"} for _ in range(3)]
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
        args, kwargs = mock_llm.call_args
        prompt_context = args[3] if len(args) >= 4 else kwargs.get("prompt_context", "")

    assert "Plan cap reached" in prompt_context


@pytest.mark.asyncio
async def test_rejection_reason_surfaces_in_prompt(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.db.get_open_orders.return_value = []
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0
    strategy.last_rejection_reason = "Plan cap reached for BTC/USD (2/2)"

    captured_prompts = []

    async def fake_invoke(prompt, timeout=30):
        captured_prompts.append(prompt)
        return _fake_response('{"action":"HOLD","symbol":"BTC/USD","reason":"test"}')

    with patch.object(strategy, "_invoke_llm", fake_invoke):
        await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    joined = "\n".join(captured_prompts)
    assert "previous order was REJECTED" in joined
    assert "Plan cap reached for BTC/USD" in joined


@pytest.mark.asyncio
async def test_prompt_includes_llm_burn_note(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0

    burn_snapshot = {
        "total_llm_cost": 2.0,
        "budget": 10.0,
        "pct_of_budget": 0.2,
        "burn_rate_per_hour": 1.0,
        "remaining_budget": 8.0,
        "hours_to_cap": 8.0,
    }
    strategy_env.cost.calculate_llm_burn.return_value = burn_snapshot

    captured_prompts = []

    async def fake_invoke(prompt, timeout=30):
        captured_prompts.append(prompt)
        return _fake_response('{"action":"HOLD","symbol":"BTC/USD","reason":"burn"}')

    with patch.object(strategy, "_invoke_llm", fake_invoke):
        await strategy.generate_signal(
            1,
            {"BTC/USD": {"price": 100}},
            1000,
            0,
            session_stats={"total_llm_cost": 2.0, "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()},
        )

    combined = "\n".join(captured_prompts)
    assert "LLM spend $2.0000/$10.00" in combined


@pytest.mark.asyncio
async def test_llm_cost_guard_blocks_when_cap_hit(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    session_stats = {"total_llm_cost": 999.0}

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0, session_stats=session_stats)

    assert signal is not None
    assert signal.action == "HOLD"


@pytest.mark.asyncio
async def test_llm_call_throttle_blocks_when_interval_short(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1002)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy._last_llm_call_ts = 1000

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)
        mock_llm.assert_not_called()

    assert signal is not None
    assert signal.action == "HOLD"


@pytest.mark.asyncio
async def test_consecutive_llm_errors_force_hold(strategy_env, set_loop_time):
    strategy = strategy_env.strategy
    set_loop_time(1000)
    strategy_env.db.get_recent_market_data.return_value = [{"price": 100}] * 50
    strategy_env.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    strategy.last_trade_ts = 0

    with patch.object(strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (None, None)
        hold_signal = None
        for _ in range(4):
            hold_signal = await strategy.generate_signal(1, {"BTC/USD": {"price": 100}}, 1000, 0)

    assert hold_signal is not None
    assert hold_signal.action == "HOLD"


def test_update_plan_schema_valid(strategy_env):
    decision = json.dumps({
        "action": "UPDATE_PLAN",
        "symbol": "BTC/USD",
        "quantity": 0,
        "reason": "tighten stop",
        "plan_id": 1,
        "stop_price": 100,
        "target_price": 110,
        "size_factor": 0.5
    })
    decision_obj = json.loads(decision)
    strategy_env.strategy._decision_schema  # touch to ensure present
    assert decision_obj["action"] == "UPDATE_PLAN"


def test_partial_close_schema_valid(strategy_env):
    decision = json.dumps({
        "action": "PARTIAL_CLOSE",
        "symbol": "BTC/USD",
        "quantity": 0,
        "reason": "trim",
        "plan_id": 2,
        "close_fraction": 0.5
    })
    decision_obj = json.loads(decision)
    assert decision_obj["action"] == "PARTIAL_CLOSE"


def test_close_position_and_pause_schema_valid(strategy_env):
    close_decision = json.dumps({
        "action": "CLOSE_POSITION",
        "symbol": "BTC/USD",
        "quantity": 0,
        "reason": "exit"
    })
    pause_decision = json.dumps({
        "action": "PAUSE_TRADING",
        "symbol": "BTC/USD",
        "quantity": 0,
        "reason": "cooldown",
        "duration_minutes": 10
    })
    assert json.loads(close_decision)["action"] == "CLOSE_POSITION"
    assert json.loads(pause_decision)["action"] == "PAUSE_TRADING"


@pytest.mark.usefixtures("test_db_path")
def test_compute_regime_flags(strategy_env):
    strategy = strategy_env.strategy
    one_h_bars = [{"close": 100 - i * 5} for i in range(12)]
    market_point = {"spread_pct": 0.1, "bid_size": 2, "ask_size": 3, "bid": 99, "ask": 101}
    flags = strategy._compute_regime_flags(
        session_id=1,
        symbol="BTC/USD",
        market_data_point=market_point,
        recent_bars={"1h": one_h_bars},
    )
    assert {"volatility", "trend", "liquidity", "depth"} <= set(flags)


@pytest.mark.usefixtures("test_db_path")
def test_build_timeframe_summary(strategy_env):
    sample = [
        {"timestamp": 0, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
        {"timestamp": 1, "open": 2, "high": 3, "low": 2, "close": 3, "volume": 20},
    ]
    strategy_env.db.get_recent_ohlcv.side_effect = lambda s_id, sym, tf, limit=50: sample
    summary = strategy_env.strategy._build_timeframe_summary(1, "BTC/USD")
    assert "1m:" in summary
    assert "1d:" in summary


class Dummy:
    def __getattr__(self, _):
        return None


def test_prompt_budget_trims_memory_and_context(monkeypatch):
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
    assert "Y" * 10 not in trimmed
    assert "RULES" in trimmed


@pytest.mark.asyncio
async def test_openai_provider_invocation_and_usage_logging(strategy_env):
    mock_db = MagicMock()
    mock_cost = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"action":"HOLD","symbol":"BTC/USD","reason":"test"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    with patch("trader_bot.strategy.LLM_PROVIDER", "OPENAI"), patch("trader_bot.strategy.OPENAI_API_KEY", "sk-test"):
        client = MagicMock()
        with patch("trader_bot.strategy.OpenAI", return_value=client):
            client.chat.completions.create.return_value = mock_response
            strategy = LLMStrategy(mock_db, strategy_env.ta, mock_cost)

    resp = await strategy._invoke_llm("prompt")
    assert resp.text == mock_choice.message.content

    strategy._log_llm_usage(session_id=1, response=resp, response_text=resp.text)
    mock_cost.calculate_llm_cost.assert_called_with(10, 5)
    mock_db.log_llm_call.assert_called_once()


def test_openai_provider_requires_key(monkeypatch):
    _reset_prompt_cache()
    monkeypatch.setattr("trader_bot.strategy.LLM_PROVIDER", "OPENAI")
    monkeypatch.setattr("trader_bot.strategy.OPENAI_API_KEY", "")
    monkeypatch.setattr("trader_bot.strategy.GEMINI_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ALLOW_UNKEYED_LLM", "false")
    with patch("trader_bot.strategy.Path.read_text", return_value="TEMPLATE"):
        strat = LLMStrategy(MagicMock(), MagicMock(), MagicMock())
    assert strat.llm_provider == "OPENAI"
    assert strat._llm_ready is False
    assert strat._openai_client is None


def test_gemini_provider_allows_unkeyed_test_mode(monkeypatch):
    _reset_prompt_cache()
    monkeypatch.setattr("trader_bot.strategy.LLM_PROVIDER", "GEMINI")
    monkeypatch.setattr("trader_bot.strategy.OPENAI_API_KEY", "")
    monkeypatch.setattr("trader_bot.strategy.GEMINI_API_KEY", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ALLOW_UNKEYED_LLM", "true")
    with patch("trader_bot.strategy.Path.read_text", return_value="TEMPLATE"):
        strat = LLMStrategy(MagicMock(), MagicMock(), MagicMock())
    assert strat.llm_provider == "GEMINI"
    assert strat._llm_ready is True
    assert strat.model is None
