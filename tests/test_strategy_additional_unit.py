import json
from collections import Counter
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from trader_bot.llm_tools import ToolName, ToolResponse
from trader_bot.strategy import (
    LLMStrategy,
    MAX_ORDER_VALUE,
    PLAN_MAX_PER_SYMBOL,
    _LLMResponse,
)


class StrategySetupMixin:
    def setUp(self):
        for attr in ("_prompt_template_cache", "_system_prompt_cache"):
            if hasattr(LLMStrategy, attr):
                delattr(LLMStrategy, attr)

        self.db = MagicMock()
        self.db.log_llm_trace.return_value = 42
        self.ta = MagicMock()
        self.cost = MagicMock()
        self.cost.calculate_llm_burn.return_value = {
            "total_llm_cost": 0.0,
            "budget": 1.0,
            "pct_of_budget": 0.0,
            "burn_rate_per_hour": 0.0,
            "remaining_budget": 1.0,
            "hours_to_cap": None,
        }
        self.cost.calculate_llm_cost.return_value = 0.0
        self.strategy = LLMStrategy(self.db, self.ta, self.cost)

    def tearDown(self):
        for attr in ("_prompt_template_cache", "_system_prompt_cache"):
            if hasattr(LLMStrategy, attr):
                delattr(LLMStrategy, attr)


class TestStrategyHelpers(StrategySetupMixin, TestCase):
    def test_priority_signal_respects_context_and_move(self):
        context = SimpleNamespace(current_iso_time="2024-01-01T00:00:00Z")
        # Latest price 105 vs past 99 (~6% move) should trigger break-glass
        self.db.get_recent_market_data.return_value = [
            {"price": 105},
            {"price": 103},
            {"price": 99},
            {"price": 98},
        ]

        triggered = self.strategy._priority_signal(1, "BTC/USD", context)
        _, kwargs = self.db.get_recent_market_data.call_args
        self.assertEqual(kwargs.get("before_timestamp"), context.current_iso_time)
        self.assertTrue(triggered)

        # Insufficient data should not trigger
        self.db.get_recent_market_data.reset_mock()
        self.db.get_recent_market_data.return_value = [{"price": 100}]
        self.assertFalse(self.strategy._priority_signal(1, "BTC/USD"))

    def test_build_timeframe_summary_includes_vol_and_volume(self):
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
        self.db.get_recent_ohlcv.side_effect = (
            lambda session_id, symbol, tf, limit=50: list(bars)
        )

        summary = self.strategy._build_timeframe_summary(1, "BTC/USD")
        self.assertTrue(summary.startswith("Multi-timeframe"))
        self.assertIn("vol", summary)
        self.assertIn("avg vol", summary)
        self.assertIn("1m:", summary)
        self.assertIn("1d:", summary)

    def test_compute_regime_flags_buckets(self):
        one_h = [{"close": 100 + (i * 0.05)} for i in range(10)]
        market_point = {
            "spread_pct": 0.1,
            "bid_size": 2,
            "ask_size": 3,
            "bid": 99,
            "ask": 101,
        }

        flags = self.strategy._compute_regime_flags(
            session_id=1,
            symbol="BTC/USD",
            market_data_point=market_point,
            recent_bars={"1h": one_h},
        )

        self.assertIn("low", flags.get("volatility", ""))
        self.assertTrue(flags.get("trend", "").startswith("up"))
        self.assertIn("ok_spread", flags.get("liquidity", ""))
        self.assertIn("depth", flags)

    def test_prompt_template_cache_single_read(self):
        calls = Counter()

        def fake_read(self):
            calls[self.name] += 1
            return "TEMPLATE {prompt_context_block}"

        for attr in ("_prompt_template_cache", "_system_prompt_cache"):
            if hasattr(LLMStrategy, attr):
                delattr(LLMStrategy, attr)

        with patch("trader_bot.strategy.Path.read_text", fake_read):
            LLMStrategy(self.db, self.ta, self.cost)
            LLMStrategy(self.db, self.ta, self.cost)

        self.assertEqual(calls["llm_prompt_template.txt"], 1)
        # System prompt should also be cached after first read
        self.assertEqual(calls["llm_system_prompt.txt"], 1)

    def test_enforce_prompt_budget_hard_clamp(self):
        prompt = "HEADER\n" + ("A" * 800)
        trimmed = self.strategy._enforce_prompt_budget(prompt, budget=120)
        self.assertLessEqual(len(trimmed.encode("utf-8")), 120)
        self.assertIn("TRIMMED", trimmed)

    def test_parse_tool_requests_coercion_and_errors(self):
        self.assertEqual(self.strategy._parse_tool_requests("not-json"), [])

        payload = json.dumps(
            {
                "tool_requests": [
                    {
                        "id": "r1",
                        "tool": "get_order_book",
                        "params": {
                            "symbol": "BTC/USD",
                            "depth": {"fast": 5, "slow": 10},
                        },
                    },
                    {
                        "id": "r2",
                        "tool": "get_recent_trades",
                        "params": {"symbol": "BTC/USD", "limit": "25"},
                    },
                ]
            }
        )
        requests = self.strategy._parse_tool_requests(payload)
        self.assertEqual(len(requests), 2)
        ob_req = next(r for r in requests if r.tool == ToolName.GET_ORDER_BOOK)
        trades_req = next(r for r in requests if r.tool == ToolName.GET_RECENT_TRADES)
        self.assertEqual(ob_req.params.depth, 10)
        self.assertEqual(trades_req.params.limit, 25)

        bad_payload = json.dumps(
            {"tool_requests": [{"id": "bad", "tool": "invalid", "params": {}}]}
        )
        self.assertEqual(self.strategy._parse_tool_requests(bad_payload), [])

    def test_clamp_quantity_respects_caps(self):
        limited = self.strategy._clamp_quantity(quantity=10, price=200, headroom=1000)
        self.assertEqual(limited, 5.0)
        self.assertEqual(
            self.strategy._clamp_quantity(quantity=1000, price=10, headroom=None),
            MAX_ORDER_VALUE / 10,
        )
        self.assertEqual(
            self.strategy._clamp_quantity(quantity=1, price=0, headroom=1000), 0.0
        )


class TestStrategyLLMFlow(StrategySetupMixin, IsolatedAsyncioTestCase):
    async def test_get_llm_decision_tool_flow_and_constraints(self):
        self.strategy._llm_ready = True
        self.strategy.last_rejection_reason = "recent rejection"
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
        self.strategy.tool_coordinator = tool_coordinator

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

        self.db.get_recent_ohlcv.return_value = [{"close": 100}] * 12
        self.db.get_open_orders.return_value = []
        self.db.get_open_trade_plans.return_value = [{"symbol": "BTC/USD"}] * (
            PLAN_MAX_PER_SYMBOL + 1
        )
        trading_context = SimpleNamespace(
            get_context_summary=lambda symbol, open_orders=None: "ctx",
            get_memory_snapshot=lambda: "mem",
        )

        with patch.object(self.strategy, "_invoke_llm", fake_invoke):
            decision_json, trace_id = await self.strategy._get_llm_decision(
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

        self.assertEqual(decision_json, '{"action":"HOLD","symbol":"BTC/USD","quantity":0,"reason":"cooldown"}')
        self.assertEqual(trace_id, 42)
        tool_coordinator.handle_requests.assert_awaited_once()
        self.assertEqual(len(prompts), 2)
        self.assertIn("Return ONLY a JSON object with this shape", prompts[0])
        self.assertIn("TOOL RESPONSES (JSON)", prompts[1])
        self.assertIn("cooldown active", prompts[1])
        self.assertIn("plan cap reached", prompts[1])

    @patch("trader_bot.strategy.asyncio.get_event_loop")
    async def test_trade_callbacks_update_cooldown(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.db.get_recent_market_data.return_value = [{"price": 100}] * 50
        self.ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}

        self.strategy.on_trade_executed(990)
        self.assertEqual(self.strategy.last_trade_ts, 990)

        with patch.object(self.strategy, "_get_llm_decision", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            signal = await self.strategy.generate_signal(
                1, {"BTC/USD": {"price": 100}}, 1000, 0
            )

        self.assertIsNone(signal)
        self.strategy.on_trade_rejected("too risky")
        self.assertEqual(self.strategy.last_rejection_reason, "too risky")
