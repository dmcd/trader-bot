import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from trader_bot.strategy import LLMStrategy


class TestLLMValidation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_ta = MagicMock()
        self.mock_cost_tracker = MagicMock()
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            delattr(LLMStrategy, "_prompt_template_cache")
        self.strategy = LLMStrategy(self.mock_db, self.mock_ta, self.mock_cost_tracker)
        self.strategy.model = MagicMock()

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_schema_validation_failure(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0
        bad_json = '{"action": "BUY"}'  # missing required fields
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = bad_json
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            self.assertIsNone(signal)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_stop_target_clamping(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0
        raw_decision = json.dumps({
            "action": "BUY",
            "symbol": "BTC/USD",
            "quantity": 0.1,
            "reason": "test",
            "stop_price": 150,
            "target_price": 300
        })
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = raw_decision
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            self.assertIsNotNone(signal)
            self.assertLessEqual(signal.stop_price, 100)
            self.assertLessEqual(signal.target_price, 102)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_null_quantity_validation(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0
        
        # JSON with null quantity
        raw_decision = json.dumps({
            "action": "HOLD",
            "symbol": "BTC/USD",
            "quantity": None,
            "reason": "test null quantity"
        })
        
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = raw_decision
            # This should not raise a schema validation error
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, 'HOLD')
            self.assertEqual(signal.quantity, 0)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_null_symbol_validation(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0
        
        # JSON with null symbol
        raw_decision = json.dumps({
            "action": "PAUSE_TRADING",
            "symbol": None,
            "quantity": 0,
            "reason": "Market closed",
            "duration_minutes": 60
        })
        
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = raw_decision
            # This should not raise a schema validation error
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, 'PAUSE_TRADING')
            self.assertIsNone(signal.symbol)

    def test_extract_json_payload_handles_chatter(self):
        noisy = (
            "Some analysis text that should be ignored before the JSON.\n\n"
            "```json\n"
            "{\n"
            "  \"action\": \"HOLD\",\n"
            "  \"symbol\": \"BTC/USD\",\n"
            "  \"quantity\": 0.0,\n"
            "  \"reason\": \"Conflicting indicators\"\n"
            "}\n"
            "```"
        )

        payload = self.strategy._extract_json_payload(noisy)
        decision = json.loads(payload)
        self.assertEqual(decision['action'], 'HOLD')

    def test_tool_request_filtering_and_alias(self):
        # Test 1: Enum lookup and extra param filtering
        # "get_recent_trades" string should resolve to ToolName.GET_RECENT_TRADES
        # and "extra_field" should be filtered out
        payload = json.dumps({
            "tool_requests": [
                {
                    "id": "req1",
                    "tool": "get_recent_trades",
                    "params": {
                        "symbol": "BTC/USD",
                        "limit": 10,
                        "extra_field": "should_be_removed"
                    }
                }
            ]
        })
        requests = self.strategy._parse_tool_requests(payload)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].tool, "get_recent_trades")
        # Check that extra_field is NOT in the model dump (it's filtered before validation)
        # Wait, ToolRequest.params is a model. We can check if it has the field.
        # The model RecentTradesParams does not have extra_field.
        # If filtering failed, validation would have raised an error and request would be skipped.
        # So existence of request implies success.
        self.assertEqual(requests[0].params.limit, 10)

        # Test 2: 4h alias mapping
        payload_alias = json.dumps({
            "tool_requests": [
                {
                    "id": "req2",
                    "tool": "get_market_data",
                    "params": {
                        "symbol": "BTC/USD",
                        "timeframes": ["4h", "1m"]
                    }
                }
            ]
        })
        requests_alias = self.strategy._parse_tool_requests(payload_alias)
        self.assertEqual(len(requests_alias), 1)
        # "4h" should be mapped to "6h" (as per our edit to llm_tools.py)
        self.assertIn("6h", requests_alias[0].params.timeframes)
        self.assertIn("1m", requests_alias[0].params.timeframes)
        self.assertNotIn("4h", requests_alias[0].params.timeframes)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_prompt_includes_plan_cap_note(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_db.get_open_orders.return_value = []
        self.mock_db.get_open_trade_plans.return_value = [
            {"symbol": "BTC/USD"}, {"symbol": "BTC/USD"}, {"symbol": "BTC/USD"}
        ]
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0

        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            args, kwargs = mock_llm.call_args
            prompt_context = args[3] if len(args) >= 4 else kwargs.get('prompt_context', '')
            self.assertIn("Plan cap reached", prompt_context)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_rejection_reason_surfaces_in_prompt(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_db.get_open_orders.return_value = []
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0
        self.strategy.last_rejection_reason = "Plan cap reached for BTC/USD (2/2)"

        captured_prompts = []

        async def fake_invoke(prompt, timeout=30):
            captured_prompts.append(prompt)
            return _fake_response('{"action":"HOLD","symbol":"BTC/USD","reason":"test"}')

        with patch.object(self.strategy, "_invoke_llm", fake_invoke):
            await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)

        self.assertTrue(captured_prompts)
        joined = "\n".join(captured_prompts)
        self.assertIn("previous order was REJECTED", joined)
        self.assertIn("Plan cap reached for BTC/USD", joined)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_llm_cost_guard_blocks_when_cap_hit(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        session_stats = {'total_llm_cost': 999.0}  # above default cap
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0, session_stats=session_stats)
            mock_llm.assert_not_called()
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, 'HOLD')

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_llm_call_throttle_blocks_when_interval_short(self, mock_loop):
        mock_loop.return_value.time.return_value = 1002
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy._last_llm_call_ts = 1000  # 2s ago, under default 5s interval
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            mock_llm.assert_not_called()
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, 'HOLD')

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_consecutive_llm_errors_force_hold(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}] * 50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        self.strategy.last_trade_ts = 0
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (None, None)
            hold_signal = None
            for _ in range(4):
                hold_signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            self.assertIsNotNone(hold_signal)
            self.assertEqual(hold_signal.action, 'HOLD')

    @patch('trader_bot.strategy.OpenAI')
    async def test_openai_provider_invocation_and_usage_logging(self, mock_openai):
        mock_db = MagicMock()
        mock_cost = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"action":"HOLD","symbol":"BTC/USD","reason":"test"}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        with patch('trader_bot.strategy.LLM_PROVIDER', 'OPENAI'), \
             patch('trader_bot.strategy.OPENAI_API_KEY', 'sk-test'):
            client = MagicMock()
            mock_openai.return_value = client
            client.chat.completions.create.return_value = mock_response
            strategy = LLMStrategy(mock_db, self.mock_ta, mock_cost)

        resp = await strategy._invoke_llm("prompt")
        self.assertEqual(resp.text, mock_choice.message.content)

        strategy._log_llm_usage(session_id=1, response=resp, response_text=resp.text)
        mock_cost.calculate_llm_cost.assert_called_with(10, 5)
        mock_db.log_llm_call.assert_called_once()


if __name__ == '__main__':
    unittest.main()
