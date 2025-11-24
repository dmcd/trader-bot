import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from strategy import LLMStrategy


class TestLLMValidation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_ta = MagicMock()
        self.mock_cost_tracker = MagicMock()
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            delattr(LLMStrategy, "_prompt_template_cache")
        self.strategy = LLMStrategy(self.mock_db, self.mock_ta, self.mock_cost_tracker)
        self.strategy.model = MagicMock()

    @patch('strategy.asyncio.get_event_loop')
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

    @patch('strategy.asyncio.get_event_loop')
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


if __name__ == '__main__':
    unittest.main()
