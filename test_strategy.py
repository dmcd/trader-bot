import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from strategy import LLMStrategy, StrategySignal

class TestLLMStrategy(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_ta = MagicMock()
        self.mock_cost_tracker = MagicMock()
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            delattr(LLMStrategy, "_prompt_template_cache")
        self.strategy = LLMStrategy(self.mock_db, self.mock_ta, self.mock_cost_tracker)
        
        # Mock the LLM model
        self.strategy.model = MagicMock()
        self.strategy.model.generate_content = MagicMock()

    def test_fees_too_high(self):
        # Mock session stats
        stats_high = {'gross_pnl': 100, 'total_fees': 60}
        # 60/100 = 60% > 50% (default cooldown)
        self.assertTrue(self.strategy._fees_too_high(stats_high))
        
        stats_low = {'gross_pnl': 100, 'total_fees': 10}
        # 10/100 = 10% < 50%
        self.assertFalse(self.strategy._fees_too_high(stats_low))

    def test_chop_filter(self):
        # Mock TA indicators
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 0.5, 'rsi': 52}
        # Tight bands (<1.0) and RSI near 50 (abs(52-50)<5) -> Chop
        self.assertTrue(self.strategy._is_choppy('BTC/USD', {}, [{'price': 100}]*20))
        
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 70}
        self.assertFalse(self.strategy._is_choppy('BTC/USD', {}, [{'price': 100}]*20))

    @patch('strategy.asyncio.get_event_loop')
    async def test_generate_signal_cooldown(self, mock_loop):
        # Setup
        mock_loop.return_value.time.return_value = 1000
        self.strategy.last_trade_ts = 990 # 10s ago
        # MIN_TRADE_INTERVAL is usually 300s
        
        # Mock dependencies to pass checks
        self.mock_db.get_session_stats.return_value = {'gross_pnl': 100, 'total_fees': 0}
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}]*50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50} # Not choppy
        
        # Execute
        signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
        
        # Should return None due to cooldown
        self.assertIsNone(signal)

    @patch('strategy.asyncio.get_event_loop')
    async def test_generate_signal_success(self, mock_loop):
        # Setup
        mock_loop.return_value.time.return_value = 1000
        self.strategy.last_trade_ts = 0 # Long ago
        
        # Mock dependencies
        self.mock_db.get_session_stats.return_value = {'gross_pnl': 100, 'total_fees': 0}
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}]*50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = '{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test"}'
        self.strategy.model.generate_content.return_value = mock_response
        
        # Execute
        # We need to patch asyncio.to_thread or make generate_content async compatible if we were running real async
        # Since we are mocking, we might need to adjust how we call it in test
        # The strategy uses asyncio.to_thread, so we need to mock that or run in async test runner
        
        # For simplicity in this unit test file, we can just mock _get_llm_decision directly
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test"}'
            
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "BUY")
            self.assertEqual(signal.quantity, 0.1)

    def test_prompt_template_loaded_and_rendered(self):
        template_body = "TEMPLATE {asset_class} {available_symbols} {prompt_context_block}"
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            delattr(LLMStrategy, "_prompt_template_cache")
        with patch('strategy.Path.read_text', return_value=template_body):
            strategy = LLMStrategy(self.mock_db, self.mock_ta, self.mock_cost_tracker)

        rendered = strategy._build_prompt(
            asset_class="crypto",
            available_symbols="BTC/USD",
            prompt_context_block="CTX",
        )

        self.assertIn("TEMPLATE crypto BTC/USD CTX", rendered)

if __name__ == '__main__':
    unittest.main()
