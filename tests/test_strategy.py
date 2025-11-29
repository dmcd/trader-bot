import asyncio
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from trader_bot.strategy import LLMStrategy, StrategySignal

class TestLLMStrategy(IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_ta = MagicMock()
        self.mock_cost_tracker = MagicMock()
        self.mock_cost_tracker.calculate_llm_burn.return_value = {
            "total_llm_cost": 0.0,
            "budget": 0.0,
            "pct_of_budget": 0.0,
            "burn_rate_per_hour": 0.0,
            "remaining_budget": 0.0,
            "hours_to_cap": None,
        }
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

    @patch('trader_bot.strategy.asyncio.get_event_loop')
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

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_generate_signal_success(self, mock_loop):
        # Setup
        mock_loop.return_value.time.return_value = 1000
        self.strategy.last_trade_ts = 0 # Long ago
        
        # Mock dependencies
        self.mock_db.get_session_stats.return_value = {'gross_pnl': 100, 'total_fees': 0}
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}]*50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        low_fee_stats = {'gross_pnl': 100, 'total_fees': 10}
        
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
            
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0, session_stats=low_fee_stats)
            
            self.assertIsNotNone(signal)
            self.assertEqual(signal.action, "BUY")
            self.assertEqual(signal.quantity, 0.1)
            self.assertTrue(hasattr(signal, "stop_price"))

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_generate_signal_blocks_on_fee_ratio(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.strategy.last_trade_ts = 0

        self.mock_db.get_recent_market_data.return_value = [{'price': 100}]*50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}

        # High fee ratio should skip trading
        high_fee_stats = {'gross_pnl': 100, 'total_fees': 60}
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test"}'
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0, session_stats=high_fee_stats)
            self.assertIsNone(signal)

    @patch('trader_bot.strategy.asyncio.get_event_loop')
    async def test_clamps_stops_targets(self, mock_loop):
        mock_loop.return_value.time.return_value = 1000
        self.strategy.last_trade_ts = 0
        self.mock_db.get_recent_market_data.return_value = [{'price': 100}]*50
        self.mock_ta.calculate_indicators.return_value = {'bb_width': 2.0, 'rsi': 50}
        low_fee_stats = {'gross_pnl': 100, 'total_fees': 10}
        # Stop above price, target far away should be clamped
        with patch.object(self.strategy, '_get_llm_decision', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.1, "reason": "Test", "stop_price": 150, "target_price": 300}'
            signal = await self.strategy.generate_signal(1, {'BTC/USD': {'price': 100}}, 1000, 0, session_stats=low_fee_stats)
            self.assertIsNotNone(signal)
            self.assertLessEqual(signal.stop_price, 100)
            self.assertLessEqual(signal.target_price, 102)  # within 2%

    def test_prompt_template_loaded_and_rendered(self):
        template_body = "TEMPLATE {asset_class} {available_symbols} {prompt_context_block}"
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            delattr(LLMStrategy, "_prompt_template_cache")
        with patch('trader_bot.strategy.Path.read_text', return_value=template_body):
            strategy = LLMStrategy(self.mock_db, self.mock_ta, self.mock_cost_tracker)

        rendered = strategy._build_prompt(
            asset_class="crypto",
            available_symbols="BTC/USD",
            prompt_context_block="CTX",
        )

        self.assertIn("TEMPLATE crypto BTC/USD CTX", rendered)

if __name__ == '__main__':
    unittest.main()
