import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone

from strategy_runner import StrategyRunner


class TestTradePlanMonitor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Stub out bot and db with mocks
        self.runner = StrategyRunner()
        self.runner.bot = MagicMock()
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 1.0
        self.runner._apply_fill_to_session_stats = MagicMock()
        self.runner.session_id = 1

    @patch('strategy_runner.datetime')
    async def test_plan_age_flatten(self, mock_dt):
        now = datetime.now(timezone.utc)
        mock_dt.now.return_value = now
        mock_dt.fromisoformat.side_effect = datetime.fromisoformat
        mock_dt.timezone = timezone

        # Plan opened 2 hours ago, age limit 60 min
        self.runner.max_plan_age_minutes = 60
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 1,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'stop_price': None,
            'target_price': None,
            'size': 0.1,
            'opened_at': (now - timedelta(minutes=120)).isoformat()
        }]
        # Simulate price_now
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db.update_trade_plan_status = MagicMock()
        self.runner.db.log_trade = MagicMock()

        await self.runner._monitor_trade_plans(price_now=100)
        self.runner.db.update_trade_plan_status.assert_called_once()


if __name__ == '__main__':
    unittest.main()
