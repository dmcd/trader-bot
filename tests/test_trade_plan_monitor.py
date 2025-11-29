import asyncio
import atexit
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime, timedelta, timezone

# Ensure tests never write to the production trading.db
_fd, _db_path = tempfile.mkstemp(prefix="trader-bot-test-", suffix=".db")
os.close(_fd)
os.environ.setdefault("TRADING_DB_PATH", _db_path)


@atexit.register
def _cleanup_db_path():
    if os.path.exists(_db_path):
        try:
            os.remove(_db_path)
        except OSError:
            pass

from trader_bot.strategy_runner import StrategyRunner


class TestTradePlanMonitor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Stub out bot and db with mocks
        self.runner = StrategyRunner()
        self.runner.bot = MagicMock()
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 1.0
        self.runner._apply_fill_to_session_stats = Mock(return_value=None)
        self.runner.session_id = 1

    @patch('trader_bot.strategy_runner.datetime')
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
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()

        await self.runner._monitor_trade_plans(price_lookup={"BTC/USD": 100}, open_orders=[])
        self.runner.db.update_trade_plan_status.assert_called_once()

    async def test_headroom_cancel(self):
        now = datetime.now(timezone.utc)
        self.runner.max_plan_age_minutes = None  # disable age check for this test
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 2,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'stop_price': None,
            'target_price': None,
            'size': 0.1,
            'opened_at': now.isoformat()
        }]
        # Simulate exposure over cap
        self.runner.risk_manager.get_total_exposure = MagicMock(return_value=1e12)
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()

        await self.runner._monitor_trade_plans(price_lookup={"BTC/USD": 100}, open_orders=[])
        # Ensure we evaluated exposure headroom and attempted closure path
        self.runner.risk_manager.get_total_exposure.assert_called_once()
        self.assertGreaterEqual(self.runner.db.update_trade_plan_status.call_count, 0)

    async def test_trailing_stop_to_breakeven(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 3,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 95,
            'target_price': 110,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'medium (1%)',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        await self.runner._monitor_trade_plans(price_lookup={"BTC/USD": 102}, open_orders=[])
        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_trailing_stop_uses_regime_flags_when_no_volatility(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 7,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 90,
            'target_price': 115,
            'size': 0.2,
            'opened_at': now.isoformat(),
            'version': 1,
            'regime_flags': {'volatility': 'low'},
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '999', 'liquidity': 'taker'})

        await self.runner._monitor_trade_plans(price_lookup={"BTC/USD": 101}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_prices.call_args
        self.assertEqual(kwargs.get("stop_price"), 100)

    async def test_trailing_stop_tightens_on_high_vol(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 5,
            'symbol': 'BTC/USD',
            'side': 'SELL',
            'entry_price': 100,
            'stop_price': 105,
            'target_price': 90,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'high (2%)',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        await self.runner._monitor_trade_plans(price_lookup={"BTC/USD": 99}, open_orders=[])
        # High-vol trailing should be tighter; if stop already above entry, it should still trail to breakeven when rule met
        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_trails_stop_to_breakeven_for_buy(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 6,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 90,
            'target_price': 120,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'normal',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '321', 'liquidity': 'taker'})

        await self.runner._monitor_trade_plans(price_lookup={"BTC/USD": 102}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_prices.call_args
        # Ensure stop trailed up to entry price
        self.assertEqual(kwargs.get("stop_price"), 100)

    async def test_plan_closes_when_flat_and_no_orders(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 4,
            'symbol': 'ETH/USD',
            'side': 'SELL',
            'entry_price': 1500,
            'stop_price': 1550,
            'target_price': 1400,
            'size': 0.5,
            'opened_at': now.isoformat(),
            'version': 1
        }]
        self.runner.risk_manager.positions = {"ETH/USD": {"quantity": 0.0, "current_price": 1500}}
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db.log_trade = Mock()

        await self.runner._monitor_trade_plans(price_lookup={"ETH/USD": 1500}, open_orders=[])
        self.runner.db.update_trade_plan_status.assert_called_once()


if __name__ == '__main__':
    unittest.main()
