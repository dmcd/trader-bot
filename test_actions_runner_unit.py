import atexit
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from strategy_runner import StrategyRunner

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


class TestActionHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner = StrategyRunner()
        # Stub bot and db
        self.runner.bot = MagicMock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 1,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 95,
            'target_price': 110,
            'size': 0.2,
            'opened_at': '2025-01-01T00:00:00'
        }]
        self.runner.db.update_trade_plan_prices = MagicMock()
        self.runner.db.log_trade = MagicMock()
        self.runner._apply_fill_to_session_stats = MagicMock()
        self.runner._update_holdings_and_realized = MagicMock(return_value=0.0)
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 0.1
        self.runner.session_id = 1
        self.runner.risk_manager.update_positions({})
        self.runner.risk_manager.update_pending_orders([])
        self.runner.risk_manager.check_trade_allowed = MagicMock(return_value=MagicMock(allowed=True, reason=""))

    async def test_update_plan_handling(self):
        signal = MagicMock()
        signal.action = 'UPDATE_PLAN'
        signal.symbol = 'BTC/USD'
        signal.stop_price = 99
        signal.target_price = 105
        signal.plan_id = 1
        signal.reason = 'tighten'
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(signal=signal, market_data={'BTC/USD': {'price': 100}}, open_orders=[] , current_equity=1000, current_exposure=0)
        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_partial_close_handling(self):
        signal = MagicMock()
        signal.action = 'PARTIAL_CLOSE'
        signal.symbol = 'BTC/USD'
        signal.plan_id = 1
        signal.close_fraction = 0.5
        signal.reason = 'trim'
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(signal=signal, market_data={'BTC/USD': {'price': 100}}, open_orders=[], current_equity=1000, current_exposure=0)
        self.runner.db.log_trade.assert_called_once()

    async def test_close_position_handling(self):
        self.runner.db.get_positions.return_value = [{'symbol': 'BTC/USD', 'quantity': 0.2, 'avg_price': 100}]
        signal = MagicMock()
        signal.action = 'CLOSE_POSITION'
        signal.symbol = 'BTC/USD'
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(signal=signal, market_data={'BTC/USD': {'price': 100}}, open_orders=[], current_equity=1000, current_exposure=0)
        self.runner.db.log_trade.assert_called()

    async def test_pause_trading_sets_pause_until(self):
        signal = MagicMock()
        signal.action = 'PAUSE_TRADING'
        signal.symbol = 'BTC/USD'
        signal.duration_minutes = 1
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(signal=signal, market_data={}, open_orders=[], current_equity=0, current_exposure=0)
        self.assertIsNotNone(self.runner._pause_until)


if __name__ == '__main__':
    unittest.main()
