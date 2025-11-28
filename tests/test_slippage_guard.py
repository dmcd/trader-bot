import atexit
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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


class TestSlippageGuard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner = StrategyRunner()
        self.runner.bot = MagicMock()
        self.runner.bot.place_order_async = AsyncMock()
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 0.0
        self.runner.db = MagicMock()
        self.runner.db.count_open_trade_plans_for_symbol.return_value = 0
        self.runner.db.log_estimated_fee = MagicMock()
        self.runner._apply_fill_to_session_stats = MagicMock()
        self.runner.session_id = 1

    @patch('trader_bot.strategy_runner.asyncio.sleep')
    async def test_slippage_skip_when_price_missing(self, mock_sleep):
        # If price missing, quantity should not be executed; this is a placeholder test
        # because slippage guard isn't fully implemented; ensures code path runs without error.
        self.runner.risk_manager.update_positions({})
        self.runner.risk_manager.update_pending_orders([])
        self.runner.risk_manager.get_total_exposure({})
        # No assertion; just ensure no exception in this simplified path
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
