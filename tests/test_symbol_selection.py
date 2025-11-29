import atexit
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from trader_bot.strategy_runner import StrategyRunner

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


class TestActiveSymbolSelection(unittest.TestCase):
    def setUp(self):
        self.runner = StrategyRunner()
        self.runner.session_id = 1
        self.runner.db = MagicMock()

    @patch("trader_bot.strategy_runner.ACTIVE_SYMBOLS", ["BTC/USD", "ETH/USD"])
    def test_merges_configured_and_state_symbols(self):
        self.runner.db.get_positions.return_value = [{'symbol': 'SOL/USD', 'quantity': 1.0}]
        self.runner.db.get_open_orders.return_value = [{'symbol': 'BTC/USD'}, {'symbol': 'ADA/USD'}]
        self.runner.db.get_open_trade_plans.return_value = [{'symbol': 'ETH/USD'}, {'symbol': 'DOGE/USD'}]

        symbols = self.runner._get_active_symbols()

        self.assertEqual(symbols[:2], ["BTC/USD", "ETH/USD"])
        self.assertIn("SOL/USD", symbols)
        self.assertIn("ADA/USD", symbols)
        self.assertIn("DOGE/USD", symbols)
        # Ensure no duplicates and ordering preserved
        self.assertEqual(len(symbols), len(set(symbols)))

    @patch("trader_bot.strategy_runner.ACTIVE_SYMBOLS", [])
    def test_fallback_to_default_when_empty(self):
        self.runner.db.get_positions.return_value = []
        self.runner.db.get_open_orders.return_value = []
        self.runner.db.get_open_trade_plans.return_value = []

        symbols = self.runner._get_active_symbols()
        self.assertEqual(symbols, ["BTC/USD"])


if __name__ == '__main__':
    unittest.main()
