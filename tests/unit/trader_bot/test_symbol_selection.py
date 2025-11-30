import unittest
from unittest.mock import MagicMock, patch

import pytest

from trader_bot.strategy_runner import StrategyRunner

pytestmark = pytest.mark.usefixtures("test_db_path")


class TestActiveSymbolSelection(unittest.TestCase):
    def setUp(self):
        self.runner = StrategyRunner()
        self.runner.session_id = 1
        self.runner.db = MagicMock()

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", ["BTC/USD", "ETH/USD"])
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

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", [])
    def test_fallback_to_default_when_empty(self):
        self.runner.db.get_positions.return_value = []
        self.runner.db.get_open_orders.return_value = []
        self.runner.db.get_open_trade_plans.return_value = []

        symbols = self.runner._get_active_symbols()
        self.assertEqual(symbols, ["BTC/USD"])

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", ["BTC/USD", "ETH/USD", "DOGE/USD"])
    def test_rebuild_symbols_use_allowlist_and_exchange_filter(self):
        runner = StrategyRunner(execute_orders=False)

        class DummyBot:
            def __init__(self):
                self.exchange = type("Ex", (), {"symbols": ["BTC/USD", "DOGE/USD", "LTC/USD"]})()

        runner.bot = DummyBot()

        symbols = runner._get_rebuild_symbols()

        # ETH/USD is not on the venue list, so only the overlapping allowed symbols are used
        self.assertEqual(symbols, ["BTC/USD", "DOGE/USD"])

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", [])
    def test_rebuild_symbols_fallback_when_allowlist_empty(self):
        runner = StrategyRunner(execute_orders=False)
        runner.bot = type("Bot", (), {"exchange": type("Ex", (), {"symbols": []})()})()

        symbols = runner._get_rebuild_symbols()

        self.assertEqual(symbols, ["BTC/USD"])


if __name__ == '__main__':
    unittest.main()
