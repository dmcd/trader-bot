import os
import tempfile
import unittest

from database import TradingDatabase
from trading_context import TradingContext


class TestTradingContext(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "context.db")
        self.db = TradingDatabase(self.db_path)
        self.session_id = self.db.get_or_create_session(starting_balance=10000.0)

        # Simple trade history and market data
        self.db.log_trade(self.session_id, "BTC/USD", "BUY", 0.1, 20000.0, 2.0, "entry")
        self.db.log_trade(self.session_id, "BTC/USD", "SELL", 0.05, 21000.0, 1.0, "trim")
        for i in range(25):
            price = 20000.0 + i * 10
            self.db.log_market_data(self.session_id, "BTC/USD", price, price - 5, price + 5)

        self.context = TradingContext(self.db, self.session_id)

    def tearDown(self):
        self.db.close()
        self.tmpdir.cleanup()

    def test_context_summary_contains_key_sections(self):
        summary = self.context.get_context_summary("BTC/USD")
        self.assertIn("TRADING SESSION CONTEXT", summary)
        self.assertIn("Performance:", summary)
        self.assertIn("Current Positions:", summary)
        self.assertIn("Recent Activity:", summary)
        self.assertIn("Market Trend", summary)


if __name__ == "__main__":
    unittest.main()
