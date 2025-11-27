import json
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
        parsed = json.loads(summary)
        self.assertIn("session", parsed)
        self.assertIn("positions", parsed)
        self.assertIn("open_orders", parsed)
        self.assertIn("recent_trades", parsed)
        self.assertIn("trend_pct", parsed)
        self.assertIn("win_rate_pct", parsed["session"])


if __name__ == "__main__":
    unittest.main()
