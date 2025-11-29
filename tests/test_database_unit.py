import os
import tempfile
import unittest

from trader_bot.database import TradingDatabase


class TestTradingDatabase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = TradingDatabase(self.db_path)
        self.session_id = self.db.get_or_create_session(starting_balance=5000.0, bot_version="test-version")

    def tearDown(self):
        self.db.close()
        self.tmpdir.cleanup()

    def test_log_and_fetch_entities(self):
        self.db.log_trade(self.session_id, "BTC/USD", "BUY", 0.1, 20000.0, fee=1.0, reason="test")
        self.db.log_llm_call(self.session_id, input_tokens=10, output_tokens=5, cost=0.001, decision="{}")
        self.db.log_market_data(self.session_id, "BTC/USD", price=20000.0, bid=19990.0, ask=20010.0, volume=1000.0)
        self.db.log_equity_snapshot(self.session_id, equity=5050.0)

        trades = self.db.get_recent_trades(self.session_id, limit=5)
        self.assertEqual(len(trades), 1)

        stats = self.db.get_session_stats(self.session_id)
        self.assertEqual(stats["total_trades"], 1)
        self.assertGreaterEqual(stats["total_fees"], 1.0)

        latest_equity = self.db.get_latest_equity(self.session_id)
        self.assertEqual(latest_equity, 5050.0)

        positions = self.db.get_net_positions_from_trades(self.session_id)
        self.assertAlmostEqual(positions["BTC/USD"], 0.1)

    def test_start_of_day_equity_persistence(self):
        baseline = 1234.5
        self.db.set_start_of_day_equity(self.session_id, baseline)
        stored = self.db.get_start_of_day_equity(self.session_id)
        self.assertEqual(stored, baseline)

    def test_log_and_fetch_ohlcv(self):
        bars = [
            {"timestamp": 1_000_000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
            {"timestamp": 1_000_060, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 20},
        ]
        self.db.log_ohlcv_batch(self.session_id, "BTC/USD", "1m", bars)
        fetched = self.db.get_recent_ohlcv(self.session_id, "BTC/USD", "1m", limit=5)
        self.assertEqual(len(fetched), 2)
        # Should return most recent first
        self.assertAlmostEqual(fetched[0]["close"], 2.0)


if __name__ == "__main__":
    unittest.main()
