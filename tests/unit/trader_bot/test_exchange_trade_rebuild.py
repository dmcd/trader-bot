import os
import tempfile
import unittest

from trader_bot.strategy_runner import StrategyRunner


class TestExchangeTradeRebuild(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rebuild.db")
        self.prev_db_path = os.environ.get("TRADING_DB_PATH")
        os.environ["TRADING_DB_PATH"] = self.db_path
        self.runner = StrategyRunner(execute_orders=False)
        self.runner.session_id = 1

    def tearDown(self):
        self.runner.db.close()
        if self.prev_db_path is None:
            os.environ.pop("TRADING_DB_PATH", None)
        else:
            os.environ["TRADING_DB_PATH"] = self.prev_db_path
        self.tmpdir.cleanup()

    def test_rebuild_skips_malformed_trades_and_logs(self):
        valid_trade = {"symbol": "BTC/USD", "side": "buy", "amount": 0.1, "price": 20000.0, "fee": {"cost": 1.5}}
        malformed_trade = {"symbol": "BTC/USD", "side": "buy", "amount": None, "price": 20000.0}

        with self.assertLogs("trader_bot.strategy_runner", level="WARNING") as logs:
            stats = self.runner._apply_exchange_trades_for_rebuild([valid_trade, malformed_trade])

        self.assertEqual(stats["total_trades"], 1)
        self.assertAlmostEqual(stats["total_fees"], 1.5)
        self.assertTrue(any("Skipping malformed trade" in msg for msg in logs.output))

    def test_rebuild_normalizes_fee_structures(self):
        trade = {
            "symbol": "ETH/USD",
            "side": "sell",
            "amount": 1.0,
            "price": 1000.0,
            "fee": [{"cost": 0.1}, {"cost": 0.2}],
        }

        stats = self.runner._apply_exchange_trades_for_rebuild([trade])

        self.assertEqual(stats["total_trades"], 1)
        self.assertAlmostEqual(stats["total_fees"], 0.3)
        self.assertIn("ETH/USD", self.runner.holdings)


if __name__ == "__main__":
    unittest.main()
