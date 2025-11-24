import os
import tempfile
import unittest
from strategy_runner import StrategyRunner

# Ensure tests never write to the production trading.db
_fd, _db_path = tempfile.mkstemp(prefix="trader-bot-test-", suffix=".db")
os.close(_fd)
os.environ.setdefault("TRADING_DB_PATH", _db_path)


class TestDeterministicOverlays(unittest.TestCase):
    def setUp(self):
        self.runner = StrategyRunner()

    def test_volatility_sizing_high_and_medium(self):
        base_qty = 10
        high = self.runner._apply_volatility_sizing(base_qty, {"volatility": "high (2.0%)"})
        med = self.runner._apply_volatility_sizing(base_qty, {"volatility": "medium (1.0%)"})
        normal = self.runner._apply_volatility_sizing(base_qty, {})
        self.assertLess(high, base_qty)
        self.assertLess(med, base_qty)
        self.assertGreater(high, 0)
        self.assertGreater(med, 0)
        self.assertEqual(normal, base_qty)

    def test_rr_filter_requires_minimum_rr(self):
        # BUY: risk 1, reward 1 -> rr 1 < min
        self.assertFalse(self.runner._passes_rr_filter('BUY', 100, 99, 101))
        # BUY: risk 1, reward 2 -> rr 2 >= min
        self.assertTrue(self.runner._passes_rr_filter('BUY', 100, 99, 102))
        # SELL: risk 1, reward 2 -> rr 2 >= min
        self.assertTrue(self.runner._passes_rr_filter('SELL', 100, 101, 98))

    def test_slippage_guard_helper(self):
        ok, move = self.runner._slippage_within_limit(100, 100.2)
        self.assertTrue(ok)
        self.assertGreaterEqual(move, 0)
        ok_bad, _ = self.runner._slippage_within_limit(100, 101)
        self.assertFalse(ok_bad)


if __name__ == '__main__':
    unittest.main()
