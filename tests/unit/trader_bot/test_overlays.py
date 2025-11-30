import unittest

import pytest

from trader_bot.strategy_runner import StrategyRunner

pytestmark = pytest.mark.usefixtures("test_db_path")


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
        # Zero/negative risk or reward should reject
        self.assertFalse(self.runner._passes_rr_filter('BUY', 100, 100, 101))
        self.assertFalse(self.runner._passes_rr_filter('BUY', 100, 101, 100))
        self.assertFalse(self.runner._passes_rr_filter('SELL', 100, 100, 99))
        self.assertFalse(self.runner._passes_rr_filter('SELL', 100, 99, 100))

    def test_stacking_block_when_existing_position_and_pending(self):
        pending = {"count_buy": 1, "buy": 100.0}
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": 1.0, "current_price": 100.0}}
        blocked = self.runner._stacking_block('BUY', "BTC/USD", open_plan_count=1, pending_data=pending, position_qty=1.0)
        self.assertTrue(blocked)
        allowed = self.runner._stacking_block('SELL', "BTC/USD", open_plan_count=1, pending_data=pending, position_qty=1.0)
        self.assertFalse(allowed)

    def test_slippage_guard_helper_dynamic_cap(self):
        rich_md = {"bid": 100, "ask": 100.1, "bid_size": 10, "ask_size": 10, "spread_pct": 0.05}
        thin_md = {"bid": 100, "ask": 100.1, "bid_size": 0.1, "ask_size": 0.1, "spread_pct": 0.5}

        # With healthy depth, small move should pass
        ok, move = self.runner._slippage_within_limit(100, 100.2, rich_md)
        self.assertTrue(ok)
        self.assertGreaterEqual(move, 0)

        # Thin book should reduce cap; same move more likely to fail
        ok_thin, _ = self.runner._slippage_within_limit(100, 100.6, thin_md)
        self.assertFalse(ok_thin)

    def test_prefer_maker_overrides(self):
        self.runner.maker_preference_default = True
        self.runner.maker_preference_overrides = {"ETH/USD": False}
        self.assertTrue(self.runner._prefer_maker("BTC/USD"))
        self.assertFalse(self.runner._prefer_maker("eth/usd"))


if __name__ == '__main__':
    unittest.main()
