import unittest

import risk_manager as rm_module
from risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        # Normalize config-sensitive constants for deterministic tests
        self.orig_max_order = rm_module.MAX_ORDER_VALUE
        self.orig_max_total_exposure = rm_module.MAX_TOTAL_EXPOSURE
        self.orig_daily_loss_pct = rm_module.MAX_DAILY_LOSS_PERCENT
        self.orig_daily_loss_abs = rm_module.MAX_DAILY_LOSS
        self.orig_min_trade_size = rm_module.MIN_TRADE_SIZE

        rm_module.MAX_ORDER_VALUE = 500.0
        rm_module.MAX_TOTAL_EXPOSURE = 1000.0
        rm_module.MAX_DAILY_LOSS_PERCENT = 5.0
        rm_module.MAX_DAILY_LOSS = 50.0
        rm_module.MIN_TRADE_SIZE = 1.0

        self.rm = RiskManager()
        # Seed start of day equity so percent checks work
        self.rm.seed_start_of_day(1000.0)

    def tearDown(self):
        rm_module.MAX_ORDER_VALUE = self.orig_max_order
        rm_module.MAX_TOTAL_EXPOSURE = self.orig_max_total_exposure
        rm_module.MAX_DAILY_LOSS_PERCENT = self.orig_daily_loss_pct
        rm_module.MAX_DAILY_LOSS = self.orig_daily_loss_abs
        rm_module.MIN_TRADE_SIZE = self.orig_min_trade_size

    def test_invalid_price_or_quantity_rejected(self):
        result = self.rm.check_trade_allowed("BHP", "BUY", 0, 100.0)
        self.assertFalse(result.allowed)
        result = self.rm.check_trade_allowed("BHP", "BUY", 1, 0.0)
        self.assertFalse(result.allowed)

    def test_order_value_cap(self):
        over_size_qty = (rm_module.MAX_ORDER_VALUE / 10.0) + 1
        result = self.rm.check_trade_allowed("BHP", "BUY", over_size_qty, price=10.0)
        self.assertFalse(result.allowed)
        self.assertIn("Order value", result.reason)

    def test_daily_loss_percent_and_absolute(self):
        # Simulate equity drop beyond percent
        self.rm.update_equity(890.0)  # 11% drawdown from 1000
        result = self.rm.check_trade_allowed("BHP", "BUY", 1, 10.0)
        self.assertFalse(result.allowed)

        # Simulate absolute loss breach
        self.rm.daily_loss = rm_module.MAX_DAILY_LOSS + 1
        result = self.rm.check_trade_allowed("BHP", "BUY", 1, 10.0)
        self.assertFalse(result.allowed)

    def test_exposure_limit_and_sandbox_adjustment(self):
        # Existing BTC sandbox seed should be ignored in PAPER mode when avg_price missing
        self.rm.is_sandbox = True
        self.rm.update_positions({
            "BTC/USD": {"quantity": 1000.0, "current_price": 30000.0},  # discounted to 0 exposure
            "ETH/USD": {"quantity": 90.0, "current_price": 90.0},  # $900 exposure using trade price override
        })

        # Small order under exposure cap passes
        result = self.rm.check_trade_allowed("ETH/USD", "BUY", 1, price=10.0)  # +$10 => $910
        self.assertTrue(result.allowed)

        # Order that pushes exposure above cap while still under order cap
        qty_to_exceed = 15  # 15 * $10 = $150; exposure becomes 900 + 150 = 1050 > cap
        result = self.rm.check_trade_allowed("ETH/USD", "BUY", qty_to_exceed, price=10.0)
        self.assertFalse(result.allowed)
        self.assertIn("Total exposure", result.reason)

    def test_safe_buffer_allows_near_cap(self):
        near_cap_price = rm_module.MAX_TOTAL_EXPOSURE / 9.5  # ~10.5% below cap
        self.rm.update_positions({"ABC": {"quantity": 9.0, "current_price": near_cap_price}})
        result = self.rm.check_trade_allowed("XYZ", "BUY", 1, price=1.0)
        self.assertTrue(result.allowed)

    def test_get_total_exposure_respects_overrides_and_sandbox(self):
        self.rm.is_sandbox = True
        self.rm.update_positions({"BTC/USD": {"quantity": 1005.0, "current_price": 30000.0}})

        exposure = self.rm.get_total_exposure(price_overrides={"BTC/USD": 20000.0})
        # Sandbox should discount 1000 BTC, leaving 5 * 20000
        self.assertAlmostEqual(100000.0, exposure)


if __name__ == "__main__":
    unittest.main()
