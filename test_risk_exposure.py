import unittest
from risk_manager import RiskManager
from config import MAX_TOTAL_EXPOSURE

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()
        # Mock config values if needed, but we rely on default config for now
        # Assuming MAX_TOTAL_EXPOSURE is 1000.0 from our previous edit

    def test_exposure_limit(self):
        # 1. Setup existing positions (e.g., $850 worth)
        self.rm.update_positions({
            'BTC': {'quantity': 0.01, 'current_price': 85000.0} # $850 value
        })
        
        # 2. Try to buy $100 more (Total $950) -> Should be ALLOWED
        result = self.rm.check_trade_allowed('ETH', 'BUY', 1.0, 100.0)
        self.assertTrue(result.allowed, f"Trade should be allowed: {result.reason}")
        
        # 3. Try to buy $180 more (Total $1030) -> Should be REJECTED by exposure limit
        # Note: Order value $180 is under MAX_ORDER_VALUE ($200) so it passes that check
        result = self.rm.check_trade_allowed('ETH', 'BUY', 1.8, 100.0)
        self.assertFalse(result.allowed, "Trade should be rejected due to exposure limit")
        self.assertIn("Total exposure", result.reason)
        
    def test_safe_buffer_warning(self):
        # 1. Setup existing positions ($850)
        self.rm.update_positions({
            'BTC': {'quantity': 0.01, 'current_price': 85000.0}
        })
        
        # 2. Buy $60 (Total $910, > 90% of 1000) -> Allowed but logs warning (we can't easily check logs here but we check allowed)
        result = self.rm.check_trade_allowed('ETH', 'BUY', 0.6, 100.0)
        self.assertTrue(result.allowed)

if __name__ == '__main__':
    unittest.main()
