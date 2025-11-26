import unittest

from cost_tracker import CostTracker


class TestCostTracker(unittest.TestCase):
    def test_gemini_fee_percentage(self):
        tracker = CostTracker("GEMINI")
        fee = tracker.calculate_trade_fee("BTC/USD", quantity=0.5, price=20000.0, action="BUY")
        expected = 0.5 * 20000.0 * tracker.fee_rates["GEMINI"]["taker"]
        self.assertAlmostEqual(fee, expected)

    def test_llm_costs_and_net_pnl(self):
        tracker = CostTracker("GEMINI")
        cost = tracker.calculate_llm_cost(input_tokens=1000, output_tokens=500)
        expected = (
            1000 * tracker.llm_costs["input_per_token"]
            + 500 * tracker.llm_costs["output_per_token"]
        )
        self.assertAlmostEqual(cost, expected)

        net = tracker.calculate_net_pnl(gross_pnl=100.0, total_fees=10.0, total_llm_cost=5.0)
        self.assertEqual(net, 85.0)


if __name__ == "__main__":
    unittest.main()
