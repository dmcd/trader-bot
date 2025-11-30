import unittest
from datetime import datetime, timezone

from trader_bot.cost_tracker import CostTracker


class TestCostTracker(unittest.TestCase):
    def test_gemini_fee_percentage(self):
        tracker = CostTracker("GEMINI")
        fee = tracker.calculate_trade_fee("BTC/USD", quantity=0.5, price=20000.0, action="BUY")
        expected = 0.5 * 20000.0 * tracker.fee_rates["GEMINI"]["taker"]
        self.assertAlmostEqual(fee, expected)

    def test_gemini_fee_maker_vs_taker(self):
        tracker = CostTracker("GEMINI")
        trade_value = 1.0 * 1000.0
        taker_fee = tracker.calculate_trade_fee("BTC/USD", quantity=1.0, price=1000.0, action="BUY", liquidity="taker")
        maker_fee = tracker.calculate_trade_fee("BTC/USD", quantity=1.0, price=1000.0, action="BUY", liquidity="maker")
        self.assertAlmostEqual(taker_fee, trade_value * tracker.fee_rates["GEMINI"]["taker"])
        self.assertAlmostEqual(maker_fee, trade_value * tracker.fee_rates["GEMINI"]["maker"])
        self.assertLess(maker_fee, taker_fee)

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

    def test_llm_costs_switch_with_provider(self):
        tracker = CostTracker("GEMINI", llm_provider="OPENAI")
        cost = tracker.calculate_llm_cost(input_tokens=2000, output_tokens=1000)
        expected = (
            2000 * tracker.llm_costs_by_provider["OPENAI"]["input_per_token"]
            + 1000 * tracker.llm_costs_by_provider["OPENAI"]["output_per_token"]
        )
        self.assertAlmostEqual(cost, expected)

    def test_unknown_llm_provider_defaults_to_gemini(self):
        tracker = CostTracker("GEMINI", llm_provider="mystery-model")
        self.assertEqual(tracker.llm_costs, tracker.llm_costs_by_provider["GEMINI"])

    def test_unknown_exchange_fee_is_zero_and_warns(self):
        tracker = CostTracker("UNKNOWN")
        fee = tracker.calculate_trade_fee("BTC/USD", quantity=1, price=10000, liquidity="maker")
        self.assertEqual(fee, 0.0)

    def test_llm_burn_rate_projection(self):
        tracker = CostTracker("GEMINI")
        start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
        now = datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc)
        stats = tracker.calculate_llm_burn(total_llm_cost=4.0, session_started=start, budget=10.0, now=now)

        self.assertAlmostEqual(stats["elapsed_hours"], 2.0)
        self.assertAlmostEqual(stats["burn_rate_per_hour"], 2.0)
        self.assertAlmostEqual(stats["pct_of_budget"], 0.4)
        self.assertAlmostEqual(stats["remaining_budget"], 6.0)
        self.assertAlmostEqual(stats["hours_to_cap"], 3.0)

    def test_llm_burn_accepts_iso_strings_and_naive_datetimes(self):
        tracker = CostTracker("GEMINI")
        now = datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc)

        stats_from_z = tracker.calculate_llm_burn(
            total_llm_cost=2.0,
            session_started="2025-01-01T00:00:00Z",
            budget=8.0,
            now=now,
        )
        self.assertAlmostEqual(stats_from_z["elapsed_hours"], 1.0)
        self.assertAlmostEqual(stats_from_z["burn_rate_per_hour"], 2.0)

        stats_from_naive = tracker.calculate_llm_burn(
            total_llm_cost=1.0,
            session_started="2025-01-01T00:00:00",
            budget=5.0,
            now=now,
        )
        self.assertAlmostEqual(stats_from_naive["elapsed_hours"], 1.0)
        self.assertAlmostEqual(stats_from_naive["burn_rate_per_hour"], 1.0)

    def test_cost_summary_includes_ratio_and_profit_flag(self):
        tracker = CostTracker("GEMINI")
        summary = tracker.get_cost_summary(total_fees=10.0, total_llm_cost=5.0, gross_pnl=20.0)
        self.assertEqual(summary["total_costs"], 15.0)
        self.assertEqual(summary["cost_ratio"], 0.75)
        self.assertTrue(summary["profitable"])

    def test_llm_burn_bad_string_defaults_to_min_window(self):
        tracker = CostTracker("GEMINI")
        now = datetime(2025, 1, 1, 0, 10, tzinfo=timezone.utc)

        stats = tracker.calculate_llm_burn(
            total_llm_cost=3.0,
            session_started="not-a-date",
            budget=6.0,
            now=now,
        )

        expected_elapsed_hours = 5.0 / 60.0  # default 5 minute window
        self.assertAlmostEqual(stats["elapsed_hours"], expected_elapsed_hours)
        self.assertAlmostEqual(stats["burn_rate_per_hour"], 3.0 / expected_elapsed_hours)
        self.assertEqual(stats["remaining_budget"], 3.0)
        self.assertAlmostEqual(stats["pct_of_budget"], 0.5)

    def test_llm_burn_min_window_clamps_to_one_second(self):
        tracker = CostTracker("GEMINI")
        now = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)

        stats = tracker.calculate_llm_burn(
            total_llm_cost=1.0,
            session_started=now,
            budget=10.0,
            now=now,
            min_window_minutes=0.0,
        )

        expected_elapsed_hours = 1.0 / 3600.0  # min window clamps to 1 second
        self.assertAlmostEqual(stats["elapsed_hours"], expected_elapsed_hours)
        self.assertAlmostEqual(stats["burn_rate_per_hour"], 3600.0)
        self.assertAlmostEqual(stats["hours_to_cap"], 9.0 / 3600.0)


if __name__ == "__main__":
    unittest.main()
