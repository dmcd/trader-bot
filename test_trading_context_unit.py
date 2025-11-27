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

    def test_memory_snapshot_is_capped_and_includes_plans_and_traces(self):
        # Create a plan and traces to populate memory
        plan_id = self.db.create_trade_plan(
            self.session_id,
            symbol="BTC/USD",
            side="BUY",
            entry_price=20500.0,
            stop_price=20000.0,
            target_price=21000.0,
            size=0.1,
            reason="test plan",
            entry_order_id="ord-1",
            entry_client_order_id="cli-1",
        )
        trace_id = self.db.log_llm_trace(
            self.session_id,
            prompt="p",
            response="r",
            decision_json='{"action":"BUY","symbol":"BTC/USD","quantity":0.1}',
            market_context=None,
        )
        self.db.update_llm_trace_execution(trace_id, {"status": "filled"})

        snapshot = self.context.get_memory_snapshot(max_bytes=500, max_plans=2, max_traces=2)
        self.assertTrue(snapshot)
        parsed = json.loads(snapshot)
        self.assertIn("open_plans", parsed)
        self.assertIn("recent_decisions", parsed)
        self.assertGreaterEqual(len(parsed["open_plans"]), 1)
        self.assertGreaterEqual(len(parsed["recent_decisions"]), 1)
        self.assertLessEqual(len(snapshot.encode("utf-8")), 500)


if __name__ == "__main__":
    unittest.main()
