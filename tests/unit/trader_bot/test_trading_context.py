import json
import os
import tempfile
import unittest
from unittest.mock import patch

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.database import TradingDatabase
from trader_bot.trading_context import TradingContext


class TestTradingContext(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "context.db")
        self.db = TradingDatabase(self.db_path)
        self.portfolio_id, _ = self.db.ensure_active_portfolio(name="test-portfolio", bot_version="test-version")

        # Simple trade history and market data
        self.db.log_trade_for_portfolio(self.portfolio_id, "BTC/USD", "BUY", 0.1, 20000.0, 2.0, "entry")
        self.db.log_trade_for_portfolio(self.portfolio_id, "BTC/USD", "SELL", 0.05, 21000.0, 1.0, "trim")
        for i in range(25):
            price = 20000.0 + i * 10
            self.db.log_market_data_for_portfolio(self.portfolio_id, "BTC/USD", price, price - 5, price + 5)

        self.context = TradingContext(self.db, self.portfolio_id, run_id="run-ctx")

    def tearDown(self):
        self.db.close()
        self.tmpdir.cleanup()

    def test_context_summary_contains_key_sections(self):
        summary = self.context.get_context_summary("BTC/USD")
        parsed = json.loads(summary)
        self.assertIn("portfolio", parsed)
        self.assertIn("positions", parsed)
        self.assertIn("open_orders", parsed)
        self.assertIn("recent_trades", parsed)
        self.assertIn("trend_pct", parsed)
        self.assertIn("win_rate_pct", parsed["portfolio"])
        self.assertIn("venue", parsed)

    def test_memory_snapshot_is_capped_and_includes_plans_and_traces(self):
        # Create a plan and traces to populate memory
        plan_id = self.db.create_trade_plan_for_portfolio(
            self.portfolio_id,
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
        trace_id = self.db.log_llm_trace_for_portfolio(
            self.portfolio_id,
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

    def test_filtering_net_quantity_and_baseline_merge(self):
        self.context.set_position_baseline({"BTC/USD": 1.0})
        self.context.set_position_baseline({"ETH/USD": -0.5})
        self.assertEqual(self.context.position_baseline["BTC/USD"], 1.0)
        self.assertEqual(self.context.position_baseline["ETH/USD"], -0.5)

        orders = [
            {"order_id": "o1", "clientOrderId": f"{CLIENT_ORDER_PREFIX}-abc"},
            {"order_id": "o2", "clientOrderId": "OTHER-123"},
            {"order_id": "o3", "info": {"client_order": f"{CLIENT_ORDER_PREFIX}-def"}},
        ]
        filtered = self.context._filter_our_orders(orders)
        self.assertEqual([o["order_id"] for o in filtered], ["o1", "o3"])

        self.assertEqual(self.context._net_quantity_with_baseline(1.5, 1.0), 0.5)
        self.assertEqual(self.context._net_quantity_with_baseline(-0.6, -0.3), -0.3)
        self.assertEqual(self.context._net_quantity_with_baseline(0.25, -0.5), 0.25)

    def test_context_summary_win_rate_trend_and_trimming(self):
        # Add a losing sell to pair with existing win
        self.db.log_trade_for_portfolio(self.portfolio_id, "BTC/USD", "SELL", 0.05, 19000.0, 1.0, "stop")
        # Add filler trades to exceed max_trades cap
        for i in range(4):
            self.db.log_trade_for_portfolio(self.portfolio_id, "ETH/USD", "BUY", 0.1, 1000 + i, 0.1, f"entry-{i}")

        positions = [
            {"symbol": f"SYM{i}", "quantity": 0.5 + i, "avg_price": 1000 + i * 10, "timestamp": None}
            for i in range(6)
        ]
        self.db.replace_positions_for_portfolio(self.portfolio_id, positions)

        open_orders = [
            {
                "order_id": f"order-{i}",
                "symbol": "BTC/USD",
                "side": "buy",
                "price": 20000 + i,
                "amount": 0.01,
                "remaining": 0.01,
                "status": "open",
                "clientOrderId": f"{CLIENT_ORDER_PREFIX}-{i}",
            }
            for i in range(6)
        ] + [{"order_id": "ignored", "clientOrderId": "OTHER", "symbol": "BTC/USD", "side": "buy", "price": 1, "amount": 0.01, "remaining": 0.01, "status": "open"}]

        summary = json.loads(self.context.get_context_summary("BTC/USD", open_orders=open_orders))
        self.assertEqual(summary["portfolio"]["win_rate_pct"], 50.0)
        self.assertIsNotNone(summary["trend_pct"])
        self.assertEqual(len(summary["positions"]), 5)
        self.assertEqual(len(summary["open_orders"]), 5)
        self.assertEqual(len(summary["recent_trades"]), 5)

    def test_context_summary_uses_equity_baseline(self):
        self.db.log_equity_snapshot_for_portfolio(self.portfolio_id, 1234.5, timestamp="2024-01-01T00:00:00Z")
        summary = json.loads(self.context.get_context_summary("BTC/USD"))
        portfolio = summary["portfolio"]
        self.assertEqual(portfolio["starting_balance"], 1234.5)
        self.assertIsNotNone(portfolio["baseline_timestamp"])

    def test_memory_snapshot_trims_large_payloads(self):
        for i in range(3):
            self.db.create_trade_plan_for_portfolio(
                self.portfolio_id,
                symbol="BTC/USD",
                side="BUY",
                entry_price=20000 + i,
                stop_price=19000,
                target_price=21000,
                size=0.1 + i,
                reason=f"plan-{i}",
                entry_order_id=f"ord-{i}",
                entry_client_order_id=f"cli-{i}",
            )
        for i in range(2):
            trace_id = self.db.log_llm_trace_for_portfolio(
                self.portfolio_id,
                prompt="p",
                response="r",
                decision_json=json.dumps({"idx": i}),
                market_context=None,
            )
            self.db.update_llm_trace_execution(trace_id, {"status": "ok"})

        with patch("trader_bot.trading_context.estimate_json_bytes") as estimate:
            estimate.side_effect = lambda payload: (
                len(payload.get("open_plans", [])) * 100 + len(payload.get("recent_decisions", [])) * 80 + 10
            )
            snapshot = self.context.get_memory_snapshot(max_bytes=200, max_plans=5, max_traces=5)
        parsed = json.loads(snapshot)
        self.assertEqual(len(parsed["recent_decisions"]), 0)
        self.assertEqual(len(parsed["open_plans"]), 1)

    def test_memory_snapshot_returns_empty_on_json_failure(self):
        class Weird:
            pass

        class BrokenDB:
            def get_open_trade_plans_for_portfolio(self, portfolio_id):
                return [{"id": 1, "symbol": "BTC/USD", "side": "BUY", "size": Weird(), "entry_price": 1}]

            def get_recent_llm_traces_for_portfolio(self, portfolio_id, limit=5):
                return []

        ctx = TradingContext(BrokenDB(), portfolio_id=1)
        result = ctx.get_memory_snapshot(max_bytes=50)
        self.assertEqual(result, "")

    def test_recent_performance_defaults_and_profit_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.db")
            db = TradingDatabase(db_path)
            portfolio_id, _ = db.ensure_active_portfolio(name="perf", bot_version="v1")
            ctx = TradingContext(db, portfolio_id)

            empty_perf = ctx.get_recent_performance()
            self.assertEqual(empty_perf, {
                'total_trades': 0,
                'avg_profit': 0,
                'win_rate': 0,
                'last_trade_profitable': None
            })

            db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "BUY", 1, 100, 1, "entry")
            db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "SELL", 1, 110, 1, "take-profit")
            db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "BUY", 1, 100, 1, "entry2")
            db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "SELL", 1, 90, 1, "stop")

            perf = ctx.get_recent_performance()
            self.assertEqual(perf["total_trades"], 4)
            self.assertAlmostEqual(perf["avg_profit"], -2)
            self.assertEqual(perf["win_rate"], 50.0)
            self.assertFalse(perf["last_trade_profitable"])
            db.close()


if __name__ == "__main__":
    unittest.main()
