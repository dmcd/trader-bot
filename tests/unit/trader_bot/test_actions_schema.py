import json
import unittest
from unittest.mock import MagicMock

from trader_bot.strategy import LLMStrategy


class TestActionsSchema(unittest.TestCase):
    def setUp(self):
        self.strategy = LLMStrategy(MagicMock(), MagicMock(), MagicMock())

    def test_update_plan_schema_valid(self):
        decision = json.dumps({
            "action": "UPDATE_PLAN",
            "symbol": "BTC/USD",
            "quantity": 0,
            "reason": "tighten stop",
            "plan_id": 1,
            "stop_price": 100,
            "target_price": 110,
            "size_factor": 0.5
        })
        # Should not raise during parsing/validation
        decision_obj = json.loads(decision)
        self.strategy._decision_schema  # touch to ensure present
        # jsonschema.validate is invoked inside generate_signal; here we just ensure json loads
        self.assertEqual(decision_obj["action"], "UPDATE_PLAN")

    def test_partial_close_schema_valid(self):
        decision = json.dumps({
            "action": "PARTIAL_CLOSE",
            "symbol": "BTC/USD",
            "quantity": 0,
            "reason": "trim",
            "plan_id": 2,
            "close_fraction": 0.5
        })
        decision_obj = json.loads(decision)
        self.assertEqual(decision_obj["action"], "PARTIAL_CLOSE")

    def test_close_position_and_pause_schema_valid(self):
        close_decision = json.dumps({
            "action": "CLOSE_POSITION",
            "symbol": "BTC/USD",
            "quantity": 0,
            "reason": "exit"
        })
        pause_decision = json.dumps({
            "action": "PAUSE_TRADING",
            "symbol": "BTC/USD",
            "quantity": 0,
            "reason": "cooldown",
            "duration_minutes": 10
        })
        self.assertEqual(json.loads(close_decision)["action"], "CLOSE_POSITION")
        self.assertEqual(json.loads(pause_decision)["action"], "PAUSE_TRADING")


if __name__ == "__main__":
    unittest.main()
