import logging
from unittest.mock import MagicMock

import pytest

from trader_bot.strategy_runner import StrategyRunner


def test_order_value_buffer_logs_and_clamps(caplog):
    runner = StrategyRunner(execute_orders=False)
    runner.risk_manager.apply_order_value_buffer = MagicMock(return_value=(0.5, 10.0))

    logger = logging.getLogger("bot_actions")
    original_propagate = logger.propagate
    logger.propagate = True

    with caplog.at_level(logging.INFO, logger="bot_actions"):
        adjusted = runner._apply_order_value_buffer(1.0, 100.0)

    logger.propagate = original_propagate

    assert adjusted == 0.5
    assert any("Trimmed order" in msg for msg in caplog.messages)
