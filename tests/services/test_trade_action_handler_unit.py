import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader_bot.services.trade_action_handler import TradeActionHandler


@pytest.fixture
def handler():
    db = MagicMock()
    bot = MagicMock()
    bot.place_order_async = AsyncMock(return_value={"order_id": "1", "liquidity": "maker"})
    risk_manager = MagicMock()
    risk_manager.check_trade_allowed.return_value = SimpleNamespace(allowed=True, reason=None)
    risk_manager.apply_order_value_buffer.return_value = (0.5, 0)
    cost_tracker = MagicMock()
    cost_tracker.calculate_trade_fee.return_value = 1.0
    portfolio = MagicMock()
    portfolio.update_holdings_and_realized.return_value = 5.0
    portfolio.apply_fill_to_session_stats.return_value = None
    portfolio.session_stats = {}
    health_manager = MagicMock()
    telemetry = []
    handler = TradeActionHandler(
        db=db,
        bot=bot,
        risk_manager=risk_manager,
        cost_tracker=cost_tracker,
        portfolio_tracker=portfolio,
        prefer_maker=lambda symbol: True,
        health_manager=health_manager,
        emit_telemetry=telemetry.append,
        log_execution_trace=lambda *args, **kwargs: None,
        actions_logger=logging.getLogger("actions_test"),
        logger=logging.getLogger("handler_test"),
    )
    handler.telemetry = telemetry
    handler.db = db
    return handler


@pytest.mark.asyncio
async def test_update_plan_calls_db(handler):
    handler.db.update_trade_plan_prices = MagicMock()

    result = await handler.handle_update_plan(plan_id=42, stop_price=10, target_price=20, reason="tune", trace_id=None)

    handler.db.update_trade_plan_prices.assert_called_once()
    assert result["status"] == "plan_updated"
    assert handler.telemetry[-1]["status"] == "plan_updated"


@pytest.mark.asyncio
async def test_partial_close_executes_and_updates_plan(handler):
    handler.db.get_open_trade_plans.return_value = [{"id": 1, "side": "BUY", "size": 1.0}]
    handler.db.update_trade_plan_size = MagicMock()

    result = await handler.handle_partial_close(
        session_id=7,
        plan_id=1,
        close_fraction=0.5,
        symbol="BTC/USD",
        price=100.0,
        current_exposure=0.0,
        trace_id=None,
    )

    handler.db.log_trade.assert_called_once()
    handler.db.update_trade_plan_size.assert_called_once()
    handler.portfolio_tracker.apply_fill_to_session_stats.assert_called_once()
    assert result["status"] == "partial_close_executed"


@pytest.mark.asyncio
async def test_close_position_handles_missing_positions(handler):
    handler.db.get_positions.return_value = []

    result = await handler.handle_close_position(session_id=1, symbol="ETH/USD", price=2000.0, trace_id=None)

    assert result["status"] == "close_position_none"
    assert handler.telemetry[-1]["status"] == "close_position_none"


def test_liquidity_filter_blocks_wide_spread(handler, caplog):
    caplog.set_level(logging.INFO)
    ok = handler.liquidity_ok({"bid": 100, "ask": 110, "bid_size": 1, "ask_size": 1}, max_spread_pct=5.0, min_top_of_book_notional=10.0)
    assert ok is False
