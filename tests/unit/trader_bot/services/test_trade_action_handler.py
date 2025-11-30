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
async def test_update_plan_missing_id_records_telemetry(handler):
    result = await handler.handle_update_plan(plan_id=None, stop_price=10, target_price=20, reason="noop", trace_id=None)

    assert result["status"] == "update_plan_missing_id"
    assert handler.telemetry[-1]["status"] == "update_plan_missing_id"


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
async def test_partial_close_missing_plan(handler):
    handler.db.get_open_trade_plans.return_value = []

    result = await handler.handle_partial_close(
        session_id=7,
        plan_id=9,
        close_fraction=0.5,
        symbol="BTC/USD",
        price=100.0,
        current_exposure=0.0,
        trace_id=None,
    )

    assert result["status"] == "partial_close_missing_plan"
    assert handler.telemetry[-1]["status"] == "partial_close_missing_plan"


@pytest.mark.asyncio
async def test_partial_close_zero_size_plan(handler):
    handler.db.get_open_trade_plans.return_value = [{"id": 3, "side": "SELL", "size": 0.0}]

    result = await handler.handle_partial_close(
        session_id=7,
        plan_id=3,
        close_fraction=0.5,
        symbol="BTC/USD",
        price=100.0,
        current_exposure=0.0,
        trace_id=None,
    )

    assert result["status"] == "partial_close_zero_qty"
    assert handler.telemetry[-1]["status"] == "partial_close_zero_qty"


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


def test_liquidity_filter_enforces_min_quote_size(handler):
    ok = handler.liquidity_ok(
        {"bid": 10.0, "ask": 10.1, "bid_size": 20, "ask_size": 25, "instrument_type": "STK"},
        max_spread_pct=1.0,
        min_top_of_book_notional=50.0,
        min_quote_size=50,
    )
    assert ok is False


def test_passes_rr_filter_requires_positive_reward(handler):
    assert handler.passes_rr_filter("BUY", price=100, stop_price=99, target_price=101, min_rr=2.0) is False
    assert handler.passes_rr_filter("SELL", price=100, stop_price=101, target_price=95, min_rr=1.5) is True
    assert handler.passes_rr_filter("BUY", price=100, stop_price=None, target_price=None, min_rr=2.0) is True


def test_slippage_guard_blocks_when_depth_thin(handler):
    allowed, move_pct = handler.slippage_within_limit(
        decision_price=100,
        latest_price=102,
        market_data_point={"spread_pct": 5.0, "bid": 100, "ask": 101, "bid_size": 0.1, "ask_size": 0.1},
        max_slippage_pct=3.0,
        max_spread_pct=1.0,
        min_top_of_book_notional=50.0,
    )
    assert allowed is False
    assert move_pct == pytest.approx(2.0)


def test_compute_slippage_cap_respects_thin_books_and_spreads(handler):
    cap = handler.compute_slippage_cap(
        market_data_point={"bid": 100, "ask": 101, "bid_size": 0.05, "ask_size": 0.05, "spread_pct": 3.0},
        max_slippage_pct=2.0,
        max_spread_pct=1.0,
        min_top_of_book_notional=20.0,
    )

    assert cap < 1.0  # tightened by thin book and spread factor
    assert cap >= 0.2  # bounded by 10% floor of original cap


def test_liquidity_filter_logs_on_thin_depth():
    actions_logger = MagicMock()
    handler = TradeActionHandler(
        db=MagicMock(),
        bot=MagicMock(),
        risk_manager=MagicMock(apply_order_value_buffer=MagicMock(return_value=(1.0, 0.0))),
        cost_tracker=MagicMock(),
        portfolio_tracker=MagicMock(),
        prefer_maker=lambda symbol: True,
        health_manager=MagicMock(),
        emit_telemetry=lambda record: None,
        log_execution_trace=lambda *args, **kwargs: None,
        actions_logger=actions_logger,
        logger=logging.getLogger("handler_test"),
    )

    ok = handler.liquidity_ok(
        {"bid": 100, "ask": 100.5, "bid_size": 0.01, "ask_size": 0.02},
        max_spread_pct=5.0,
        min_top_of_book_notional=5.0,
    )

    assert ok is False
    actions_logger.info.assert_called_once()


def test_apply_order_value_buffer_logs_trim():
    actions_logger = MagicMock()
    handler = TradeActionHandler(
        db=MagicMock(),
        bot=MagicMock(),
        risk_manager=MagicMock(apply_order_value_buffer=MagicMock(return_value=(0.5, 1.0))),
        cost_tracker=MagicMock(),
        portfolio_tracker=MagicMock(),
        prefer_maker=lambda symbol: True,
        health_manager=MagicMock(),
        emit_telemetry=lambda record: None,
        log_execution_trace=lambda *args, **kwargs: None,
        actions_logger=actions_logger,
        logger=logging.getLogger("handler_test"),
    )

    handler.apply_order_value_buffer(1.0, 100.0)
    actions_logger.info.assert_called_once()


def test_stacking_block_flags_same_side_and_pending():
    handler = TradeActionHandler(
        db=MagicMock(),
        bot=MagicMock(),
        risk_manager=MagicMock(),
        cost_tracker=MagicMock(),
        portfolio_tracker=MagicMock(),
        prefer_maker=lambda symbol: True,
        health_manager=MagicMock(),
        emit_telemetry=lambda record: None,
        log_execution_trace=lambda *args, **kwargs: None,
        actions_logger=logging.getLogger("actions_test"),
        logger=logging.getLogger("handler_test"),
    )

    assert handler.stacking_block("BUY", open_plan_count=1, pending_data={"count_buy": 2}, position_qty=0.1) is True
    assert handler.stacking_block("SELL", open_plan_count=1, pending_data={"count_buy": 2}, position_qty=0.1) is False
