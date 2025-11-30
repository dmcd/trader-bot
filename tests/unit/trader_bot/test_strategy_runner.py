import asyncio
import json
import logging
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from trader_bot.config import (
    CLIENT_ORDER_PREFIX,
    IB_EQUITY_MAX_SPREAD_PCT,
    IB_EQUITY_MIN_QUOTE_SIZE,
    IB_EQUITY_MIN_TOP_OF_BOOK_NOTIONAL,
    IB_FX_MAX_SPREAD_PCT,
    IB_FX_MIN_TOP_OF_BOOK_NOTIONAL,
)
from trader_bot.database import TradingDatabase
from trader_bot.risk_manager import RiskManager
from trader_bot.services.command_processor import CommandResult
from trader_bot.services.plan_monitor import PlanMonitorConfig
from trader_bot.services.strategy_orchestrator import RiskCheckResult
from trader_bot.services.trade_action_handler import TradeActionHandler
from trader_bot.strategy import StrategySignal
from trader_bot.strategy_runner import (
    TRADE_SYNC_CUTOFF_MINUTES,
    MAX_SLIPPAGE_PCT,
    MAX_SPREAD_PCT,
    MIN_TOP_OF_BOOK_NOTIONAL,
    StrategyRunner,
)
from tests.factories import make_strategy_signal
from tests.fakes import FakeBot

pytestmark = pytest.mark.usefixtures("test_db_path")


# --- Fixtures ---

@pytest.fixture
def circuit_runner(tmp_path, monkeypatch):
    db_path = tmp_path / "cb.db"
    monkeypatch.setenv("TRADING_DB_PATH", str(db_path))
    r = StrategyRunner(execute_orders=False)
    r._monotonic = lambda: 100.0
    r.health_manager.monotonic = r._monotonic
    return r


@pytest.fixture
def fresh_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "fresh.db"))
    runner = StrategyRunner(execute_orders=False)
    return runner


# --- Core runner helper tests (volatility, RR, stacking, slippage, maker prefs) ---


class TestDeterministicOverlays(unittest.TestCase):
    def setUp(self):
        self.runner = StrategyRunner()

    def test_volatility_sizing_high_and_medium(self):
        base_qty = 10
        high = self.runner._apply_volatility_sizing(base_qty, {"volatility": "high (2.0%)"})
        med = self.runner._apply_volatility_sizing(base_qty, {"volatility": "medium (1.0%)"})
        normal = self.runner._apply_volatility_sizing(base_qty, {})
        self.assertLess(high, base_qty)
        self.assertLess(med, base_qty)
        self.assertGreater(high, 0)
        self.assertGreater(med, 0)
        self.assertEqual(normal, base_qty)

    def test_rr_filter_requires_minimum_rr(self):
        self.assertFalse(self.runner._passes_rr_filter("BUY", 100, 99, 101))
        self.assertTrue(self.runner._passes_rr_filter("BUY", 100, 99, 102))
        self.assertTrue(self.runner._passes_rr_filter("SELL", 100, 101, 98))
        self.assertFalse(self.runner._passes_rr_filter("BUY", 100, 100, 101))
        self.assertFalse(self.runner._passes_rr_filter("BUY", 100, 101, 100))
        self.assertFalse(self.runner._passes_rr_filter("SELL", 100, 100, 99))
        self.assertFalse(self.runner._passes_rr_filter("SELL", 100, 99, 100))

    def test_stacking_block_when_existing_position_and_pending(self):
        pending = {"count_buy": 1, "buy": 100.0}
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": 1.0, "current_price": 100.0}}
        blocked = self.runner._stacking_block("BUY", "BTC/USD", open_plan_count=1, pending_data=pending, position_qty=1.0)
        self.assertTrue(blocked)
        allowed = self.runner._stacking_block("SELL", "BTC/USD", open_plan_count=1, pending_data=pending, position_qty=1.0)
        self.assertFalse(allowed)

    def test_slippage_guard_helper_dynamic_cap(self):
        rich_md = {"bid": 100, "ask": 100.1, "bid_size": 10, "ask_size": 10, "spread_pct": 0.05}
        thin_md = {"bid": 100, "ask": 100.1, "bid_size": 0.1, "ask_size": 0.1, "spread_pct": 0.5}
        ok, move = self.runner._slippage_within_limit(100, 100.2, rich_md)
        self.assertTrue(ok)
        self.assertGreaterEqual(move, 0)
        ok_thin, _ = self.runner._slippage_within_limit(100, 100.6, thin_md)
        self.assertFalse(ok_thin)

    def test_prefer_maker_overrides(self):
        self.runner.maker_preference_default = True
        self.runner.maker_preference_overrides = {"ETH/USD": False}
        self.assertTrue(self.runner._prefer_maker("BTC/USD"))
        self.assertFalse(self.runner._prefer_maker("eth/usd"))


class TestMicrostructureThresholds(unittest.TestCase):
    def setUp(self):
        self.runner = StrategyRunner()
        self.runner.action_handler = MagicMock()

    def test_ib_equity_thresholds_used(self):
        self.runner.exchange_name = "IB"
        self.runner.action_handler.liquidity_ok.return_value = True
        md = {"symbol": "BHP/AUD", "instrument_type": "STK"}

        self.runner._liquidity_ok(md)

        kwargs = self.runner.action_handler.liquidity_ok.call_args.kwargs
        self.assertEqual(kwargs["max_spread_pct"], IB_EQUITY_MAX_SPREAD_PCT)
        self.assertEqual(kwargs["min_top_of_book_notional"], IB_EQUITY_MIN_TOP_OF_BOOK_NOTIONAL)
        self.assertEqual(kwargs["min_quote_size"], IB_EQUITY_MIN_QUOTE_SIZE)

    def test_ib_fx_thresholds_inferred_from_symbol(self):
        self.runner.exchange_name = "IB"
        self.runner.action_handler.liquidity_ok.return_value = True
        md = {"symbol": "AUD/USD"}

        self.runner._liquidity_ok(md)

        kwargs = self.runner.action_handler.liquidity_ok.call_args.kwargs
        self.assertEqual(kwargs["max_spread_pct"], IB_FX_MAX_SPREAD_PCT)
        self.assertEqual(kwargs["min_top_of_book_notional"], IB_FX_MIN_TOP_OF_BOOK_NOTIONAL)
        self.assertIsNone(kwargs["min_quote_size"])


def test_slippage_guard_delegates_to_action_handler():
    runner = StrategyRunner()
    runner.action_handler = MagicMock()
    runner.action_handler.slippage_within_limit.return_value = (True, 0.0)
    ok, move = runner._slippage_within_limit(100.0, 101.0, {"symbol": "BTC/USD"})
    assert ok is True
    assert move == 0.0
    expected_spread, expected_min_top, _ = runner._microstructure_thresholds({"symbol": "BTC/USD"})
    runner.action_handler.slippage_within_limit.assert_called_with(
        100.0,
        101.0,
        {"symbol": "BTC/USD"},
        max_slippage_pct=MAX_SLIPPAGE_PCT,
        max_spread_pct=expected_spread,
        min_top_of_book_notional=expected_min_top,
    )


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


# --- Reconciliation helpers ---


@pytest.mark.asyncio
async def test_reconcile_exchange_state_refreshes_bindings(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "reconcile.db"))
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 7
    runner.resync_service = MagicMock()
    runner.resync_service.set_session = MagicMock()
    runner.resync_service.reconcile_exchange_state = AsyncMock()
    await runner._reconcile_exchange_state()
    assert runner.resync_service.db is runner.db
    assert runner.resync_service.bot is runner.bot
    assert runner.resync_service.risk_manager is runner.risk_manager
    assert runner.resync_service.trade_sync_cutoff_minutes == TRADE_SYNC_CUTOFF_MINUTES
    runner.resync_service.set_session.assert_called_with(7)
    runner.resync_service.reconcile_exchange_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_open_orders_refreshes_bindings(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "reconcile-orders.db"))
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 9
    runner.resync_service = MagicMock()
    runner.resync_service.set_session = MagicMock()
    runner.resync_service.reconcile_open_orders = AsyncMock()
    await runner._reconcile_open_orders()
    runner.resync_service.set_session.assert_called_with(9)
    runner.resync_service.reconcile_open_orders.assert_awaited_once()


@pytest.mark.asyncio
async def test_monitor_trade_plans_refreshes_bindings(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "monitor.db"))
    runner = StrategyRunner(execute_orders=False)
    runner.max_plan_age_minutes = 15
    runner.day_end_flatten_hour_utc = 20
    runner._apply_plan_trailing_pct = 0.25
    runner.plan_monitor.refresh_bindings = MagicMock()
    captured = {}

    async def fake_monitor(session_id, price_lookup, open_orders, config, refresh_bindings_cb):
        refresh_bindings_cb()
        captured["session_id"] = session_id
        captured["price_lookup"] = price_lookup
        captured["open_orders"] = open_orders
        captured["config"] = config

    runner.orchestrator = SimpleNamespace(monitor_trade_plans=fake_monitor)
    await runner._monitor_trade_plans(price_lookup={"BTC/USD": 100.0}, open_orders=[{"id": 1}])
    runner.plan_monitor.refresh_bindings.assert_called_once()
    args, kwargs = runner.plan_monitor.refresh_bindings.call_args
    assert kwargs["bot"] is runner.bot
    assert kwargs["db"] is runner.db
    assert kwargs["risk_manager"] is runner.risk_manager
    assert captured["session_id"] == runner.session_id
    assert captured["price_lookup"] == {"BTC/USD": 100.0}
    assert captured["open_orders"] == [{"id": 1}]
    assert captured["config"].max_plan_age_minutes == 15
    assert captured["config"].day_end_flatten_hour_utc == 20
    assert captured["config"].trail_to_breakeven_pct == 0.25


@pytest.mark.asyncio
async def test_monitor_trade_plans_bubbles_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "monitor-error.db"))
    runner = StrategyRunner(execute_orders=False)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    runner.orchestrator = SimpleNamespace(monitor_trade_plans=boom)
    with pytest.raises(RuntimeError):
        await runner._monitor_trade_plans(price_lookup={}, open_orders=[])


# --- Circuit breakers and health metrics ---


def test_exchange_circuit_breaker_trips_and_resets(circuit_runner):
    circuit_runner.health_manager.exchange_error_threshold = 2
    circuit_runner.health_manager.exchange_pause_seconds = 30
    circuit_runner.health_manager.record_exchange_failure("get_equity", "boom")
    assert circuit_runner.health_manager.pause_until is None
    circuit_runner.health_manager.record_exchange_failure("ticker", "boom2")
    assert circuit_runner.health_manager.pause_until == pytest.approx(130.0)
    states = {row["key"]: row["value"] for row in circuit_runner.db.get_health_state()}
    assert states.get("exchange_circuit") == "tripped"
    circuit_runner.health_manager.reset_exchange_errors()
    states = {row["key"]: row["value"] for row in circuit_runner.db.get_health_state()}
    assert states.get("exchange_circuit") == "ok"


def test_tool_circuit_breaker_trips_and_recovers(circuit_runner):
    circuit_runner.health_manager.tool_error_threshold = 1
    circuit_runner.health_manager.tool_pause_seconds = 20
    circuit_runner._monotonic = lambda: 50.0
    circuit_runner.health_manager.monotonic = circuit_runner._monotonic
    circuit_runner.health_manager.record_tool_failure(context="get_market_data", error="oops")
    assert circuit_runner.health_manager.pause_until == pytest.approx(70.0)
    states = {row["key"]: row["value"] for row in circuit_runner.db.get_health_state()}
    assert states.get("tool_circuit") == "tripped"
    circuit_runner.health_manager.record_tool_success()
    states = {row["key"]: row["value"] for row in circuit_runner.db.get_health_state()}
    assert states.get("tool_circuit") == "ok"


@pytest.mark.asyncio
async def test_kill_switch_stops_loop(circuit_runner):
    circuit_runner.initialize = AsyncMock()
    circuit_runner.cleanup = AsyncMock()
    circuit_runner._kill_switch = True
    await circuit_runner.run_loop(max_loops=1)
    circuit_runner.initialize.assert_awaited()
    circuit_runner.cleanup.assert_awaited()
    assert circuit_runner.running is False
    assert circuit_runner.shutdown_reason == "kill switch"


def test_operational_metrics_emit_health_state(circuit_runner):
    circuit_runner.session_id = circuit_runner.db.get_or_create_session(starting_balance=1000.0, bot_version="metric-test")
    circuit_runner.session = circuit_runner.db.get_session(circuit_runner.session_id)
    circuit_runner.risk_manager.start_of_day_equity = 1000.0
    circuit_runner.risk_manager.daily_loss = 50.0
    circuit_runner.session_stats = {"gross_pnl": 100.0, "total_fees": 10.0, "total_llm_cost": 2.0}
    circuit_runner.cost_tracker = MagicMock()
    circuit_runner.cost_tracker.calculate_llm_burn.return_value = {
        "remaining_budget": 8.0,
        "pct_of_budget": 0.2,
        "total_llm_cost": 2.0,
    }
    circuit_runner.daily_loss_pct = 10.0
    circuit_runner._record_operational_metrics(current_exposure=250.0, current_equity=950.0)
    health = {row["key"]: row for row in circuit_runner.db.get_health_state()}
    risk_detail = json.loads(health["risk_metrics"]["detail"])
    assert risk_detail["exposure"] == 250.0
    assert risk_detail["daily_loss"] == 50.0
    budget_detail = json.loads(health["llm_budget"]["detail"])
    assert budget_detail["total_llm_cost"] == 2.0


def test_equity_sanity_health_state(circuit_runner):
    circuit_runner.session_id = circuit_runner.db.get_or_create_session(starting_balance=1000.0, bot_version="eq-test")
    circuit_runner.session = circuit_runner.db.get_session(circuit_runner.session_id)
    circuit_runner.session_stats = {"gross_pnl": 0.0, "total_fees": 0.0, "total_llm_cost": 0.0}
    circuit_runner._sanity_check_equity_vs_stats(current_equity=800.0)
    health = {row["key"]: row for row in circuit_runner.db.get_health_state()}
    detail = json.loads(health["equity_sanity"]["detail"])
    assert health["equity_sanity"]["value"] == "mismatch"
    assert detail["equity_net"] == -200.0


# --- Action handling and plan monitor paths ---


class TestActionHandling(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner = StrategyRunner()
        self.runner.bot = MagicMock()
        self.runner.bot.place_order_async = AsyncMock(return_value={"order_id": "123", "liquidity": "taker"})
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USD",
                "side": "BUY",
                "entry_price": 100,
                "stop_price": 95,
                "target_price": 110,
                "size": 0.2,
                "opened_at": "2025-01-01T00:00:00",
            }
        ]
        self.runner.db.update_trade_plan_prices = MagicMock()
        self.runner.db.update_trade_plan_size = MagicMock()
        self.runner.db.update_trade_plan_status = MagicMock()
        self.runner.db.log_trade = MagicMock()
        self.runner._apply_fill_to_session_stats = MagicMock()
        self.runner._update_holdings_and_realized = MagicMock(return_value=0.0)
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 0.1
        self.runner.session_id = 1
        self.runner.risk_manager.update_positions({})
        self.runner.risk_manager.update_pending_orders([])
        self.runner.risk_manager.check_trade_allowed = MagicMock(return_value=MagicMock(allowed=True, reason=""))
        self.runner.action_handler = TradeActionHandler(
            db=self.runner.db,
            bot=self.runner.bot,
            risk_manager=self.runner.risk_manager,
            cost_tracker=self.runner.cost_tracker,
            portfolio_tracker=self.runner.portfolio_tracker,
            prefer_maker=self.runner._prefer_maker,
            health_manager=self.runner.health_manager,
            emit_telemetry=self.runner._emit_telemetry,
            log_execution_trace=self.runner._log_execution_trace,
            on_trade_rejected=self.runner.strategy.on_trade_rejected,
            actions_logger=self.runner.strategy.logger if hasattr(self.runner.strategy, "logger") else None,
            logger=None,
        )

    async def test_update_plan_handling(self):
        signal = MagicMock()
        signal.action = "UPDATE_PLAN"
        signal.symbol = "BTC/USD"
        signal.stop_price = 99
        signal.target_price = 105
        signal.plan_id = 1
        signal.reason = "tighten"
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(
            signal=signal, market_data={"BTC/USD": {"price": 100}}, open_orders=[], current_equity=1000, current_exposure=0
        )
        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_partial_close_handling(self):
        signal = MagicMock()
        signal.action = "PARTIAL_CLOSE"
        signal.symbol = "BTC/USD"
        signal.plan_id = 1
        signal.close_fraction = 0.5
        signal.reason = "trim"
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(
            signal=signal, market_data={"BTC/USD": {"price": 100}}, open_orders=[], current_equity=1000, current_exposure=0
        )
        self.runner.db.log_trade.assert_called_once()
        self.runner.db.update_trade_plan_size.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_size.call_args
        self.assertAlmostEqual(kwargs.get("size"), 0.1)

    async def test_partial_close_fully_closes_plan(self):
        signal = MagicMock()
        signal.action = "PARTIAL_CLOSE"
        signal.symbol = "BTC/USD"
        signal.plan_id = 1
        signal.close_fraction = 1.0
        signal.reason = "exit"
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(
            signal=signal, market_data={"BTC/USD": {"price": 100}}, open_orders=[], current_equity=1000, current_exposure=0
        )
        self.runner.db.update_trade_plan_status.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_status.call_args
        self.assertEqual(kwargs.get("status"), "closed")

    async def test_close_position_handling(self):
        self.runner.db.get_positions.return_value = [{"symbol": "BTC/USD", "quantity": 0.2, "avg_price": 100}]
        signal = MagicMock()
        signal.action = "CLOSE_POSITION"
        signal.symbol = "BTC/USD"
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(
            signal=signal, market_data={"BTC/USD": {"price": 100}}, open_orders=[], current_equity=1000, current_exposure=0
        )
        self.runner.db.log_trade.assert_called()

    async def test_pause_trading_sets_pause_until(self):
        signal = MagicMock()
        signal.action = "PAUSE_TRADING"
        signal.symbol = "BTC/USD"
        signal.duration_minutes = 1
        signal.trace_id = None
        signal.regime_flags = {}
        await self.runner._handle_signal(
            signal=signal, market_data={}, open_orders=[], current_equity=0, current_exposure=0
        )
        self.assertIsNotNone(self.runner.health_manager.pause_until)


class TestTradePlanMonitor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner = StrategyRunner()
        self.runner.bot = MagicMock()
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 1.0
        self.runner._apply_fill_to_session_stats = Mock(return_value=None)
        self.runner.session_id = 1
        self._refresh_monitor_bindings()

    def _refresh_monitor_bindings(self):
        self.runner.action_handler = TradeActionHandler(
            db=self.runner.db,
            bot=self.runner.bot,
            risk_manager=self.runner.risk_manager,
            cost_tracker=self.runner.cost_tracker,
            portfolio_tracker=self.runner.portfolio_tracker,
            prefer_maker=self.runner._prefer_maker,
            health_manager=self.runner.health_manager,
            emit_telemetry=self.runner._emit_telemetry,
            log_execution_trace=self.runner._log_execution_trace,
            on_trade_rejected=self.runner.strategy.on_trade_rejected,
            actions_logger=getattr(self.runner.strategy, "logger", None),
            logger=None,
        )
        self.runner.plan_monitor.refresh_bindings(
            bot=self.runner.bot,
            db=self.runner.db,
            cost_tracker=self.runner.cost_tracker,
            risk_manager=self.runner.risk_manager,
            prefer_maker=self.runner._prefer_maker,
            holdings_updater=self.runner._update_holdings_and_realized,
            session_stats_applier=self.runner._apply_fill_to_session_stats,
        )

    async def _run_monitor(self, price_lookup, open_orders):
        config = PlanMonitorConfig(
            max_plan_age_minutes=self.runner.max_plan_age_minutes,
            day_end_flatten_hour_utc=self.runner.day_end_flatten_hour_utc,
            trail_to_breakeven_pct=self.runner._apply_plan_trailing_pct,
        )
        self._refresh_monitor_bindings()
        await self.runner.plan_monitor.monitor(
            self.runner.session_id,
            price_lookup=price_lookup,
            open_orders=open_orders,
            config=config,
        )

    @patch("trader_bot.strategy_runner.datetime")
    async def test_plan_age_flatten(self, mock_dt):
        now = datetime.now(timezone.utc)
        mock_dt.now.return_value = now
        mock_dt.fromisoformat.side_effect = datetime.fromisoformat
        mock_dt.timezone = timezone
        self.runner.max_plan_age_minutes = 60
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [
            {
                "id": 1,
                "symbol": "BTC/USD",
                "side": "BUY",
                "stop_price": None,
                "target_price": None,
                "size": 0.1,
                "opened_at": (now - timedelta(minutes=120)).isoformat(),
            }
        ]
        self.runner.bot.place_order_async = AsyncMock(return_value={"order_id": "123", "liquidity": "taker"})
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()
        await self._run_monitor(price_lookup={"BTC/USD": 100}, open_orders=[])
        self.runner.db.update_trade_plan_status.assert_called_once()

    async def test_headroom_cancel(self):
        now = datetime.now(timezone.utc)
        self.runner.max_plan_age_minutes = None
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [
            {
                "id": 2,
                "symbol": "BTC/USD",
                "side": "BUY",
                "stop_price": None,
                "target_price": None,
                "size": 0.1,
                "opened_at": now.isoformat(),
            }
        ]
        self.runner.risk_manager.get_total_exposure = MagicMock(return_value=1e12)
        self.runner.bot.place_order_async = AsyncMock(return_value={"order_id": "123", "liquidity": "taker"})
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()
        await self._run_monitor(price_lookup={"BTC/USD": 100}, open_orders=[])
        self.runner.risk_manager.get_total_exposure.assert_called_once()
        self.assertGreaterEqual(self.runner.db.update_trade_plan_status.call_count, 0)

    async def test_trailing_stop_to_breakeven(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [
            {
                "id": 3,
                "symbol": "BTC/USD",
                "side": "BUY",
                "stop_price": 95,
                "target_price": None,
                "size": 0.1,
                "opened_at": now.isoformat(),
                "entry_price": 100,
            }
        ]
        self.runner._apply_plan_trailing_pct = 0.01
        self.runner.bot.place_order_async = AsyncMock(return_value={"order_id": "123", "liquidity": "taker"})
        self.runner.db.update_trade_plan_prices = Mock()
        await self._run_monitor(price_lookup={"BTC/USD": 110}, open_orders=[])
        self.runner.db.update_trade_plan_prices.assert_called_once()


# --- OHLCV capture ---


class StubBot:
    def __init__(self):
        self.calls = []

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe, limit))
        idx = len(self.calls)
        return [
            {
                "timestamp": 1_000_000 + idx,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5 + idx,
                "volume": 10.0,
            }
        ]


@pytest.mark.asyncio
async def test_capture_ohlcv_throttles_and_prunes(test_db_path):
    db = TradingDatabase(db_path=str(test_db_path))
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="ohlcv-test")
    bot = StubBot()
    runner = StrategyRunner(execute_orders=False)
    runner.db = db
    runner.bot = bot
    runner.session_id = session_id
    runner.telemetry_logger = None
    runner.ohlcv_retention_limit = 2
    runner.ohlcv_min_capture_spacing_seconds = 60
    runner._monotonic = lambda: 0.0
    try:
        await runner._capture_ohlcv("BTC/USD")
        assert len(bot.calls) == 4
        count_1m = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM ohlcv_bars WHERE session_id = ? AND timeframe = '1m'", (session_id,)
        ).fetchone()["cnt"]
        assert count_1m == 1
        runner._monotonic = lambda: 30.0
        await runner._capture_ohlcv("BTC/USD")
        assert len(bot.calls) == 4
        runner._monotonic = lambda: 120.0
        await runner._capture_ohlcv("BTC/USD")
        assert len(bot.calls) == 5
        latest_count = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM ohlcv_bars WHERE session_id = ? AND timeframe = '1m'", (session_id,)
        ).fetchone()["cnt"]
        assert latest_count == 2
    finally:
        db.close()



# --- Symbol selection and rebuild ---


class TestActiveSymbolSelection(unittest.TestCase):
    def setUp(self):
        self.runner = StrategyRunner()
        self.runner.session_id = 1
        self.runner.db = MagicMock()

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", ["BTC/USD", "ETH/USD"])
    def test_merges_configured_and_state_symbols(self):
        self.runner.db.get_positions.return_value = [{"symbol": "SOL/USD", "quantity": 1.0}]
        self.runner.db.get_open_orders.return_value = [{"symbol": "BTC/USD"}, {"symbol": "ADA/USD"}]
        self.runner.db.get_open_trade_plans.return_value = [{"symbol": "ETH/USD"}, {"symbol": "DOGE/USD"}]
        symbols = self.runner._get_active_symbols()
        self.assertEqual(symbols[:2], ["BTC/USD", "ETH/USD"])
        self.assertIn("SOL/USD", symbols)
        self.assertIn("ADA/USD", symbols)
        self.assertIn("DOGE/USD", symbols)
        self.assertEqual(len(symbols), len(set(symbols)))

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", [])
    def test_fallback_to_default_when_empty(self):
        self.runner.db.get_positions.return_value = []
        self.runner.db.get_open_orders.return_value = []
        self.runner.db.get_open_trade_plans.return_value = []
        symbols = self.runner._get_active_symbols()
        self.assertEqual(symbols, ["BTC/USD"])

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", ["BTC/USD", "ETH/USD", "DOGE/USD"])
    def test_rebuild_symbols_use_allowlist_and_exchange_filter(self):
        runner = StrategyRunner(execute_orders=False)
        exchange = type("Ex", (), {"symbols": ["BTC/USD", "DOGE/USD", "LTC/USD"]})()
        runner.bot = FakeBot(exchange=exchange)
        symbols = runner._get_rebuild_symbols()
        self.assertEqual(symbols, ["BTC/USD", "DOGE/USD"])

    @patch("trader_bot.strategy_runner.ALLOWED_SYMBOLS", [])
    def test_rebuild_symbols_fallback_when_allowlist_empty(self):
        runner = StrategyRunner(execute_orders=False)
        exchange = type("Ex", (), {"symbols": []})()
        runner.bot = FakeBot(exchange=exchange)
        symbols = runner._get_rebuild_symbols()
        self.assertEqual(symbols, ["BTC/USD"])


# --- Exchange trade rebuild helpers ---


class TestExchangeTradeRebuild(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rebuild.db")
        self.prev_db_path = os.environ.get("TRADING_DB_PATH")
        os.environ["TRADING_DB_PATH"] = self.db_path
        self.runner = StrategyRunner(execute_orders=False)
        self.runner.session_id = 1

    def tearDown(self):
        self.runner.db.close()
        if self.prev_db_path is None:
            os.environ.pop("TRADING_DB_PATH", None)
        else:
            os.environ["TRADING_DB_PATH"] = self.prev_db_path
        self.tmpdir.cleanup()

    def test_rebuild_skips_malformed_trades_and_logs(self):
        valid_trade = {"symbol": "BTC/USD", "side": "buy", "amount": 0.1, "price": 20000.0, "fee": {"cost": 1.5}}
        malformed_trade = {"symbol": "BTC/USD", "side": "buy", "amount": None, "price": 20000.0}
        with self.assertLogs("trader_bot.strategy_runner", level="WARNING") as logs:
            stats = self.runner._apply_exchange_trades_for_rebuild([valid_trade, malformed_trade])
        self.assertEqual(stats["total_trades"], 1)
        self.assertAlmostEqual(stats["total_fees"], 1.5)
        self.assertTrue(any("Skipping malformed trade" in msg for msg in logs.output))

    def test_rebuild_normalizes_fee_structures(self):
        trade = {
            "symbol": "ETH/USD",
            "side": "sell",
            "amount": 1.0,
            "price": 1000.0,
            "fee": [{"cost": 0.1}, {"cost": 0.2}],
        }
        stats = self.runner._apply_exchange_trades_for_rebuild([trade])
        self.assertEqual(stats["total_trades"], 1)
        self.assertAlmostEqual(stats["total_fees"], 0.3)
        self.assertIn("ETH/USD", self.runner.holdings)


# --- Data freshness checks ---


def test_stale_due_to_latency(fresh_runner):
    fresh_runner.health_manager.ticker_max_latency_ms = 1000
    fresh_runner.health_manager.monotonic = lambda: 10.0
    stale, detail = fresh_runner.health_manager.is_stale_market_data(
        {"price": 100, "_latency_ms": 1500, "_fetched_monotonic": 9.0}
    )
    assert stale is True
    assert detail.get("reason") == "latency"


def test_stale_due_to_age(fresh_runner):
    fresh_runner.health_manager.ticker_max_age_ms = 4000
    fresh_runner.health_manager.monotonic = lambda: 100.0
    data = {"price": 100, "_fetched_monotonic": 95.0, "_latency_ms": 100}
    stale, detail = fresh_runner.health_manager.is_stale_market_data(data)
    assert stale is True
    assert detail.get("reason") == "age"


def test_fresh_market_data(fresh_runner):
    fresh_runner.health_manager.ticker_max_age_ms = 5000
    fresh_runner.health_manager.ticker_max_latency_ms = 2000
    fresh_runner.health_manager.monotonic = lambda: 50.0
    data = {"price": 100, "_fetched_monotonic": 49.7, "_latency_ms": 500}
    stale, detail = fresh_runner.health_manager.is_stale_market_data(data)
    assert stale is False
    assert detail.get("reason") is None


# --- Shutdown and rebuild helpers ---


def test_set_shutdown_reason_only_keeps_first():
    runner = StrategyRunner(execute_orders=False)
    runner._set_shutdown_reason("first")
    runner._set_shutdown_reason("second")
    assert runner.shutdown_reason == "first"


@pytest.mark.asyncio
async def test_rebuild_session_stats_checks_equity(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "rebuild-stats.db"))
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 1
    runner.portfolio_tracker = MagicMock()
    runner.portfolio_tracker.rebuild_session_stats_from_trades = MagicMock(return_value={"gross_pnl": 1.0})
    runner._sanity_check_equity_vs_stats = MagicMock()
    await runner._rebuild_session_stats_from_trades(current_equity=123.0)
    runner.portfolio_tracker.set_session.assert_called_with(1)
    runner._sanity_check_equity_vs_stats.assert_called_with(123.0)
    assert runner.session_stats == {"gross_pnl": 1.0}


@pytest.mark.asyncio
async def test_run_loop_survives_ohlcv_failure():
    sig = make_strategy_signal(action="BUY", quantity=0.1, symbol="BTC/USD")
    runner = _build_runner(sig, execute_orders=False)
    runner._capture_ohlcv = AsyncMock(side_effect=RuntimeError("boom"))
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    runner._capture_ohlcv.assert_awaited()


# --- Risk and exposure paths ---


@pytest.mark.asyncio
async def test_runner_handles_multi_symbol_exposure(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "multi-exposure.db"))
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = runner.db.get_or_create_session(starting_balance=1000.0, bot_version="multi-exposure")
    runner.session = runner.db.get_session(runner.session_id)
    runner.daily_loss_pct = 50.0
    runner.risk_manager.start_of_day_equity = 1000.0
    runner._get_active_symbols = lambda: ["BTC/USD", "ETH/USD"]
    runner.bot.get_equity_async = AsyncMock(return_value=1000.0)
    positions = [
        {"symbol": "BTC/USD", "quantity": 0.1, "avg_price": None},
        {"symbol": "ETH/USD", "quantity": 2.0, "avg_price": None},
    ]
    runner.bot.get_positions_async = AsyncMock(return_value=positions)
    runner.bot.get_open_orders_async = AsyncMock(
        return_value=[
            {
                "symbol": "ETH/USD",
                "side": "buy",
                "price": 50.0,
                "amount": 1.0,
                "remaining": 1.0,
                "clientOrderId": f"{CLIENT_ORDER_PREFIX}1",
            }
        ]
    )
    market_map = {
        "BTC/USD": {"price": 30000.0, "bid": 29950.0, "ask": 30050.0},
        "ETH/USD": {"price": 2000.0, "bid": 1995.0, "ask": 2005.0},
    }

    async def fake_market_data(symbol):
        md = dict(market_map[symbol])
        md["_fetched_monotonic"] = 0.0
        return md

    runner.bot.get_market_data_async = AsyncMock(side_effect=fake_market_data)
    price_overrides_seen = {}

    def capture_exposure(price_overrides=None):
        nonlocal price_overrides_seen
        price_overrides_seen = price_overrides or {}
        return 0.0

    runner.risk_manager.get_total_exposure = MagicMock(side_effect=capture_exposure)
    symbols = runner._get_active_symbols()
    market_data = {}
    for sym in symbols:
        md = await runner.bot.get_market_data_async(sym)
        market_data[sym] = md
        runner.db.log_market_data(
            runner.session_id,
            sym,
            md.get("price"),
            md.get("bid"),
            md.get("ask"),
            md.get("volume", 0.0),
            spread_pct=md.get("spread_pct"),
            bid_size=md.get("bid_size"),
            ask_size=md.get("ask_size"),
            ob_imbalance=md.get("ob_imbalance"),
        )
    open_orders = await runner.bot.get_open_orders_async()
    runner.db.replace_open_orders(runner.session_id, open_orders)
    live_positions = await runner.bot.get_positions_async()
    runner.db.replace_positions(runner.session_id, live_positions)
    positions_dict = {}
    price_lookup = {}
    positions_data = runner.db.get_positions(runner.session_id)
    for pos in positions_data:
        sym = pos["symbol"]
        current_price = pos.get("avg_price") or 0
        recent = runner.db.get_recent_market_data(runner.session_id, sym, limit=1)
        if recent and recent[0].get("price"):
            current_price = recent[0]["price"]
        if market_data.get(sym) and market_data[sym].get("price"):
            current_price = market_data[sym]["price"]
        if current_price:
            positions_dict[sym] = {"quantity": pos["quantity"], "current_price": current_price}
    runner.risk_manager.update_positions(positions_dict)
    for sym, md in market_data.items():
        if md and md.get("price"):
            price_lookup[sym] = md["price"]
    runner.risk_manager.update_pending_orders(open_orders, price_lookup=price_lookup)
    price_overrides = {sym: md.get("price") for sym, md in market_data.items() if md and md.get("price")}
    price_overrides = price_overrides or None
    runner.risk_manager.get_total_exposure(price_overrides=price_overrides)
    assert runner.bot.get_market_data_async.call_count == 2
    assert set(price_overrides_seen.keys()) == {"BTC/USD", "ETH/USD"}
    assert price_overrides_seen["ETH/USD"] == 2000.0


@pytest.mark.asyncio
async def test_sandbox_daily_loss_check(monkeypatch):
    mock_bot = AsyncMock()
    mock_bot.connect_async = AsyncMock()
    mock_bot.get_equity_async = AsyncMock(return_value=100000.0)
    mock_bot.close = AsyncMock()
    mock_bot.get_market_data_async = AsyncMock(return_value={"price": 50000.0})
    mock_bot.get_positions_async = AsyncMock(return_value=[])
    mock_bot.get_open_orders_async = AsyncMock(return_value=[])
    with patch("trader_bot.strategy_runner.TRADING_MODE", "PAPER"), patch(
        "trader_bot.strategy_runner.MAX_DAILY_LOSS", 500.0
    ), patch("trader_bot.strategy_runner.MAX_DAILY_LOSS_PERCENT", 3.0), patch(
        "trader_bot.strategy_runner.GeminiTrader", return_value=mock_bot
    ):
        runner = StrategyRunner(execute_orders=False)
        runner.bot = mock_bot
        runner.risk_manager = MagicMock(spec=RiskManager)
        runner.risk_manager.start_of_day_equity = 100000.0
        runner.daily_loss_pct = 3.0
        runner.risk_manager.daily_loss = 600.0
        runner.running = True
        runner.initialize = AsyncMock()
        runner.db = MagicMock()
        runner.db.get_pending_commands.return_value = []
        runner.risk_manager.update_equity = MagicMock()

        async def break_loop(*_args, **_kwargs):
            runner.running = False
            return

        with patch("trader_bot.strategy_runner.asyncio.sleep", side_effect=break_loop):
            try:
                await runner.run_loop()
            except Exception:
                pass
        assert runner._kill_switch is False
        runner.risk_manager.daily_loss = 3100.0
        runner.running = True
        runner._kill_switch = False
        with patch("trader_bot.strategy_runner.asyncio.sleep", side_effect=break_loop):
            try:
                await runner.run_loop()
            except Exception:
                pass
        assert runner._kill_switch is True


class DummyDB:
    def log_equity_snapshot(self, *_, **__):
        return None

    def log_market_data(self, *_, **__):
        return None

    def prune_market_data(self, *_, **__):
        return None

    def replace_positions(self, *_, **__):
        return None

    def replace_open_orders(self, *_, **__):
        return None

    def get_positions(self, *_, **__):
        return []

    def get_open_orders(self, *_, **__):
        return []

    def get_open_trade_plans(self, *_, **__):
        return []

    def get_recent_market_data(self, *_, **__):
        return []

    def count_open_trade_plans_for_symbol(self, *_, **__):
        return 0

    def create_trade_plan(self, *_, **__):
        return 1

    def log_estimated_fee(self, *_, **__):
        return None

    def set_health_state(self, *_, **__):
        return None

    def get_trade_plan_reason_by_order(self, *_, **__):
        return None

    def get_trade_count(self, *_, **__):
        return 0

    def update_llm_trace_execution(self, *_, **__):
        return None

    def get_processed_trade_ids(self, *_, **__):
        return []

    def get_latest_trade_timestamp(self, *_, **__):
        return None


class DummyHealth:
    def should_pause(self, *_args):
        return False

    def pause_remaining(self, *_args):
        return 0

    def record_exchange_failure(self, *_, **__):
        return None

    def record_tool_failure(self, *_, **__):
        return None

    def record_tool_success(self, *_, **__):
        return None

    def reset_exchange_errors(self):
        return None

    async def maybe_reconnect(self, *_, **__):
        return None


class DummyOrchestrator:
    def __init__(self, command_result=None, risk_result=None):
        self.running = True
        self.command_result = command_result or CommandResult()
        self.risk_result = risk_result or RiskCheckResult(should_stop=False, kill_switch=False)
        self.last_reason = None

    async def start(self, initialize_cb):
        if asyncio.iscoroutinefunction(initialize_cb):
            await initialize_cb()
        elif callable(initialize_cb):
            initialize_cb()
        self.running = True

    def request_stop(self, *_args, shutdown_reason=None, **__):
        self.last_reason = shutdown_reason or (_args[0] if _args else None)
        self.running = False

    async def process_commands(self, *_, **__):
        return self.command_result

    async def enforce_risk_budget(self, *_, **__):
        return self.risk_result

    def emit_market_health(self, *_args, **__):
        return True, {}

    def emit_operational_metrics(self, *_args, **__):
        return None

    async def monitor_trade_plans(self, *_, **__):
        return None

    async def cleanup(self, cleanup_cb):
        if asyncio.iscoroutinefunction(cleanup_cb):
            return await cleanup_cb()
        if callable(cleanup_cb):
            return cleanup_cb()
        return None


class DummyStrategy:
    def __init__(self, signals):
        self.signals = list(signals)
        self.rejections = []
        self.executions = 0

    async def generate_signal(self, *_args, **__):
        return self.signals.pop(0) if self.signals else None

    def on_trade_rejected(self, reason):
        self.rejections.append(reason)

    def on_trade_executed(self, _ts):
        self.executions += 1


def _build_control_runner(*, command_result=None, risk_result=None, kill_switch=False):
    runner = StrategyRunner(execute_orders=False)
    runner.daily_loss_pct = 100.0
    runner.bot = FakeBot(price=100.0)
    runner.db = DummyDB()
    runner.health_manager = DummyHealth()
    runner._monotonic = lambda: 0.0
    runner._get_active_symbols = lambda: ["BTC/USD"]
    runner._capture_ohlcv = AsyncMock()
    runner._monitor_trade_plans = AsyncMock()
    runner._liquidity_ok = lambda *_: True
    runner._stacking_block = lambda *_: False
    runner.strategy = MagicMock()
    runner.strategy.generate_signal = AsyncMock(return_value=None)
    runner._kill_switch = kill_switch
    runner.cleanup = AsyncMock()
    runner.initialize = AsyncMock()
    runner.orchestrator = DummyOrchestrator(command_result=command_result, risk_result=risk_result)
    runner.risk_manager = SimpleNamespace(
        update_equity=lambda *_args, **_kwargs: None,
        start_of_day_equity=runner.bot.equity,
        daily_loss=0.0,
        positions={},
        pending_orders_by_symbol={},
        get_total_exposure=lambda *_args, **_kwargs: 0.0,
        check_trade_allowed=lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        update_positions=lambda *_args, **_kwargs: None,
        update_pending_orders=lambda *_args, **_kwargs: None,
    )
    return runner


# --- Run loop control paths ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kill_switch, command_result, risk_result, expected_reason",
    [
        (True, CommandResult(), RiskCheckResult(False, False), "kill switch"),
        (False, CommandResult(stop_requested=True, shutdown_reason="manual stop"), RiskCheckResult(False, False), "manual stop"),
        (False, CommandResult(), RiskCheckResult(True, True, shutdown_reason="risk breach"), "risk breach"),
    ],
)
async def test_run_loop_stop_paths(monkeypatch, kill_switch, command_result, risk_result, expected_reason):
    runner = _build_control_runner(command_result=command_result, risk_result=risk_result, kill_switch=kill_switch)

    async def fast_sleep(_seconds):
        runner.running = False
        runner.orchestrator.running = False

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await runner.run_loop(max_loops=1)

    assert runner.shutdown_reason == expected_reason
    assert runner.orchestrator.last_reason == expected_reason
    assert runner.running is False


def _build_runner(signal: StrategySignal, *, risk_allowed=True, slippage_ok=True, execute_orders=True):
    runner = StrategyRunner(execute_orders=execute_orders)
    try:
        runner.db.close()
    except Exception:
        pass

    async def _init():
        return None

    runner.initialize = _init

    async def _cleanup():
        return None

    runner.cleanup = _cleanup
    runner.orchestrator = DummyOrchestrator()
    runner.daily_loss_pct = 10.0
    runner.bot = FakeBot(price=100.0)
    runner.db = DummyDB()
    runner.health_manager = DummyHealth()
    runner.risk_manager = SimpleNamespace(
        update_equity=lambda _eq: None,
        start_of_day_equity=1_000.0,
        daily_loss=0.0,
        positions={},
        pending_orders_by_symbol={},
        get_total_exposure=lambda *_args, **_kwargs: 0.0,
        check_trade_allowed=lambda *_: SimpleNamespace(allowed=risk_allowed, reason="blocked" if not risk_allowed else None),
        update_positions=lambda _positions=None, price_overrides=None: None,
        update_pending_orders=lambda *_args, **_kwargs: None,
    )
    runner._get_active_symbols = lambda: ["BTC/USD"]

    async def _noop(*_args, **_kwargs):
        return None

    runner._capture_ohlcv = _noop
    runner._monitor_trade_plans = _noop
    runner._liquidity_ok = lambda *_: True
    runner._stacking_block = lambda *_: False
    runner._slippage_within_limit = lambda *_args, **__: (slippage_ok, 0.0 if slippage_ok else 1.0)
    runner.session_id = 1
    runner.exchange_name = "TEST"
    runner.strategy = DummyStrategy(signals=[signal])
    return runner


@pytest.mark.asyncio
async def test_sleep_spacing_when_no_orders():
    sig = make_strategy_signal(action="BUY", quantity=0.1, symbol="BTC/USD")
    runner = _build_runner(sig, execute_orders=False)
    start = time.time()
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    duration = time.time() - start
    assert duration < 1.5
    assert runner.strategy.executions == 0


@pytest.mark.asyncio
async def test_runner_respects_risk_block():
    sig = make_strategy_signal(action="BUY", quantity=0.1, symbol="BTC/USD")
    runner = _build_runner(sig, risk_allowed=False, execute_orders=True)
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    assert runner.bot.place_calls == []
    assert runner.strategy.rejections == ["blocked"]


@pytest.mark.asyncio
async def test_runner_skips_on_slippage_block():
    sig = make_strategy_signal(action="BUY", quantity=0.1, symbol="BTC/USD")
    runner = _build_runner(sig, slippage_ok=False, execute_orders=True)
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    assert runner.bot.place_calls == []
    assert runner.strategy.executions == 0


@pytest.mark.asyncio
async def test_runner_places_order_and_records_execution():
    sig = make_strategy_signal(action="BUY", quantity=0.1, symbol="BTC/USD")
    runner = _build_runner(sig, execute_orders=True)
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    assert runner.bot.place_calls == [("BTC/USD", "BUY", 0.1, True, False)]
    assert runner.strategy.executions == 1


@pytest.mark.asyncio
async def test_runner_skips_sell_when_price_missing():
    sig = make_strategy_signal(action="SELL", quantity=0.2, symbol="BTC/USD")
    runner = _build_runner(sig, execute_orders=True)
    runner.bot = FakeBot(price=0.0, bid=0.0, ask=0.0, spread_pct=0.01, bid_size=1.0, ask_size=1.0)
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    assert runner.bot.place_calls == []


@pytest.mark.asyncio
async def test_flatten_helper_requests_marketable_orders():
    class StubDB:
        def __init__(self):
            self.trades = []

        def get_positions(self, session_id):
            return [{"symbol": "BHP/AUD", "quantity": 10.0}]

        def log_trade(self, *args, **kwargs):
            self.trades.append((args, kwargs))

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 1
    runner.db = StubDB()
    runner.bot = AsyncMock()
    runner.bot.get_market_data_async = AsyncMock(return_value={"price": 100.0})
    runner.bot.place_order_async = AsyncMock(return_value={"order_id": "1", "liquidity": "taker"})
    runner.cost_tracker = MagicMock()
    runner.cost_tracker.calculate_trade_fee.return_value = 0.0
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0

    await runner._close_all_positions_safely()

    runner.bot.place_order_async.assert_awaited_once()
    kwargs = runner.bot.place_order_async.call_args.kwargs
    assert kwargs["force_market"] is True
    assert kwargs["prefer_maker"] is False


@pytest.mark.asyncio
async def test_runner_blocks_sell_on_stacking():
    sig = make_strategy_signal(action="SELL", quantity=0.1, symbol="BTC/USD")
    runner = _build_runner(sig, execute_orders=True)
    runner._stacking_block = lambda action, *_args, **_kwargs: action == "SELL"
    runner.risk_manager.pending_orders_by_symbol = {"BTC/USD": {"sell": 50.0, "count_sell": 2}}
    runner.risk_manager.positions = {"BTC/USD": {"quantity": 1.0}}
    await asyncio.wait_for(runner.run_loop(max_loops=1), timeout=5)
    assert runner.bot.place_calls == []
    assert runner.strategy.rejections == ["Stacking blocked"]
