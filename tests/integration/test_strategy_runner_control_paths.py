import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from unittest.mock import ANY, AsyncMock, MagicMock

from trader_bot.services.command_processor import CommandResult
from trader_bot.services.strategy_orchestrator import RiskCheckResult
from trader_bot.strategy_runner import StrategyRunner
from tests.fakes import FakeBot

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("test_db_path")]


class DummyOrchestrator:
    def __init__(self, command_result=None, risk_result=None):
        self.command_result = command_result or CommandResult()
        self.risk_result = risk_result or RiskCheckResult(False, False)
        self.running = True
        self.last_reason = None
        self.cleanup_called = False
        self.metrics = []

    async def start(self, initialize_cb):
        # Skip expensive initialize; just flag as running
        self.running = True

    def request_stop(self, shutdown_reason=None):
        self.last_reason = shutdown_reason
        self.running = False
        return shutdown_reason

    async def process_commands(self, *_args, **_kwargs):
        return self.command_result

    async def enforce_risk_budget(self, *_args, **_kwargs):
        return self.risk_result

    def emit_market_health(self, primary_data):
        return True, {"age": primary_data.get("_latency_ms", 0) if primary_data else 0}

    def emit_operational_metrics(self, current_exposure, current_equity):
        self.metrics.append((current_exposure, current_equity))

    async def monitor_trade_plans(self, *_args, **_kwargs):
        return None

    async def cleanup(self, cleanup_cb):
        self.cleanup_called = True
        # Avoid running full runner cleanup in unit tests
        return None


class DummyDB:
    def __init__(self):
        self.replaced_open_orders = 0

    def log_equity_snapshot(self, *_args, **_kwargs):
        return None

    def log_market_data(self, *_args, **_kwargs):
        return None

    def get_pending_commands(self):
        return []

    def get_positions(self, *_args, **_kwargs):
        return []

    def replace_positions(self, *_args, **_kwargs):
        return None

    def get_open_orders(self, *_args, **_kwargs):
        return []

    def replace_open_orders(self, *_args, **_kwargs):
        self.replaced_open_orders += 1
        return None

    def prune_market_data(self, *_args, **_kwargs):
        return None

    def get_recent_market_data(self, *_args, **_kwargs):
        return []

    def get_open_trade_plans(self, *_args, **_kwargs):
        return []

    def count_open_trade_plans_for_symbol(self, *_args, **_kwargs):
        return 0

    def get_trade_count(self, *_args, **_kwargs):
        return 0


class DummyRiskManager:
    def __init__(self):
        self.current_equity = 0.0
        self.positions = {}
        self.pending_orders_by_symbol = {}

    def update_equity(self, *_args, **_kwargs):
        if _args:
            try:
                self.current_equity = float(_args[0])
            except Exception:
                pass

    def update_positions(self, *_args, **_kwargs):
        return None

    def update_pending_orders(self, *_args, **_kwargs):
        return None

    def get_total_exposure(self, *_args, **_kwargs):
        return 0.0

    def check_trade_allowed(self, *_args, **_kwargs):
        return SimpleNamespace(allowed=True, reason=None)


@pytest.mark.asyncio
async def test_equity_fetch_failure_short_circuits(monkeypatch):
    runner = StrategyRunner(execute_orders=False)
    runner.bot = FakeBot(price=101.0, spread_pct=0.5)
    runner.db = DummyDB()
    runner.risk_manager = DummyRiskManager()
    runner.orchestrator = DummyOrchestrator()
    runner.health_manager.record_exchange_failure = MagicMock()

    async def boom():
        raise RuntimeError("equity unavailable")

    runner.bot.get_equity_async = boom

    async def fast_sleep(_seconds):
        runner.running = False
        runner.orchestrator.running = False

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await runner.run_loop(max_loops=1)

    runner.health_manager.record_exchange_failure.assert_called_with("get_equity_async", ANY)
    assert runner.orchestrator.cleanup_called


def test_get_sync_symbols_dedupes_and_fallback():
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 7
    runner.db = MagicMock()
    runner.db.get_distinct_trade_symbols.return_value = ["BTC/USD", "BTC/USD", None]
    runner.db.get_positions.return_value = [{"symbol": "ETH/USD"}, {"symbol": "USD"}]
    runner.db.get_open_trade_plans.return_value = [{"symbol": "ETH/USD"}, {"symbol": ""}]
    runner.db.get_open_orders.return_value = [{"symbol": "SOL/USD"}]

    symbols = runner._get_sync_symbols()

    assert symbols == {"BTC/USD", "ETH/USD", "SOL/USD"}

    runner.db.get_distinct_trade_symbols.return_value = []
    runner.db.get_positions.return_value = []
    runner.db.get_open_trade_plans.return_value = []
    runner.db.get_open_orders.return_value = []

    assert runner._get_sync_symbols() == {"BTC/USD"}


def test_record_operational_metrics_emits_budget_branches():
    runner = StrategyRunner(execute_orders=False)
    runner.session = {"created_at": datetime.now(timezone.utc)}
    runner.session_stats = {"gross_pnl": 100.0, "total_fees": 50.0, "total_llm_cost": 2.0}
    runner.risk_manager = DummyRiskManager()
    runner._record_health_state = MagicMock()
    runner.cost_tracker = MagicMock()
    runner.cost_tracker.calculate_llm_burn.side_effect = [
        {"remaining_budget": 0.0, "pct_of_budget": 1.1},
        {"remaining_budget": 5.0, "pct_of_budget": 0.85},
    ]

    runner._record_operational_metrics(current_exposure=500.0, current_equity=1200.0)
    runner._record_operational_metrics(current_exposure=200.0, current_equity=1100.0)

    llm_statuses = [call.args[1] for call in runner._record_health_state.call_args_list if call.args[0] == "llm_budget"]
    assert "cap_hit" in llm_statuses
    assert "near_cap" in llm_statuses
    risk_calls = [call for call in runner._record_health_state.call_args_list if call.args[0] == "risk_metrics"]
    assert risk_calls
    risk_detail = risk_calls[0].args[2]
    assert risk_detail["net_pnl"] == pytest.approx(48.0)


@pytest.mark.asyncio
async def test_run_loop_pauses_when_health_manager_requests(monkeypatch):
    runner = StrategyRunner(execute_orders=False)
    runner.bot = FakeBot(price=101.0, spread_pct=0.5)
    runner.db = DummyDB()
    runner.risk_manager = DummyRiskManager()
    runner.session_id = 1
    runner._get_active_symbols = lambda: ["BTC/USD"]
    runner._capture_ohlcv = AsyncMock()
    runner.health_manager.should_pause = MagicMock(return_value=True)
    runner.health_manager.pause_remaining = MagicMock(return_value=30.0)
    runner.orchestrator = DummyOrchestrator()
    runner.cleanup = AsyncMock()
    runner.bot.get_market_data_async = AsyncMock()

    async def fast_sleep(_seconds):
        runner.running = False
        runner.orchestrator.running = False

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await runner.run_loop(max_loops=1)

    runner.health_manager.pause_remaining.assert_called()
    runner.bot.get_market_data_async.assert_not_called()


@pytest.mark.asyncio
async def test_market_health_gating_records_stale(monkeypatch):
    runner = StrategyRunner(execute_orders=False)
    runner.bot = FakeBot(price=101.0, spread_pct=0.5)
    runner.db = DummyDB()
    runner.risk_manager = DummyRiskManager()
    runner.session_id = 1
    runner._record_health_state = MagicMock()
    runner._capture_ohlcv = AsyncMock()
    runner._get_active_symbols = lambda: ["BTC/USD"]
    runner.health_manager.should_pause = MagicMock(return_value=False)
    runner.health_manager.pause_remaining = MagicMock(return_value=0)
    runner.health_manager.record_exchange_failure = MagicMock()
    runner.orchestrator = DummyOrchestrator()
    runner.orchestrator.emit_market_health = MagicMock(return_value=(False, {"age": 999}))
    runner.cleanup = AsyncMock()

    async def fast_sleep(_seconds):
        runner.running = False
        runner.orchestrator.running = False

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await runner.run_loop(max_loops=1)

    runner._record_health_state.assert_any_call("market_data", "stale", {"age": 999})


@pytest.mark.asyncio
async def test_cancel_action_refreshes_open_orders(monkeypatch):
    runner = StrategyRunner(execute_orders=False)
    runner.bot = FakeBot(price=101.0, spread_pct=0.5)
    runner.db = DummyDB()
    runner.risk_manager = DummyRiskManager()
    runner.session_id = 1
    runner._capture_ohlcv = AsyncMock()
    runner._get_active_symbols = lambda: ["BTC/USD"]
    runner._liquidity_ok = lambda _md: True
    runner.health_manager.should_pause = MagicMock(return_value=False)
    runner.health_manager.pause_remaining = MagicMock(return_value=0)
    runner.orchestrator = DummyOrchestrator()
    runner.orchestrator.emit_market_health = MagicMock(return_value=(True, {}))
    runner.sync_trades_from_exchange = AsyncMock()
    runner._record_health_state = MagicMock()
    runner.strategy = MagicMock()
    signal = SimpleNamespace(action="CANCEL", symbol="BTC/USD", quantity=0.1, reason="cleanup", order_id="123", stop_price=None, target_price=None, trace_id=5)
    runner.strategy.generate_signal = AsyncMock(side_effect=[signal, None])
    runner.cleanup = AsyncMock()

    async def fast_sleep(_seconds):
        runner.running = False
        runner.orchestrator.running = False

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await runner.run_loop(max_loops=1)

    assert runner.bot.cancel_calls == 1
    assert runner.db.replaced_open_orders == 2  # once during loop, once after cancel refresh


@pytest.mark.asyncio
async def test_handle_signal_routes_to_action_handler():
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 42
    runner.action_handler = MagicMock()
    runner.action_handler.handle_update_plan = AsyncMock(return_value={"status": "updated"})
    runner.action_handler.handle_partial_close = AsyncMock(return_value={"status": "partial"})
    runner.action_handler.handle_close_position = AsyncMock(return_value={"status": "closed"})
    runner._emit_telemetry = MagicMock()
    market_data = {"BTC/USD": {"price": 100.0}}

    update_signal = SimpleNamespace(action="UPDATE_PLAN", symbol="BTC/USD", plan_id=1, stop_price=90.0, target_price=120.0, trace_id=7)
    await runner._handle_signal(update_signal, market_data, [], 0.0, 0.0)
    runner.action_handler.handle_update_plan.assert_awaited_with(1, 90.0, 120.0, "Update plan", 7)

    partial_signal = SimpleNamespace(action="PARTIAL_CLOSE", symbol="BTC/USD", plan_id=2, close_fraction=0.25, trace_id=8)
    await runner._handle_signal(partial_signal, market_data, [], 0.0, 1000.0)
    runner.action_handler.handle_partial_close.assert_awaited()

    close_signal = SimpleNamespace(action="CLOSE_POSITION", symbol="BTC/USD", trace_id=9)
    await runner._handle_signal(close_signal, market_data, [], 0.0, 0.0)
    runner.action_handler.handle_close_position.assert_awaited()
