import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from trader_bot.services.command_processor import CommandResult
from trader_bot.services.strategy_orchestrator import StrategyOrchestrator


def _build_orchestrator(record_cb=lambda *_: None, risk_manager=None, health_manager=None, plan_monitor=None, command_processor=None):
    return StrategyOrchestrator(
        command_processor=command_processor or MagicMock(),
        plan_monitor=plan_monitor or MagicMock(),
        risk_manager=risk_manager or MagicMock(),
        health_manager=health_manager or MagicMock(),
        record_operational_metrics=record_cb,
        loop_interval_seconds=0.1,
        logger=MagicMock(),
        actions_logger=MagicMock(),
    )


@pytest.mark.asyncio
async def test_start_and_cleanup_sequence_runs_hooks():
    events = []

    async def initializer():
        events.append("init")

    async def cleanup():
        events.append("cleanup")

    orchestrator = _build_orchestrator()
    await orchestrator.start(initializer)
    assert orchestrator.running is True

    await orchestrator.cleanup(cleanup)
    assert orchestrator.running is False
    assert events == ["init", "cleanup"]


@pytest.mark.asyncio
async def test_enforce_risk_budget_triggers_shutdown_and_kill_switch():
    risk_manager = SimpleNamespace(start_of_day_equity=1000.0, daily_loss=200.0)
    orchestrator = _build_orchestrator(risk_manager=risk_manager)
    close_positions = AsyncMock()
    shutdown_reasons = []

    def set_shutdown_reason(reason):
        shutdown_reasons.append(reason)

    result = await orchestrator.enforce_risk_budget(
        current_equity=750.0,
        daily_loss_pct_limit=10.0,
        max_daily_loss=150.0,
        trading_mode="LIVE",
        close_positions_cb=close_positions,
        set_shutdown_reason=set_shutdown_reason,
    )

    assert result.should_stop is True
    assert result.kill_switch is True
    assert shutdown_reasons == ["daily loss 20.00% > 10.0%"]
    close_positions.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_commands_passthrough_and_stop_request():
    command_processor = MagicMock()
    command_processor.process = AsyncMock(return_value=CommandResult(stop_requested=True, shutdown_reason="manual stop"))
    orchestrator = _build_orchestrator(command_processor=command_processor)

    async def noop():
        return None

    await orchestrator.start(noop)
    result = await orchestrator.process_commands(AsyncMock(), lambda *_: None)
    orchestrator.request_stop(result.shutdown_reason)

    assert orchestrator.running is False
    assert result.shutdown_reason == "manual stop"
    command_processor.process.assert_awaited_once()


def test_emit_health_and_operational_metrics():
    health_manager = MagicMock()
    health_manager.is_stale_market_data.return_value = (False, {"latency_ms": 10})
    metrics = []

    def record_metrics(exposure, equity):
        metrics.append((exposure, equity))

    orchestrator = _build_orchestrator(record_cb=record_metrics, health_manager=health_manager)
    ok, detail = orchestrator.emit_market_health({"price": 100})
    orchestrator.emit_operational_metrics(5.0, 10.0)

    assert ok is True
    assert detail == {"latency_ms": 10}
    assert metrics == [(5.0, 10.0)]
