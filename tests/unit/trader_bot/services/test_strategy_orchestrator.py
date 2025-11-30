import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader_bot.services.command_processor import CommandProcessor, CommandResult
from trader_bot.services.health_manager import HealthCircuitManager
from trader_bot.services.strategy_orchestrator import StrategyOrchestrator


def _build_orchestrator(
    record_cb=lambda *_: None,
    risk_manager=None,
    health_manager=None,
    plan_monitor=None,
    command_processor=None,
    logger=None,
    actions_logger=None,
):
    return StrategyOrchestrator(
        command_processor=command_processor or MagicMock(),
        plan_monitor=plan_monitor or MagicMock(),
        risk_manager=risk_manager or MagicMock(),
        health_manager=health_manager or MagicMock(),
        record_operational_metrics=record_cb,
        loop_interval_seconds=0.1,
        logger=logger or MagicMock(),
        actions_logger=actions_logger or MagicMock(),
    )


class _Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_close_all_positions_command_executes_and_marks():
    db = MagicMock()
    db.get_pending_commands.return_value = [{"id": 1, "command": "CLOSE_ALL_POSITIONS"}]
    db.mark_command_executed = MagicMock()
    close_cb = AsyncMock()
    stop_cb = MagicMock()

    processor = CommandProcessor(db)
    result = await processor.process(close_cb, stop_cb)

    close_cb.assert_awaited_once()
    db.mark_command_executed.assert_called_once_with(1)
    stop_cb.assert_not_called()
    assert result.stop_requested is False


@pytest.mark.asyncio
async def test_stop_bot_command_sets_reason_and_returns_stop():
    db = MagicMock()
    db.get_pending_commands.return_value = [{"id": 3, "command": "STOP_BOT"}]
    db.mark_command_executed = MagicMock()
    close_cb = AsyncMock()
    stop_cb = MagicMock()

    processor = CommandProcessor(db)
    result = await processor.process(close_cb, stop_cb)

    close_cb.assert_not_called()
    db.mark_command_executed.assert_called_once_with(3)
    stop_cb.assert_called_once_with("manual stop")
    assert result.stop_requested is True
    assert result.shutdown_reason == "manual stop"


def test_pause_window_accumulates_longer_durations():
    clock = _Clock(10.0)
    manager = HealthCircuitManager(monotonic=clock)

    first = manager.request_pause(20)
    assert first == pytest.approx(30.0)

    clock.now = 25.0
    second = manager.request_pause(2)
    assert second == pytest.approx(30.0)
    assert manager.should_pause(clock.now) is True
    assert manager.pause_remaining(clock.now) == pytest.approx(5.0)

    clock.now = 31.0
    assert manager.should_pause(clock.now) is False
    assert manager.pause_remaining(clock.now) == 0.0


@pytest.mark.asyncio
async def test_maybe_reconnect_throttles_calls():
    clock = _Clock(0.0)
    manager = HealthCircuitManager(monotonic=clock, reconnect_cooldown_seconds=30, logger=logging.getLogger("test"))
    bot = type("Bot", (), {})()
    bot.connect_async = AsyncMock()

    first = await manager.maybe_reconnect(bot)
    assert first is True
    bot.connect_async.assert_awaited_once()

    clock.now = 10.0
    second = await manager.maybe_reconnect(bot)
    assert second is False
    assert bot.connect_async.call_count == 1

    clock.now = 45.0
    third = await manager.maybe_reconnect(bot)
    assert third is True
    assert bot.connect_async.call_count == 2


@pytest.mark.asyncio
async def test_start_and_cleanup_sequence_runs_hooks(fake_logger):
    events = []

    async def initializer():
        events.append("init")

    async def cleanup():
        events.append("cleanup")

    orchestrator = _build_orchestrator(logger=fake_logger, actions_logger=fake_logger)
    await orchestrator.start(initializer)
    assert orchestrator.running is True

    await orchestrator.cleanup(cleanup)
    assert orchestrator.running is False
    assert events == ["init", "cleanup"]


@pytest.mark.asyncio
async def test_enforce_risk_budget_triggers_shutdown_and_kill_switch(fake_logger):
    risk_manager = SimpleNamespace(start_of_day_equity=1000.0, daily_loss=200.0)
    orchestrator = _build_orchestrator(risk_manager=risk_manager, logger=fake_logger, actions_logger=fake_logger)
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
async def test_process_commands_passthrough_and_stop_request(fake_logger):
    command_processor = MagicMock()
    command_processor.process = AsyncMock(return_value=CommandResult(stop_requested=True, shutdown_reason="manual stop"))
    orchestrator = _build_orchestrator(command_processor=command_processor, logger=fake_logger, actions_logger=fake_logger)

    async def noop():
        return None

    await orchestrator.start(noop)
    result = await orchestrator.process_commands(AsyncMock(), lambda *_: None)
    orchestrator.request_stop(result.shutdown_reason)

    assert orchestrator.running is False
    assert result.shutdown_reason == "manual stop"
    command_processor.process.assert_awaited_once()


def test_emit_health_and_operational_metrics(fake_logger):
    health_manager = MagicMock()
    health_manager.is_stale_market_data.return_value = (False, {"latency_ms": 10})
    metrics = []

    def record_metrics(exposure, equity):
        metrics.append((exposure, equity))

    orchestrator = _build_orchestrator(
        record_cb=record_metrics,
        health_manager=health_manager,
        logger=fake_logger,
        actions_logger=fake_logger,
    )
    ok, detail = orchestrator.emit_market_health({"price": 100})
    orchestrator.emit_operational_metrics(5.0, 10.0)

    assert ok is True
    assert detail == {"latency_ms": 10}
    assert metrics == [(5.0, 10.0)]
