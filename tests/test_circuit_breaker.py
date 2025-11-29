from unittest.mock import AsyncMock

import pytest

from trader_bot.strategy_runner import StrategyRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    db_path = tmp_path / "cb.db"
    monkeypatch.setenv("TRADING_DB_PATH", str(db_path))
    r = StrategyRunner(execute_orders=False)
    r._monotonic = lambda: 100.0
    return r


def test_exchange_circuit_breaker_trips_and_resets(runner):
    runner.exchange_error_threshold = 2
    runner.exchange_pause_seconds = 30

    runner._record_exchange_failure("get_equity", "boom")
    assert runner._pause_until is None

    runner._record_exchange_failure("ticker", "boom2")
    assert runner._pause_until == pytest.approx(130.0)

    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("exchange_circuit") == "tripped"

    runner._reset_exchange_errors()
    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("exchange_circuit") == "ok"


def test_tool_circuit_breaker_trips_and_recovers(runner):
    runner.tool_error_threshold = 1
    runner.tool_pause_seconds = 20
    runner._monotonic = lambda: 50.0

    runner._record_tool_failure(context="get_market_data", error="oops")
    assert runner._pause_until == pytest.approx(70.0)

    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("tool_circuit") == "tripped"

    runner._record_tool_success()
    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("tool_circuit") == "ok"


@pytest.mark.asyncio
async def test_kill_switch_stops_loop(runner):
    runner.initialize = AsyncMock()
    runner.cleanup = AsyncMock()
    runner._kill_switch = True

    await runner.run_loop(max_loops=1)

    runner.initialize.assert_awaited()
    runner.cleanup.assert_awaited()
    assert runner.running is False
    assert runner.shutdown_reason == "kill switch"
