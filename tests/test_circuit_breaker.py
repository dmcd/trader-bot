import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader_bot.strategy_runner import StrategyRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    db_path = tmp_path / "cb.db"
    monkeypatch.setenv("TRADING_DB_PATH", str(db_path))
    r = StrategyRunner(execute_orders=False)
    r._monotonic = lambda: 100.0
    r.health_manager.monotonic = r._monotonic
    return r


def test_exchange_circuit_breaker_trips_and_resets(runner):
    runner.health_manager.exchange_error_threshold = 2
    runner.health_manager.exchange_pause_seconds = 30

    runner.health_manager.record_exchange_failure("get_equity", "boom")
    assert runner.health_manager.pause_until is None

    runner.health_manager.record_exchange_failure("ticker", "boom2")
    assert runner.health_manager.pause_until == pytest.approx(130.0)

    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("exchange_circuit") == "tripped"

    runner.health_manager.reset_exchange_errors()
    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("exchange_circuit") == "ok"


def test_tool_circuit_breaker_trips_and_recovers(runner):
    runner.health_manager.tool_error_threshold = 1
    runner.health_manager.tool_pause_seconds = 20
    runner._monotonic = lambda: 50.0
    runner.health_manager.monotonic = runner._monotonic

    runner.health_manager.record_tool_failure(context="get_market_data", error="oops")
    assert runner.health_manager.pause_until == pytest.approx(70.0)

    states = {row["key"]: row["value"] for row in runner.db.get_health_state()}
    assert states.get("tool_circuit") == "tripped"

    runner.health_manager.record_tool_success()
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


def test_operational_metrics_emit_health_state(runner):
    runner.session_id = runner.db.get_or_create_session(starting_balance=1000.0, bot_version="metric-test")
    runner.session = runner.db.get_session(runner.session_id)
    runner.risk_manager.start_of_day_equity = 1000.0
    runner.risk_manager.daily_loss = 50.0
    runner.session_stats = {"gross_pnl": 100.0, "total_fees": 10.0, "total_llm_cost": 2.0}
    runner.cost_tracker = MagicMock()
    runner.cost_tracker.calculate_llm_burn.return_value = {
        "remaining_budget": 8.0,
        "pct_of_budget": 0.2,
        "total_llm_cost": 2.0,
    }
    runner.daily_loss_pct = 10.0

    runner._record_operational_metrics(current_exposure=250.0, current_equity=950.0)

    health = {row["key"]: row for row in runner.db.get_health_state()}
    risk_detail = json.loads(health["risk_metrics"]["detail"])
    assert risk_detail["exposure"] == 250.0
    assert risk_detail["daily_loss"] == 50.0
    budget_detail = json.loads(health["llm_budget"]["detail"])
    assert budget_detail["total_llm_cost"] == 2.0


def test_equity_sanity_health_state(runner):
    runner.session_id = runner.db.get_or_create_session(starting_balance=1000.0, bot_version="eq-test")
    runner.session = runner.db.get_session(runner.session_id)
    runner.session_stats = {"gross_pnl": 0.0, "total_fees": 0.0, "total_llm_cost": 0.0}

    runner._sanity_check_equity_vs_stats(current_equity=800.0)

    health = {row["key"]: row for row in runner.db.get_health_state()}
    detail = json.loads(health["equity_sanity"]["detail"])
    assert health["equity_sanity"]["value"] == "mismatch"
    assert detail["equity_net"] == -200.0
