import pytest

from trader_bot.strategy_runner import StrategyRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "fresh.db"))
    r = StrategyRunner(execute_orders=False)
    # Keep health manager in sync with deterministic monotonic for tests
    return r


def test_stale_due_to_latency(runner):
    runner.health_manager.ticker_max_latency_ms = 1000
    runner.health_manager.monotonic = lambda: 10.0
    stale, detail = runner.health_manager.is_stale_market_data(
        {"price": 100, "_latency_ms": 1500, "_fetched_monotonic": 9.0}
    )
    assert stale is True
    assert detail.get("reason") == "latency"


def test_stale_due_to_age(runner):
    runner.health_manager.ticker_max_age_ms = 4000
    runner.health_manager.monotonic = lambda: 100.0
    data = {"price": 100, "_fetched_monotonic": 95.0, "_latency_ms": 100}
    stale, detail = runner.health_manager.is_stale_market_data(data)
    assert stale is True
    assert detail.get("reason") == "age"


def test_fresh_market_data(runner):
    runner.health_manager.ticker_max_age_ms = 5000
    runner.health_manager.ticker_max_latency_ms = 2000
    runner.health_manager.monotonic = lambda: 50.0
    data = {"price": 100, "_fetched_monotonic": 49.7, "_latency_ms": 500}
    stale, detail = runner.health_manager.is_stale_market_data(data)
    assert stale is False
    assert detail.get("reason") is None
