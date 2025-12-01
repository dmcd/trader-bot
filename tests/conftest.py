"""Test session fixtures.

Ensures database writes during tests go to an isolated file instead of the
production `trading.db`.
"""

import atexit
import logging
import os
import tempfile
import warnings
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
import asyncio
from types import SimpleNamespace

# Flag that we are under pytest so logger_config can direct logs to test files
os.environ.setdefault("PYTEST_RUNNING", "1")
# Test-safe defaults to avoid long sleeps and hard stops during pytest
os.environ.setdefault("TRADING_MODE", "PAPER")
os.environ.setdefault("LOOP_INTERVAL_SECONDS", "1")
os.environ.setdefault("EXCHANGE_PAUSE_SECONDS", "1")
os.environ.setdefault("TOOL_PAUSE_SECONDS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore::ResourceWarning")
warnings.simplefilter("ignore", ResourceWarning)

# Ensure config picks up the short test cadence even if imported early
import trader_bot.config as _config
_config.LOOP_INTERVAL_SECONDS = int(os.environ.get("LOOP_INTERVAL_SECONDS", "1"))

from trader_bot.strategy import LLMStrategy
from tests.fakes import FakeBot, FakeExchange

_original_showwarning = warnings.showwarning
_original_warn = warnings.warn


def _suppress_resource_warning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, ResourceWarning):
        return
    return _original_showwarning(message, category, filename, lineno, file=file, line=line)


warnings.showwarning = _suppress_resource_warning


def _suppress_resource_warn(*args, **kwargs):
    category = kwargs.get("category")
    if category is None and len(args) >= 2:
        category = args[1]
    if category and issubclass(category, ResourceWarning):
        return
    return _original_warn(*args, **kwargs)


warnings.warn = _suppress_resource_warn


def pytest_configure(config):
    warnings.simplefilter("ignore", ResourceWarning)


@atexit.register
def _silence_resource_warnings_on_exit():
    warnings.filterwarnings("ignore", category=ResourceWarning)

_cleanup_target = None


def _ensure_test_db_path():
    global _cleanup_target

    if os.environ.get("TRADING_DB_PATH"):
        return os.environ["TRADING_DB_PATH"]

    fd, path = tempfile.mkstemp(prefix="trader-bot-test-", suffix=".db")
    os.close(fd)
    os.environ["TRADING_DB_PATH"] = path
    _cleanup_target = path
    return path


TEST_DB_PATH = _ensure_test_db_path()


@atexit.register
def _remove_temp_db():
    if _cleanup_target and os.path.exists(_cleanup_target):
        try:
            os.remove(_cleanup_target)
        except OSError:
            pass


@pytest.fixture
def test_db_path(tmp_path, monkeypatch):
    """Provide an isolated DB path and set TRADING_DB_PATH for the test."""
    path = tmp_path / "trader-bot-test.db"
    monkeypatch.setenv("TRADING_DB_PATH", str(path))
    return path


@pytest.fixture
def fake_logger():
    """Shared lightweight logger mock for tests."""
    logger = MagicMock(spec=logging.Logger)
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest_asyncio.fixture
async def set_loop_time(monkeypatch):
    """Override asyncio loop time for deterministic cooldown checks."""
    loop = asyncio.get_running_loop()
    current_time = loop.time()

    def fake_time():
        return current_time

    monkeypatch.setattr(loop, "time", fake_time, raising=False)

    def _set(ts: float):
        nonlocal current_time
        current_time = ts

    return _set


@pytest.fixture
def strategy_env(monkeypatch):
    """Lightweight strategy harness with mocked dependencies."""
    monkeypatch.setenv("ALLOW_UNKEYED_LLM", "true")
    for attr in ("_prompt_template_cache", "_system_prompt_cache"):
        if hasattr(LLMStrategy, attr):
            delattr(LLMStrategy, attr)

    db = MagicMock()
    portfolio_id = 1
    db.get_portfolio_stats.return_value = {"gross_pnl": 0, "total_fees": 0, "total_llm_cost": 0}
    db.get_recent_market_data_for_portfolio.return_value = [{"price": 100}] * 50
    db.get_recent_ohlcv_for_portfolio.return_value = [{"close": 100, "volume": 1}] * 2
    db.get_open_orders_for_portfolio.return_value = []
    db.get_open_trade_plans_for_portfolio.return_value = []
    db.log_llm_trace_for_portfolio.return_value = 42
    db.portfolio_id = portfolio_id

    ta = MagicMock()
    ta.calculate_indicators.return_value = {"bb_width": 2.0, "rsi": 50}
    cost = MagicMock()
    cost.calculate_llm_burn.return_value = {
        "total_llm_cost": 0.0,
        "budget": 1.0,
        "pct_of_budget": 0.0,
        "burn_rate_per_hour": 0.0,
        "remaining_budget": 1.0,
        "hours_to_cap": None,
    }

    strategy = LLMStrategy(db, ta, cost, portfolio_id=portfolio_id, run_id="test-run")
    return SimpleNamespace(db=db, ta=ta, cost=cost, strategy=strategy, portfolio_id=portfolio_id)


@pytest.fixture
def fake_bot():
    """Default fake bot instance for tests that only need simple responses."""
    return FakeBot()


@pytest.fixture
def fake_bot_factory():
    """Factory for fake bots with per-test configuration."""
    def _factory(**overrides):
        return FakeBot(**overrides)

    return _factory


@pytest.fixture
def fake_exchange_factory():
    """Factory for ccxt-like exchange doubles."""
    def _factory(**overrides):
        return FakeExchange(**overrides)

    return _factory
