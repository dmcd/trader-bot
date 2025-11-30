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

# Flag that we are under pytest so logger_config can direct logs to test files
os.environ.setdefault("PYTEST_RUNNING", "1")
# Test-safe defaults to avoid long sleeps and hard stops during pytest
os.environ.setdefault("TRADING_MODE", "PAPER")
os.environ.setdefault("LOOP_INTERVAL_SECONDS", "1")
os.environ.setdefault("EXCHANGE_PAUSE_SECONDS", "1")
os.environ.setdefault("TOOL_PAUSE_SECONDS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore::ResourceWarning")
warnings.simplefilter("ignore", ResourceWarning)

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
