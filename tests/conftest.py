"""Test session fixtures.

Ensures database writes during tests go to an isolated file instead of the
production `trading.db`.
"""

import atexit
import os
import tempfile

# Flag that we are under pytest so logger_config can direct logs to test files
os.environ.setdefault("PYTEST_RUNNING", "1")
# Test-safe defaults to avoid long sleeps and hard stops during pytest
os.environ.setdefault("TRADING_MODE", "PAPER")
os.environ.setdefault("LOOP_INTERVAL_SECONDS", "1")
os.environ.setdefault("EXCHANGE_PAUSE_SECONDS", "1")
os.environ.setdefault("TOOL_PAUSE_SECONDS", "1")

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
