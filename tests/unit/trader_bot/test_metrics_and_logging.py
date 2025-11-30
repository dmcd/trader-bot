import logging
import sys
import types
from pathlib import Path

import pytest

from trader_bot.database import TradingDatabase
from trader_bot.logger_config import LoggerWriter, setup_logging
from trader_bot.metrics_validator import MetricsDrift


_DEFAULT = object()


def _reset_logging_streams(stdout, stderr):
    logging.getLogger().handlers.clear()
    logging.getLogger("bot_actions").handlers.clear()
    logging.getLogger("telemetry").handlers.clear()
    sys.stdout = stdout
    sys.stderr = stderr


def test_setup_logging_respects_test_mode(tmp_path, monkeypatch):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    monkeypatch.setenv("PYTEST_RUNNING", "1")
    monkeypatch.chdir(tmp_path)

    try:
        bot_logger = setup_logging()
        root_handlers = logging.getLogger().handlers
        file_handlers = [h for h in root_handlers if isinstance(h, logging.FileHandler)]
        filenames = {Path(h.baseFilename).name for h in file_handlers}

        assert "console_test.log" in filenames
        assert any(isinstance(h, logging.StreamHandler) and h.level == logging.DEBUG for h in root_handlers)
        assert isinstance(sys.stdout, LoggerWriter)
        assert bot_logger.handlers and isinstance(bot_logger.handlers[0], logging.FileHandler)
        bot_log_path = Path(bot_logger.handlers[0].baseFilename)
        assert bot_log_path.name == "bot_test.log"
        assert bot_log_path.parent.name == "test"
    finally:
        _reset_logging_streams(original_stdout, original_stderr)


def test_setup_logging_defaults_without_test_env(tmp_path, monkeypatch):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    monkeypatch.delenv("PYTEST_RUNNING", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    fake_sys = types.SimpleNamespace(stdout=original_stdout, stderr=original_stderr, modules={})
    monkeypatch.setattr("trader_bot.logger_config.sys", fake_sys)
    monkeypatch.chdir(tmp_path)

    try:
        bot_logger = setup_logging()
        root_handlers = logging.getLogger().handlers
        file_handlers = [h for h in root_handlers if isinstance(h, logging.FileHandler)]
        filenames = {Path(h.baseFilename).name for h in file_handlers}

        assert "console.log" in filenames
        assert any(isinstance(h, logging.StreamHandler) and h.level == logging.INFO for h in root_handlers)
        assert isinstance(fake_sys.stdout, LoggerWriter)

        bot_log_path = Path(bot_logger.handlers[0].baseFilename)
        assert bot_log_path.name == "bot.log"
        assert bot_log_path.parent == tmp_path

        telemetry_handlers = logging.getLogger("telemetry").handlers
        telemetry_filenames = {Path(h.baseFilename).name for h in telemetry_handlers if isinstance(h, logging.FileHandler)}
        assert "telemetry.log" in telemetry_filenames
    finally:
        _reset_logging_streams(original_stdout, original_stderr)


class StubDB:
    def __init__(self, session_marker=_DEFAULT, equity_marker=_DEFAULT, should_fail_log=False):
        self.session = (
            {"starting_balance": 100.0, "net_pnl": -10.0}
            if session_marker is _DEFAULT
            else session_marker
        )
        self.equity = 120.0 if equity_marker is _DEFAULT else equity_marker
        self.should_fail_log = should_fail_log
        self.log_calls = 0

    def get_session(self, session_id):
        return self.session

    def get_latest_equity(self, session_id):
        return self.equity

    def log_llm_call(self, session_id, *args, **kwargs):
        self.log_calls += 1
        if self.should_fail_log:
            raise RuntimeError("log failed")


def test_metrics_drift_flags_threshold_and_logs():
    db = StubDB()
    drift = MetricsDrift(session_id=1, db=db).check_drift(threshold_pct=5.0)

    assert drift["reference_equity"] == 90.0
    assert drift["latest_equity"] == 120.0
    assert drift["exceeded"] is True
    assert db.log_calls == 1


def test_metrics_drift_handles_missing_rows():
    missing_session_db = StubDB(session_marker=None)
    with pytest.raises(ValueError):
        MetricsDrift(session_id=2, db=missing_session_db).check_drift()

    missing_equity_db = StubDB(equity_marker=None)
    with pytest.raises(ValueError):
        MetricsDrift(session_id=3, db=missing_equity_db).check_drift()


def test_metrics_drift_swallows_logging_failures():
    db = StubDB(should_fail_log=True)
    drift = MetricsDrift(session_id=4, db=db).check_drift(threshold_pct=200.0)

    assert drift["exceeded"] is False
    assert db.log_calls == 1


@pytest.fixture
def metrics_db(test_db_path):
    db = TradingDatabase(db_path=str(test_db_path))
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="test")
    db.log_equity_snapshot(session_id, 1010.0)
    db.update_session_totals(session_id, net_pnl=10.0)
    try:
        yield db, session_id
    finally:
        db.close()


def test_no_drift_within_threshold(metrics_db):
    db, session_id = metrics_db
    validator = MetricsDrift(session_id, db=db)
    result = validator.check_drift(threshold_pct=2.0)
    assert result["exceeded"] is False
    assert result["drift"] == pytest.approx(0.0)


def test_detects_drift_beyond_threshold(metrics_db):
    db, session_id = metrics_db
    db.log_equity_snapshot(session_id, 1200.0)
    validator = MetricsDrift(session_id, db=db)
    result = validator.check_drift(threshold_pct=1.0)
    assert result["exceeded"] is True
    assert result["drift_pct"] > 0
