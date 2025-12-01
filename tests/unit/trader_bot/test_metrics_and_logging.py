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
    def __init__(self, portfolio_marker=_DEFAULT, stats_marker=_DEFAULT, equity_marker=_DEFAULT, should_fail_log=False):
        self.portfolio = {"id": 1} if portfolio_marker is _DEFAULT else portfolio_marker
        self.stats = {"net_pnl": -10.0} if stats_marker is _DEFAULT else stats_marker
        self.equity = 120.0 if equity_marker is _DEFAULT else equity_marker
        self.should_fail_log = should_fail_log
        self.log_calls = 0
        self.conn = self

    def cursor(self):
        return self

    def execute(self, *_args, **_kwargs):
        return self

    def fetchone(self):
        if self.equity is None:
            return None
        return {"equity": self.equity, "timestamp": "2024-01-01T00:00:00Z"}

    def get_first_equity_snapshot_for_portfolio(self, portfolio_id):
        return self.fetchone()

    def get_portfolio(self, portfolio_id):
        return self.portfolio

    def get_portfolio_stats(self, portfolio_id):
        return self.stats or {}

    def get_latest_equity_for_portfolio(self, portfolio_id):
        return self.equity

    def log_llm_call_for_portfolio(self, portfolio_id, *args, **kwargs):
        self.log_calls += 1
        if self.should_fail_log:
            raise RuntimeError("log failed")


def test_metrics_drift_flags_threshold_and_logs():
    db = StubDB()
    drift = MetricsDrift(portfolio_id=1, db=db).check_drift(threshold_pct=5.0)

    assert drift["reference_equity"] == 110.0
    assert drift["latest_equity"] == 120.0
    assert drift["exceeded"] is True
    assert drift["baseline_timestamp"].startswith("2024-01-01")
    assert db.log_calls == 1


def test_metrics_drift_handles_missing_rows():
    missing_portfolio_db = StubDB(portfolio_marker=None)
    with pytest.raises(ValueError):
        MetricsDrift(portfolio_id=2, db=missing_portfolio_db).check_drift()

    missing_equity_db = StubDB(equity_marker=None)
    with pytest.raises(ValueError):
        MetricsDrift(portfolio_id=3, db=missing_equity_db).check_drift()


def test_metrics_drift_swallows_logging_failures():
    db = StubDB(should_fail_log=True)
    drift = MetricsDrift(portfolio_id=4, db=db).check_drift(threshold_pct=200.0)

    assert drift["exceeded"] is False
    assert db.log_calls == 1


@pytest.fixture
def metrics_db(test_db_path):
    db = TradingDatabase(db_path=str(test_db_path))
    portfolio_id, _ = db.ensure_active_portfolio(name="metrics-test", bot_version="test")
    db.log_equity_snapshot_for_portfolio(portfolio_id, 1000.0)
    db.log_equity_snapshot_for_portfolio(portfolio_id, 1010.0)
    db.set_portfolio_stats_cache(portfolio_id, {"gross_pnl": 10.0, "total_fees": 0.0, "total_llm_cost": 0.0})
    try:
        yield db, portfolio_id
    finally:
        db.close()


def test_no_drift_within_threshold(metrics_db):
    db, portfolio_id = metrics_db
    validator = MetricsDrift(portfolio_id, db=db)
    result = validator.check_drift(threshold_pct=2.0)
    assert result["exceeded"] is False
    assert result["drift"] == pytest.approx(0.0)


def test_detects_drift_beyond_threshold(metrics_db):
    db, portfolio_id = metrics_db
    db.log_equity_snapshot_for_portfolio(portfolio_id, 1200.0)
    validator = MetricsDrift(portfolio_id, db=db)
    result = validator.check_drift(threshold_pct=1.0)
    assert result["exceeded"] is True
    assert result["drift_pct"] > 0
