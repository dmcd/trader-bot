import pytest

from trader_bot.database import TradingDatabase
from trader_bot.metrics_validator import MetricsDrift


@pytest.fixture
def metrics_db(test_db_path):
    db = TradingDatabase()
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
