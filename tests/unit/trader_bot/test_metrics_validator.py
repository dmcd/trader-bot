import atexit
import os
import tempfile
import unittest

from trader_bot.database import TradingDatabase
from trader_bot.metrics_validator import MetricsDrift


_fd, _db_path = tempfile.mkstemp(prefix="trader-bot-test-", suffix=".db")
os.close(_fd)
os.environ.setdefault("TRADING_DB_PATH", _db_path)


@atexit.register
def _cleanup_db_path():
    if os.path.exists(_db_path):
        try:
            os.remove(_db_path)
        except OSError:
            pass


class TestMetricsValidator(unittest.TestCase):
    def setUp(self):
        self.db = TradingDatabase()
        # Seed session row and equity snapshot
        self.session_id = self.db.get_or_create_session(starting_balance=1000.0, bot_version="test")
        self.db.log_equity_snapshot(self.session_id, 1010.0)
        self.db.update_session_totals(self.session_id, net_pnl=10.0)

    def test_no_drift_within_threshold(self):
        validator = MetricsDrift(self.session_id, db=self.db)
        result = validator.check_drift(threshold_pct=2.0)
        self.assertFalse(result["exceeded"])
        self.assertAlmostEqual(result["drift"], 0.0)

    def test_detects_drift_beyond_threshold(self):
        self.db.log_equity_snapshot(self.session_id, 1200.0)
        validator = MetricsDrift(self.session_id, db=self.db)
        result = validator.check_drift(threshold_pct=1.0)
        self.assertTrue(result["exceeded"])
        self.assertGreater(result["drift_pct"], 0)


if __name__ == "__main__":
    unittest.main()
