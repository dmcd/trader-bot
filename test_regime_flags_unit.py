import os
import tempfile
import unittest
from unittest.mock import MagicMock

from strategy import LLMStrategy

# Ensure tests never write to the production trading.db
_fd, _db_path = tempfile.mkstemp(prefix="trader-bot-test-", suffix=".db")
os.close(_fd)
os.environ.setdefault("TRADING_DB_PATH", _db_path)


class TestRegimeFlags(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.mock_ta = MagicMock()
        self.mock_cost = MagicMock()
        self.strategy = LLMStrategy(self.mock_db, self.mock_ta, self.mock_cost)

    def test_compute_regime_flags(self):
        one_h_bars = []
        # Build descending timestamps with noticeable volatility and downtrend
        for i in range(12):
            one_h_bars.append({"close": 100 - i * 5})
        market_point = {"spread_pct": 0.1, "bid_size": 2, "ask_size": 3, "bid": 99, "ask": 101}
        flags = self.strategy._compute_regime_flags(
            session_id=1,
            symbol="BTC/USD",
            market_data_point=market_point,
            recent_bars={"1h": one_h_bars},
        )
        self.assertIn("volatility", flags)
        self.assertIn("trend", flags)
        self.assertIn("liquidity", flags)
        self.assertIn("depth", flags)

    def test_build_timeframe_summary(self):
        # Mock DB to return simple OHLCV for each timeframe
        sample = [
            {"timestamp": 0, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10},
            {"timestamp": 1, "open": 2, "high": 3, "low": 2, "close": 3, "volume": 20},
        ]
        self.mock_db.get_recent_ohlcv.side_effect = lambda s_id, sym, tf, limit=50: sample
        summary = self.strategy._build_timeframe_summary(1, "BTC/USD")
        self.assertIn("1m:", summary)
        self.assertIn("1d:", summary)


if __name__ == "__main__":
    unittest.main()
