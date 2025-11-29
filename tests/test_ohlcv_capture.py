import pytest

from trader_bot.database import TradingDatabase
from trader_bot.strategy_runner import StrategyRunner


class StubBot:
    def __init__(self):
        self.calls = []

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe, limit))
        idx = len(self.calls)
        return [
            {
                "timestamp": 1_000_000 + idx,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5 + idx,
                "volume": 10.0,
            }
        ]


@pytest.mark.asyncio
async def test_capture_ohlcv_throttles_and_prunes(tmp_path):
    db_path = tmp_path / "ohlcv.db"
    db = TradingDatabase(db_path=str(db_path))
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="ohlcv-test")

    bot = StubBot()
    runner = StrategyRunner(execute_orders=False)
    runner.db = db
    runner.bot = bot
    runner.session_id = session_id
    runner.telemetry_logger = None
    runner.ohlcv_retention_limit = 2
    runner.ohlcv_min_capture_spacing_seconds = 60
    runner._monotonic = lambda: 0.0

    await runner._capture_ohlcv("BTC/USD")

    # Initial capture should fetch all timeframes
    assert len(bot.calls) == 4
    count_1m = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM ohlcv_bars WHERE session_id = ? AND timeframe = '1m'", (session_id,)
    ).fetchone()["cnt"]
    assert count_1m == 1

    # Within spacing window, nothing new should be fetched
    runner._monotonic = lambda: 30.0
    await runner._capture_ohlcv("BTC/USD")
    assert len(bot.calls) == 4

    # After spacing for 1m only, only that timeframe should fetch again
    runner._monotonic = lambda: 120.0
    await runner._capture_ohlcv("BTC/USD")
    assert len(bot.calls) == 5

    latest_count = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM ohlcv_bars WHERE session_id = ? AND timeframe = '1m'", (session_id,)
    ).fetchone()["cnt"]
    assert latest_count == 2
