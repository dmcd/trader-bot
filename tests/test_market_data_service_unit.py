import pytest

from trader_bot.database import TradingDatabase
from trader_bot.services.market_data_service import MarketDataService


class StubBot:
    def __init__(self):
        self.calls = []

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe, limit))
        return [
            {
                "timestamp": 1_000_000,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10.0,
            }
        ]


def test_timeframe_to_seconds():
    svc = MarketDataService(db=None, bot=None)
    assert svc.timeframe_to_seconds("1m") == 60
    assert svc.timeframe_to_seconds("2h") == 7200
    assert svc.timeframe_to_seconds("1d") == 86400
    assert svc.timeframe_to_seconds("bad") == 0


@pytest.mark.asyncio
async def test_capture_ohlcv_spacing_and_prune(tmp_path):
    db_path = tmp_path / "ohlcv.db"
    db = TradingDatabase(db_path=str(db_path))
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="ohlcv-test")
    bot = StubBot()
    svc = MarketDataService(
        db=db,
        bot=bot,
        session_id=session_id,
        monotonic=lambda: 0.0,
        ohlcv_min_capture_spacing_seconds=60,
        ohlcv_retention_limit=2,
    )

    await svc.capture_ohlcv("BTC/USD")
    assert len(bot.calls) == 4
    count_1m = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM ohlcv_bars WHERE session_id = ? AND timeframe = '1m'", (session_id,)
    ).fetchone()["cnt"]
    assert count_1m == 1

    # Within spacing window, nothing new should be fetched
    svc.monotonic = lambda: 30.0
    await svc.capture_ohlcv("BTC/USD")
    assert len(bot.calls) == 4

    # After spacing for 1m only, only that timeframe should fetch again
    svc.monotonic = lambda: 120.0
    await svc.capture_ohlcv("BTC/USD")
    assert len(bot.calls) == 5
    latest_count = db.conn.execute(
        "SELECT COUNT(*) as cnt FROM ohlcv_bars WHERE session_id = ? AND timeframe = '1m'", (session_id,)
    ).fetchone()["cnt"]
    assert latest_count == 2
