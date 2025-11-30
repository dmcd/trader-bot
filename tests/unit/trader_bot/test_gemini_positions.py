import pytest

from trader_bot.gemini_trader import GeminiTrader


class DummyExchange:
    def __init__(self, balances, tickers):
        self._balances = balances
        self._tickers = tickers

    async def fetch_balance(self):
        return {"total": self._balances, "timestamp": 12345}

    async def fetch_ticker(self, symbol):
        data = self._tickers.get(symbol)
        if data is None:
            raise RuntimeError(f"no ticker for {symbol}")
        return data


@pytest.mark.asyncio
async def test_get_positions_async_populates_mark_prices():
    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = DummyExchange(
        balances={"BTC": 0.5, "ETH": 2.0, "USD": 1000.0},
        tickers={
            "BTC/USD": {"last": 30000.0},
            "ETH/USD": {"bid": 1900.0, "ask": 1910.0},
        },
    )

    positions = await trader.get_positions_async()

    btc = next(p for p in positions if p["symbol"] == "BTC/USD")
    assert btc["avg_price"] == 30000.0
    assert btc["current_price"] == 30000.0

    eth = next(p for p in positions if p["symbol"] == "ETH/USD")
    assert eth["avg_price"] == pytest.approx(1905.0)
    assert eth["current_price"] == pytest.approx(1905.0)

    usd = next(p for p in positions if p["symbol"] == "USD")
    assert usd["avg_price"] == 0.0
    assert usd["current_price"] == 0.0


@pytest.mark.asyncio
async def test_get_positions_async_handles_missing_price():
    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = DummyExchange(balances={"SOL": 1.0}, tickers={})

    positions = await trader.get_positions_async()

    sol = positions[0]
    assert sol["avg_price"] == 0.0
    assert sol["current_price"] == 0.0
