import pytest

from trader_bot import server
from trader_bot import shadow_runner

pytestmark = pytest.mark.integration


class StubBot:
    def __init__(self):
        self.connected = False
        self.calls = []

    async def connect_async(self):
        self.connected = True

    async def get_account_summary_async(self):
        self.calls.append(("account", None))
        return {"balance": 10}

    async def get_market_data_async(self, symbol):
        self.calls.append(("market", symbol))
        return {"symbol": symbol, "price": 123}

    async def place_order_async(self, symbol, side, quantity):
        self.calls.append(("order", symbol, side, quantity))
        return {"id": "order-1", "symbol": symbol, "side": side, "qty": quantity}


@pytest.mark.asyncio
async def test_server_requires_bot_instance():
    server.bot = None
    with pytest.raises(RuntimeError):
        await server.get_account_info()


@pytest.mark.asyncio
async def test_server_routes_calls_through_bot():
    bot = StubBot()
    server.bot = bot

    account = await server.get_account_info()
    quote = await server.get_stock_price("BTC/USD")
    buy = await server.buy_stock("ETH/USD", 1.5)
    sell = await server.sell_stock("ETH/USD", 1.0)

    assert bot.connected is True
    assert account["balance"] == 10
    assert quote["symbol"] == "BTC/USD"
    assert buy["side"] == "BUY"
    assert sell["side"] == "SELL"
    assert ("order", "ETH/USD", "BUY", 1.5) in bot.calls
    assert ("order", "ETH/USD", "SELL", 1.0) in bot.calls


@pytest.mark.asyncio
async def test_shadow_runner_invokes_strategy_runner(monkeypatch):
    called = {}

    class DummyRunner:
        def __init__(self, execute_orders):
            called["init_execute_orders"] = execute_orders

        async def run_loop(self):
            called["ran"] = True

    monkeypatch.setattr(shadow_runner, "StrategyRunner", DummyRunner)

    await shadow_runner.main()

    assert called["init_execute_orders"] is False
    assert called.get("ran") is True
