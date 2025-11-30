import asyncio

import pytest

from trader_bot import server
from trader_bot.trader import BaseTrader
from trader_bot.utils import get_client_order_id


class StubTrader(BaseTrader):
    def __init__(self):
        self.connected = False

    async def connect_async(self):
        self.connected = True

    async def close(self):
        self.connected = False

    async def get_account_summary_async(self):
        return []

    async def get_market_data_async(self, symbol):
        return {"symbol": symbol}

    async def place_order_async(self, symbol, action, quantity, prefer_maker: bool = True, force_market: bool = False):
        return {
            "order_id": "1",
            "symbol": symbol,
            "action": action,
            "qty": quantity,
            "prefer_maker": prefer_maker,
            "force_market": force_market,
        }

    async def get_equity_async(self):
        return 0.0

    async def get_positions_async(self):
        return []

    async def get_open_orders_async(self):
        return []

    async def cancel_open_order_async(self, order_id):
        return True

    async def get_my_trades_async(self, symbol: str, since: int = None, limit: int = None):
        return []

    async def get_trades_from_timestamp(self, symbol: str, timestamp: int) -> list:
        return []

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> list:
        return []


class MissingMethodTrader(BaseTrader):
    async def connect_async(self):
        return None


def test_base_trader_enforces_abstract_methods():
    with pytest.raises(TypeError):
        MissingMethodTrader()

    stub = StubTrader()
    assert isinstance(stub, BaseTrader)


@pytest.mark.asyncio
async def test_ensure_bot_connected_runs_connect_and_raises_when_missing():
    server.bot = None
    with pytest.raises(RuntimeError):
        await server._ensure_bot_connected()

    stub = StubTrader()
    server.bot = stub
    await server._ensure_bot_connected()
    assert stub.connected is True
    server.bot = None


def test_get_client_order_id_fallbacks():
    assert get_client_order_id(None) == ""
    assert get_client_order_id({"clientOrderId": "abc"}) == "abc"
    assert get_client_order_id({"client_order_id": 123}) == "123"
    assert get_client_order_id({"info": {"client_order": "xyz"}}) == "xyz"
    assert get_client_order_id({"orderRef": "ib-ref"}) == "ib-ref"
