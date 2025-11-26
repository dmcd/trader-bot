import pytest

from data_fetch_coordinator import DataFetchCoordinator
from llm_tools import (
    OrderBookParams,
    ToolName,
    ToolRequest,
    normalize_trades,
)


class StubExchange:
    def __init__(self):
        self.ohlcv_calls = 0
        self.order_book_calls = 0
        self.trade_calls = 0

    async def fetch_ohlcv(self, symbol, timeframe, limit):
        self.ohlcv_calls += 1
        return [
            [1, 10, 11, 9, 10.5, 100],
            [2, 10.5, 12, 10, 11, 120],
        ]

    async def fetch_order_book(self, symbol, limit):
        self.order_book_calls += 1
        return {"bids": [[99, 1]], "asks": [[101, 2]], "timestamp": 123}

    async def fetch_trades(self, symbol, limit):
        self.trade_calls += 1
        return [
            {"timestamp": 1, "price": 100, "amount": 0.1, "side": "buy"},
            {"timestamp": 2, "price": 101, "amount": 0.2, "side": "sell"},
        ]


@pytest.mark.asyncio
async def test_market_data_uses_cache_between_calls():
    exchange = StubExchange()
    coordinator = DataFetchCoordinator(exchange, cache_ttl_seconds=60)
    params = ToolRequest(
        id="m1",
        tool=ToolName.GET_MARKET_DATA,
        params={"symbol": "BTC/USD", "timeframes": ["1m"], "limit": 2},
    ).params

    await coordinator.fetch_market_data(params)
    assert exchange.ohlcv_calls == 1
    await coordinator.fetch_market_data(params)
    assert exchange.ohlcv_calls == 1  # cached


@pytest.mark.asyncio
async def test_handle_requests_returns_normalized_order_book_and_clamps_size():
    exchange = StubExchange()
    coordinator = DataFetchCoordinator(exchange, max_json_bytes=10)
    reqs = [
        ToolRequest(
            id="ob1",
            tool=ToolName.GET_ORDER_BOOK,
            params=OrderBookParams(symbol="ETH/USD", depth=10),
        )
    ]
    responses = await coordinator.handle_requests(reqs)
    assert len(responses) == 1
    resp = responses[0]
    assert resp.error is None
    assert resp.data["meta"]["depth"] == 10
    assert resp.data.get("truncated") is True
    assert exchange.order_book_calls == 1


def test_normalize_trades_truncates():
    raw = [
        {"timestamp": 1, "price": 10, "amount": 0.1, "side": "buy"},
        {"timestamp": 2, "price": 11, "amount": 0.2, "side": "sell"},
        {"timestamp": 3, "price": 12, "amount": 0.3, "side": "buy"},
    ]
    shaped = normalize_trades(raw, requested_limit=2, max_trades=2)
    assert shaped["returned"] == 2
    assert shaped["truncated"] is True
