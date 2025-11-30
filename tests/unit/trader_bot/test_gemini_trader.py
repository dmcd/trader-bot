import pytest

from trader_bot.gemini_trader import GeminiTrader
from tests.fakes import FakeExchange


class SandboxExchange:
    def __init__(self):
        self.markets = {
            "BTC/USD": {
                "precision": {"price": None, "amount": None},
                "limits": None,
            }
        }


class FallbackExchange:
    def __init__(self):
        self.closed = False
        self.markets = {
            "BTC/USD": {
                "precision": {"price": 0.1, "amount": 0.01},
                "limits": {"amount": {"min": 0.001}},
            }
        }

    async def load_markets(self):
        return self.markets

    async def close(self):
        self.closed = True


class ExplodingFallback:
    def __init__(self):
        self.closed = False

    async def load_markets(self):
        raise RuntimeError("fail")

    async def close(self):
        self.closed = True


class OrderExchange:
    def __init__(self):
        self.markets = {"BTC/USD": {"precision": {"price": 2, "amount": 3}}}
        self.order_log = []
        self.last_amount = None

    async def fetch_ticker(self, symbol):
        return {"bid": 100.0, "ask": 101.0, "last": 100.5}

    def market(self, symbol):
        return self.markets[symbol]

    def price_to_precision(self, symbol, price):
        return float(f"{price:.2f}")

    def amount_to_precision(self, symbol, amount):
        self.last_amount = float(f"{amount:.3f}")
        return self.last_amount

    async def create_limit_order(self, symbol, side, quantity, price, params):
        self.order_log.append({"postOnly": params.get("postOnly"), "price": price, "side": side})
        if len(self.order_log) == 1:
            return {"id": "1", "status": "rejected", "info": {}}
        return {"id": "2", "status": "open", "info": {}}

    async def fetch_order(self, order_id, symbol):
        last_price = self.order_log[-1]["price"]
        return {
            "id": order_id,
            "status": "open",
            "filled": 0,
            "remaining": self.last_amount or 0,
            "average": last_price,
        }


class MarkPriceExchange:
    def __init__(self):
        self.calls = []

    async def fetch_ticker(self, symbol):
        self.calls.append(symbol)
        if symbol == "FAIL":
            raise RuntimeError("boom")
        if symbol == "MID":
            return {"bid": 10.0, "ask": 12.0}
        return {"last": 5.0}


class TradesExchange:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []

    async def fetch_my_trades(self, symbol, since=None, limit=None):
        self.calls.append((symbol, since, limit))
        if self.should_fail:
            raise RuntimeError("no trades")
        return [{"id": 1, "symbol": symbol, "ts": since}]


class ConnectStubExchange:
    def __init__(self):
        self.urls = {"api": {}}
        self.markets = {"BTC/USD": {"precision": {"price": 0.01, "amount": 0.001}}}
        self.load_calls = 0
        self.sandbox_mode = False
        self.closed = False

    async def load_markets(self):
        self.load_calls += 1
        return self.markets

    def set_sandbox_mode(self, flag):
        self.sandbox_mode = flag

    async def close(self):
        self.closed = True


class MarketDataStub:
    def __init__(self):
        self.ticker_calls = 0

    async def fetch_ticker(self, symbol):
        self.ticker_calls += 1
        return {"last": 10.0, "bid": 9.0, "ask": 11.0}

    async def fetch_order_book(self, symbol, limit=5):
        raise RuntimeError("order book down")


@pytest.mark.asyncio
async def test_populate_precisions_uses_fallback_metadata(monkeypatch):
    trader = GeminiTrader()
    trader.exchange = SandboxExchange()
    fallback = FallbackExchange()
    monkeypatch.setattr("trader_bot.gemini_trader.ccxt.gemini", lambda *_args, **_kwargs: fallback)

    await trader._populate_precisions()

    market = trader.exchange.markets["BTC/USD"]
    assert market["precision"]["price"] == 0.1
    assert market["precision"]["amount"] == 0.01
    assert market["limits"] == fallback.markets["BTC/USD"]["limits"]
    assert fallback.closed is True


@pytest.mark.asyncio
async def test_populate_precisions_defaults_when_fallback_fails(monkeypatch):
    trader = GeminiTrader()
    trader.exchange = SandboxExchange()
    failing_fallback = ExplodingFallback()
    monkeypatch.setattr("trader_bot.gemini_trader.ccxt.gemini", lambda *_args, **_kwargs: failing_fallback)

    await trader._populate_precisions()

    market = trader.exchange.markets["BTC/USD"]
    assert market["precision"]["price"] == 0.01
    assert market["precision"]["amount"] == pytest.approx(1e-8)
    assert failing_fallback.closed is True


@pytest.mark.asyncio
async def test_place_order_retries_after_post_only_rejection():
    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = OrderExchange()

    result = await trader.place_order_async("BTC/USD", "BUY", 0.5, prefer_maker=True)

    assert len(trader.exchange.order_log) == 2  # maker then taker retry
    assert trader.exchange.order_log[0]["postOnly"] is True
    assert trader.exchange.order_log[0]["price"] < trader.exchange.order_log[1]["price"]
    assert result["liquidity"] == "taker"


@pytest.mark.asyncio
async def test_place_order_respects_taker_preference():
    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = OrderExchange()

    result = await trader.place_order_async("BTC/USD", "BUY", 0.25, prefer_maker=False)

    assert len(trader.exchange.order_log) == 1
    assert trader.exchange.order_log[0]["postOnly"] is None
    assert trader.exchange.order_log[0]["price"] == pytest.approx(101.0)
    assert result["liquidity"] == "taker"


@pytest.mark.asyncio
async def test_fetch_mark_prices_filters_failures():
    trader = GeminiTrader()
    trader.exchange = MarkPriceExchange()

    prices = await trader._fetch_mark_prices(["BTC/USD", "MID", "FAIL"])

    assert prices["BTC/USD"] == 5.0
    assert prices["MID"] == pytest.approx(11.0)
    assert "FAIL" not in prices
    assert set(trader.exchange.calls) == {"BTC/USD", "MID", "FAIL"}


@pytest.mark.asyncio
async def test_get_trades_from_timestamp_passes_since_and_handles_errors():
    trader = GeminiTrader()
    trader.connected = True
    good_exchange = TradesExchange()
    trader.exchange = good_exchange

    trades = await trader.get_trades_from_timestamp("BTC/USD", 12345)
    assert trades and trades[0]["ts"] == 12345
    assert good_exchange.calls == [("BTC/USD", 12345, None)]

    trader.exchange = TradesExchange(should_fail=True)
    trades = await trader.get_trades_from_timestamp("ETH/USD", 999)
    assert trades == []
    assert trader.exchange.calls == [("ETH/USD", 999, None)]


@pytest.mark.asyncio
async def test_connect_async_sets_sandbox_urls_and_is_idempotent(monkeypatch):
    stub = ConnectStubExchange()
    monkeypatch.setattr("trader_bot.gemini_trader.ccxt.gemini", lambda *_args, **_kwargs: stub)
    trader = GeminiTrader()

    await trader.connect_async()
    await trader.connect_async()  # guard against duplicate connects

    assert trader.connected is True
    assert stub.sandbox_mode is True
    assert stub.urls["api"]["public"] == "https://api.sandbox.gemini.com"
    assert stub.load_calls == 1
    assert stub.closed is False


@pytest.mark.asyncio
async def test_get_market_data_handles_order_book_failure_and_invalid_symbol():
    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = MarketDataStub()

    result = await trader.get_market_data_async("BTC/USD")
    assert result["symbol"] == "BTC/USD"
    assert result["bid"] == pytest.approx(9.0)
    assert result["ask"] == pytest.approx(11.0)
    assert result["spread_pct"] is None  # order book unavailable

    none_result = await trader.get_market_data_async("USD")
    assert none_result is None
    assert trader.exchange.ticker_calls == 1  # invalid symbol skipped


@pytest.mark.asyncio
async def test_calculate_total_usd_handles_inverted_and_missing_markets():
    trader = GeminiTrader()
    trader.connected = True

    class ValuationExchange:
        def __init__(self):
            self.markets = {"USD/ETH": {}, "DOGE/USD": {}}

        async def fetch_ticker(self, symbol):
            if symbol == "USD/ETH":
                return {"last": 0.002}  # invert to 500
            if symbol == "DOGE/USD":
                raise RuntimeError("no market")
            return {}

    trader.exchange = ValuationExchange()
    balance = {"total": {"USD": 10.0, "ETH": 2.0, "DOGE": 5.0}}

    total, price_map = await trader._calculate_total_usd(balance)

    assert total == pytest.approx(1010.0)  # 10 USD + 2 * 500
    assert price_map["ETH"] == pytest.approx(500.0)
    assert "DOGE" not in price_map


@pytest.mark.asyncio
async def test_get_equity_async_handles_balance_errors():
    trader = GeminiTrader()
    trader.connected = True

    class FailingExchange:
        async def fetch_balance(self):
            raise RuntimeError("fail")

    trader.exchange = FailingExchange()

    result = await trader.get_equity_async()
    assert result is None


@pytest.mark.asyncio
async def test_place_order_returns_none_when_ticker_missing_prices():
    trader = GeminiTrader()
    trader.connected = True

    class EmptyTickerExchange:
        def __init__(self):
            self.markets = {"BTC/USD": {"precision": {"price": 2, "amount": 3}}}

        async def fetch_ticker(self, symbol):
            return {}

        def market(self, symbol):
            return self.markets[symbol]

        def price_to_precision(self, symbol, price):
            raise AssertionError("should not format price when missing")

        def amount_to_precision(self, symbol, amount):
            raise AssertionError("should not format amount when missing")

    trader.exchange = EmptyTickerExchange()

    result = await trader.place_order_async("BTC/USD", "BUY", 1.0, prefer_maker=True)
    assert result is None


@pytest.mark.asyncio
async def test_fetch_ohlcv_handles_missing_method():
    class NoMethodExchange:
        pass

    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = NoMethodExchange()

    candles = await trader.fetch_ohlcv("BTC/USD")
    assert candles == []


@pytest.mark.asyncio
async def test_get_positions_async_populates_mark_prices():
    trader = GeminiTrader()
    trader.connected = True
    trader.exchange = FakeExchange(
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
    trader.exchange = FakeExchange(balances={"SOL": 1.0}, tickers={})

    positions = await trader.get_positions_async()

    sol = positions[0]
    assert sol["avg_price"] == 0.0
    assert sol["current_price"] == 0.0
