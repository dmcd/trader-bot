import pytest

from trader_bot.ib_trader import IBTrader


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTicker:
    def __init__(
        self,
        price: float | None,
        bid: float | None = None,
        ask: float | None = None,
        *,
        bid_size: float | None = None,
        ask_size: float | None = None,
        last: float | None = None,
        volume: float | None = None,
    ):
        self._price = price
        self.bid = bid
        self.ask = ask
        self.bidSize = bid_size
        self.askSize = ask_size
        self.last = last
        self.volume = volume

    def marketPrice(self):
        return self._price

    def midpoint(self):
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self._price


class FakeOrderStatus:
    def __init__(
        self,
        status: str,
        *,
        filled: float | None = None,
        remaining: float | None = None,
        avg_fill_price: float | None = None,
        order_id: int | None = None,
        commission: float | None = None,
        liquidity: str | None = None,
    ):
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avgFillPrice = avg_fill_price
        self.orderId = order_id
        self.commission = commission
        self.liquidity = liquidity


class FakeTrade:
    def __init__(self, statuses):
        self._statuses = list(statuses or [])
        if not self._statuses:
            self._statuses.append(FakeOrderStatus("Submitted"))
        self.orderStatus = self._statuses[0]
        self.order = None
        self.contract = None

    def advance_status(self):
        if len(self._statuses) > 1:
            self._statuses.pop(0)
            self.orderStatus = self._statuses[0]


class FakeIB:
    def __init__(
        self,
        *,
        connected: bool = False,
        fail_connect: bool = False,
        async_disconnect: bool = False,
        account_values=None,
        market_data=None,
        order_statuses=None,
    ):
        self.connected = connected
        self.fail_connect = fail_connect
        self.connect_calls = []
        self.req_time_calls = 0
        self.disconnect_calls = 0
        self.disconnect_async_calls = 0
        self.account_values_calls = 0
        self.req_mkt_data_calls = []
        self.place_order_calls = []
        self.account_values = account_values or []
        self.market_data = market_data or {}
        self.order_statuses = order_statuses or {}
        if not async_disconnect:
            # Shadow the coroutine so getattr returns None
            self.disconnectAsync = None

    async def connectAsync(self, host, port, clientId):
        self.connect_calls.append({"host": host, "port": port, "clientId": clientId})
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True
        return True

    def isConnected(self):
        return self.connected

    async def reqCurrentTimeAsync(self):
        self.req_time_calls += 1
        return 123

    def reqCurrentTime(self):
        self.req_time_calls += 1
        return 123

    async def accountValuesAsync(self):
        self.account_values_calls += 1
        return list(self.account_values)

    def accountValues(self):
        self.account_values_calls += 1
        return list(self.account_values)

    def _contract_key(self, contract):
        pair_fn = getattr(contract, "pair", None)
        if callable(pair_fn):
            val = pair_fn()
            if isinstance(val, str):
                return val.replace(".", "")
            return val
        symbol = getattr(contract, "symbol", None)
        currency = getattr(contract, "currency", None)
        if symbol and currency:
            return f"{symbol}{currency}"
        return str(contract)

    async def reqMktDataAsync(self, contract, *args, **kwargs):
        key = self._contract_key(contract)
        self.req_mkt_data_calls.append(key)
        return self.market_data.get(key)

    async def placeOrderAsync(self, contract, order):
        key = self._contract_key(contract)
        self.place_order_calls.append({"contract": key, "order": order})
        statuses = self.order_statuses.get(key, [])
        trade = FakeTrade(statuses)
        trade.order = order
        trade.contract = contract
        return trade

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    async def disconnectAsync(self):
        self.disconnect_async_calls += 1
        self.connected = False


@pytest.mark.asyncio
async def test_connects_and_pings_with_paper_defaults():
    clock = FakeClock()
    fake_ib = FakeIB()
    trader = IBTrader(
        host="paper-host",
        port=None,
        client_id=7,
        paper=True,
        ib_client=fake_ib,
        monotonic=clock,
        heartbeat_interval=10,
        connect_timeout=1.0,
    )

    await trader.connect_async()

    assert fake_ib.connect_calls == [{"host": "paper-host", "port": 7497, "clientId": 7}]
    assert trader.connected is True
    assert fake_ib.req_time_calls == 1
    assert trader._last_ping_mono == pytest.approx(clock.now)


@pytest.mark.asyncio
async def test_reuses_connection_and_pings_when_stale():
    clock = FakeClock()
    fake_ib = FakeIB(connected=True)
    trader = IBTrader(
        ib_client=fake_ib,
        monotonic=clock,
        heartbeat_interval=5,
    )
    trader.connected = True
    trader._last_ping_mono = clock()

    clock.advance(6)
    await trader.connect_async()

    assert fake_ib.connect_calls == []
    assert fake_ib.req_time_calls == 1
    assert trader._last_ping_mono == pytest.approx(clock.now)


@pytest.mark.asyncio
async def test_close_disconnects_gracefully():
    fake_ib = FakeIB(connected=True)
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    await trader.close()

    assert trader.connected is False
    assert fake_ib.disconnect_calls == 1
    assert fake_ib.disconnect_async_calls == 0


@pytest.mark.asyncio
async def test_close_prefers_async_disconnect_when_available():
    fake_ib = FakeIB(connected=True, async_disconnect=True)
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    await trader.close()

    assert trader.connected is False
    assert fake_ib.disconnect_async_calls == 1
    assert fake_ib.disconnect_calls == 0


@pytest.mark.asyncio
async def test_connect_failure_marks_disconnected():
    fake_ib = FakeIB(fail_connect=True)
    trader = IBTrader(ib_client=fake_ib)

    with pytest.raises(RuntimeError):
        await trader.connect_async()

    assert trader.connected is False


class AccountValue:
    def __init__(self, tag, value, currency="AUD", account="ABC123"):
        self.tag = tag
        self.value = value
        self.currency = currency
        self.account = account


@pytest.mark.asyncio
async def test_get_account_summary_returns_numeric_entries():
    fake_ib = FakeIB(
        connected=True,
        account_values=[
            AccountValue("NetLiquidation", "1000", "AUD", "A1"),
            AccountValue("UnrealizedPnL", "5.5", "AUD", "A1"),
            AccountValue("Bad", "not-a-number", "AUD", "A1"),
        ],
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    summary = await trader.get_account_summary_async()

    assert summary == [
        {"account": "A1", "tag": "NetLiquidation", "value": 1000.0, "currency": "AUD"},
        {"account": "A1", "tag": "UnrealizedPnL", "value": 5.5, "currency": "AUD"},
    ]


@pytest.mark.asyncio
async def test_get_equity_converts_currencies_and_caches_fx_quotes():
    clock = FakeClock()
    fake_ib = FakeIB(
        connected=True,
        account_values=[
            AccountValue("NetLiquidation", "1000", "AUD"),
            AccountValue("NetLiquidation", "200", "USD"),
        ],
        market_data={
            "USDAUD": FakeTicker(1.5),
        },
    )
    trader = IBTrader(
        ib_client=fake_ib,
        monotonic=clock,
        fx_cache_ttl=5.0,
    )
    trader.connected = True

    equity = await trader.get_equity_async()

    assert equity == pytest.approx(1300.0)
    assert fake_ib.req_mkt_data_calls == ["USDAUD"]

    # Cached within ttl
    equity_cached = await trader.get_equity_async()
    assert equity_cached == pytest.approx(1300.0)
    assert fake_ib.req_mkt_data_calls == ["USDAUD"]

    # After ttl expires, fetch again
    clock.advance(6)
    equity_again = await trader.get_equity_async()
    assert equity_again == pytest.approx(1300.0)
    assert fake_ib.req_mkt_data_calls == ["USDAUD", "USDAUD"]


@pytest.mark.asyncio
async def test_get_equity_falls_back_to_cash_when_no_net_liquidation():
    fake_ib = FakeIB(
        connected=True,
        account_values=[
            AccountValue("TotalCashValue", "500", "AUD"),
        ],
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    equity = await trader.get_equity_async()

    assert equity == pytest.approx(500.0)


@pytest.mark.asyncio
async def test_get_market_data_maps_top_of_book_fields():
    fake_ib = FakeIB(
        connected=True,
        market_data={
            "BHPAUD": FakeTicker(
                price=101.0,
                bid=100.5,
                ask=101.5,
                bid_size=200,
                ask_size=120,
                last=101.25,
                volume=15000,
            )
        },
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    md = await trader.get_market_data_async("BHP/AUD")

    assert md["symbol"] == "BHP/AUD"
    assert md["price"] == pytest.approx(101.25)
    assert md["bid"] == pytest.approx(100.5)
    assert md["ask"] == pytest.approx(101.5)
    assert md["bid_size"] == pytest.approx(200)
    assert md["ask_size"] == pytest.approx(120)
    assert md["volume"] == pytest.approx(15000)
    assert md["spread_pct"] == pytest.approx((101.5 - 100.5) / 101.0 * 100)
    assert md["ob_imbalance"] == pytest.approx((200 - 120) / (200 + 120))


@pytest.mark.asyncio
async def test_get_market_data_falls_back_to_mid_when_no_last():
    fake_ib = FakeIB(
        connected=True,
        market_data={
            "AUDUSD": FakeTicker(
                price=None,
                bid=0.655,
                ask=0.656,
                bid_size=1_000_000,
                ask_size=800_000,
                last=None,
                volume=None,
            )
        },
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    md = await trader.get_market_data_async("AUD/USD")

    assert md["price"] == pytest.approx((0.655 + 0.656) / 2)
    assert md["spread_pct"] == pytest.approx((0.656 - 0.655) / ((0.655 + 0.656) / 2) * 100)
    assert md["ob_imbalance"] == pytest.approx((1_000_000 - 800_000) / (1_000_000 + 800_000))


@pytest.mark.asyncio
async def test_place_order_limit_prefers_maker_and_maps_status():
    fake_ib = FakeIB(
        connected=True,
        market_data={"BHPAUD": FakeTicker(price=101.0, bid=100.5, ask=101.5)},
        order_statuses={
            "BHPAUD": [
                FakeOrderStatus("Submitted", filled=5, remaining=5, avg_fill_price=101.1, order_id=99),
                FakeOrderStatus("Filled", filled=10, remaining=0, avg_fill_price=101.2, order_id=99, commission=0.35, liquidity="added"),
            ]
        },
    )
    trader = IBTrader(ib_client=fake_ib, order_wait_timeout=0.5)
    trader.connected = True

    result = await trader.place_order_async("BHP/AUD", "BUY", 10, prefer_maker=True)

    assert fake_ib.place_order_calls
    call = fake_ib.place_order_calls[0]
    assert call["contract"] == "BHPAUD"
    limit_order = call["order"]
    assert getattr(limit_order, "orderRef") is not None
    assert getattr(limit_order, "totalQuantity") == 10
    assert getattr(limit_order, "lmtPrice") == pytest.approx(100.5 * 0.999)

    assert result["status"] == "filled"
    assert result["filled"] == 10
    assert result["remaining"] == 0
    assert result["avg_fill_price"] == pytest.approx(101.2)
    assert result["fee"] == pytest.approx(0.35)
    assert result["liquidity"] == "added"


@pytest.mark.asyncio
async def test_place_order_allows_market_when_requested():
    fake_ib = FakeIB(
        connected=True,
        market_data={"AUDUSD": FakeTicker(price=0.655, bid=0.654, ask=0.656)},
        order_statuses={
            "AUDUSD": [
                FakeOrderStatus("Filled", filled=5000, remaining=0, avg_fill_price=0.6555, order_id=12, commission=0.05),
            ]
        },
    )
    trader = IBTrader(ib_client=fake_ib, order_wait_timeout=0.3)
    trader.connected = True

    result = await trader.place_order_async("AUD/USD", "SELL", 5000, prefer_maker=False, force_market=True)

    order_obj = fake_ib.place_order_calls[0]["order"]
    assert order_obj.__class__.__name__ == "MarketOrder"
    assert result["status"] == "filled"
    assert result["avg_fill_price"] == pytest.approx(0.6555)
    assert result["fee"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_place_order_times_out_with_partial_status():
    fake_ib = FakeIB(
        connected=True,
        market_data={"CSLAUD": FakeTicker(price=25.0, bid=24.9, ask=25.1)},
        order_statuses={
            "CSLAUD": [
                FakeOrderStatus("Submitted", filled=3, remaining=7, avg_fill_price=25.0, order_id=101),
            ]
        },
    )
    trader = IBTrader(ib_client=fake_ib, order_wait_timeout=0.1, order_poll_interval=0.05)
    trader.connected = True

    result = await trader.place_order_async("CSL/AUD", "BUY", 10, prefer_maker=False)

    assert result["status"] in ("submitted", "open")
    assert result["filled"] == 3
    assert result["remaining"] == 7
