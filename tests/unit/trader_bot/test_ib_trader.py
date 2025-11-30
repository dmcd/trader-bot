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
    def __init__(self, price: float, bid: float | None = None, ask: float | None = None):
        self._price = price
        self.bid = bid
        self.ask = ask

    def marketPrice(self):
        return self._price

    def midpoint(self):
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self._price


class FakeIB:
    def __init__(
        self,
        *,
        connected: bool = False,
        fail_connect: bool = False,
        async_disconnect: bool = False,
        account_values=None,
        market_data=None,
    ):
        self.connected = connected
        self.fail_connect = fail_connect
        self.connect_calls = []
        self.req_time_calls = 0
        self.disconnect_calls = 0
        self.disconnect_async_calls = 0
        self.account_values_calls = 0
        self.req_mkt_data_calls = []
        self.account_values = account_values or []
        self.market_data = market_data or {}
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
            return pair_fn()
        symbol = getattr(contract, "symbol", None)
        currency = getattr(contract, "currency", None)
        if symbol and currency:
            return f"{symbol}{currency}"
        return str(contract)

    async def reqMktDataAsync(self, contract, *args, **kwargs):
        key = self._contract_key(contract)
        self.req_mkt_data_calls.append(key)
        return self.market_data.get(key)

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
