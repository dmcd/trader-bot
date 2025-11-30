import pytest

from trader_bot.ib_trader import IBTrader


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeIB:
    def __init__(self, *, connected: bool = False, fail_connect: bool = False, async_disconnect: bool = False):
        self.connected = connected
        self.fail_connect = fail_connect
        self.connect_calls = []
        self.req_time_calls = 0
        self.disconnect_calls = 0
        self.disconnect_async_calls = 0
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
