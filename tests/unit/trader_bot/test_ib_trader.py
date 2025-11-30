import pytest

from trader_bot.ib_trader import IBTrader
from tests.ib_fakes import (
    FakeBar,
    FakeClock,
    FakeContract,
    FakeFill,
    FakeIB,
    FakeOrderStatus,
    FakePortfolioEntry,
    FakeTicker,
    FakeTrade,
    load_ib_fixture_bundle,
    make_fake_ib_from_bundle,
)


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


def test_fixture_loader_builds_fake_ib():
    bundle = load_ib_fixture_bundle()
    fake_ib = make_fake_ib_from_bundle(bundle)

    assert fake_ib.isConnected() is True
    assert fake_ib.account_values
    assert "BHPAUD" in fake_ib.market_data
    assert fake_ib.order_statuses
    assert fake_ib.historical_data


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
    assert md["instrument_type"] == "STK"
    assert md["price"] == pytest.approx(101.25)
    assert md["bid"] == pytest.approx(100.5)
    assert md["ask"] == pytest.approx(101.5)
    assert md["bid_size"] == pytest.approx(200)
    assert md["ask_size"] == pytest.approx(120)
    assert md["volume"] == pytest.approx(15000)
    assert md["spread_pct"] == pytest.approx((101.5 - 100.5) / 101.0 * 100)
    assert md["ob_imbalance"] == pytest.approx((200 - 120) / (200 + 120))
    assert md["tick_size"] == pytest.approx(trader.equity_tick_size)
    assert md["venue"] == "IB"


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
    assert md["instrument_type"] == "FX"
    assert md["tick_size"] == pytest.approx(trader.fx_tick_size)
    assert md["venue"] == "IB"


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
    assert getattr(limit_order, "lmtPrice") == pytest.approx(100.39)

    assert result["status"] == "filled"
    assert result["filled"] == 10
    assert result["remaining"] == 0
    assert result["avg_fill_price"] == pytest.approx(101.2)
    assert result["fee"] == pytest.approx(0.35)
    assert result["liquidity"] == "added"


@pytest.mark.asyncio
async def test_limit_price_tick_size_rounds_maker_quotes():
    fake_ib = FakeIB(
        connected=True,
        market_data={"BHPAUD": FakeTicker(price=100.08, bid=100.07, ask=100.09)},
    )
    trader = IBTrader(ib_client=fake_ib, order_wait_timeout=0.5, equity_tick_size=0.05)
    trader.connected = True

    md = await trader.get_market_data_async("BHP/AUD")

    buy_price = trader._compute_limit_price(md, "BUY", prefer_maker=True)
    sell_price = trader._compute_limit_price(md, "SELL", prefer_maker=True)

    assert buy_price == pytest.approx(99.95)
    assert sell_price == pytest.approx(100.2)


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
async def test_force_market_falls_back_to_marketable_limit_for_equities():
    fake_ib = FakeIB(
        connected=True,
        market_data={"BHPAUD": FakeTicker(price=100.0, bid=99.9, ask=100.1)},
        order_statuses={
            "BHPAUD": [
                FakeOrderStatus("Filled", filled=10, remaining=0, avg_fill_price=99.7, order_id=24),
            ]
        },
    )
    trader = IBTrader(ib_client=fake_ib, order_wait_timeout=0.3, equity_tick_size=0.01)
    trader.connected = True

    result = await trader.place_order_async("BHP/AUD", "SELL", 10, prefer_maker=False, force_market=True)

    order_obj = fake_ib.place_order_calls[0]["order"]
    assert order_obj.__class__.__name__ == "LimitOrder"
    assert order_obj.lmtPrice == pytest.approx(99.65)
    assert result["status"] == "filled"


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


@pytest.mark.asyncio
async def test_get_positions_maps_long_and_short_with_prices():
    portfolio = [
        FakePortfolioEntry(FakeContract("BHP", "AUD", sec_type="STK"), position=10, avg_cost=100.0),
        FakePortfolioEntry(FakeContract("AUD", "USD", sec_type="FX", exchange="IDEALPRO"), position=-5000, avg_cost=0.655),
    ]
    fake_ib = FakeIB(
        connected=True,
        portfolio=portfolio,
        market_data={
            "BHPAUD": FakeTicker(price=101.0),
            "AUDUSD": FakeTicker(price=0.656),
        },
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    positions = await trader.get_positions_async()

    assert positions == [
        {
            "symbol": "BHP/AUD",
            "quantity": 10,
            "avg_price": 100.0,
            "current_price": 101.0,
            "timestamp": positions[0]["timestamp"],
        },
        {
            "symbol": "AUD/USD",
            "quantity": -5000,
            "avg_price": 0.655,
            "current_price": 0.656,
            "timestamp": positions[1]["timestamp"],
        },
    ]
    assert positions[0]["timestamp"] is not None
    assert positions[1]["timestamp"] is not None


@pytest.mark.asyncio
async def test_get_positions_handles_missing_price():
    portfolio = [
        FakePortfolioEntry(FakeContract("CSL", "AUD", sec_type="STK"), position=3, avg_cost=25.0),
    ]
    fake_ib = FakeIB(
        connected=True,
        portfolio=portfolio,
        market_data={},
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    positions = await trader.get_positions_async()

    assert positions[0]["current_price"] is None


@pytest.mark.asyncio
async def test_get_open_orders_filters_prefix_and_maps_fields():
    open_orders = [
        FakeTrade(
            [FakeOrderStatus("Submitted", remaining=5, filled=0, avg_fill_price=None, order_id=1, perm_id=9001)],
            order_id=1,
        ),
        FakeTrade(
            [FakeOrderStatus("PreSubmitted", remaining=2, filled=3, avg_fill_price=100.2, order_id=2, perm_id=9002)],
            order_id=2,
        ),
    ]
    open_orders[0].order = type("Order", (), {"totalQuantity": 5, "lmtPrice": 100.5, "action": "BUY", "orderRef": "BOT-v1-abc"})()
    open_orders[0].contract = FakeContract("BHP", "AUD", sec_type="STK")
    open_orders[1].order = type("Order", (), {"totalQuantity": 5, "lmtPrice": 100.0, "action": "SELL", "orderRef": "OTHER-abc"})()
    open_orders[1].contract = FakeContract("CSL", "AUD", sec_type="STK")

    fake_ib = FakeIB(connected=True, open_orders=open_orders)
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    orders = await trader.get_open_orders_async()

    assert len(orders) == 1
    order = orders[0]
    assert order["symbol"] == "BHP/AUD"
    assert order["side"] == "buy"
    assert order["amount"] == 5
    assert order["price"] == 100.5
    assert order["remaining"] == 5
    assert order["status"] == "open"
    assert order["order_id"] == 1


@pytest.mark.asyncio
async def test_get_my_trades_maps_fills_and_filters_since_and_symbol():
    fills = [
        FakeFill(price=100.0, size=2, timestamp_ms=1_700_000_000_000, commission=0.1, liquidity="added"),
        FakeFill(price=101.0, size=1, timestamp_ms=1_700_000_100_000, commission=0.05, liquidity="removed"),
    ]
    other_fills = [FakeFill(price=50.0, size=1, timestamp_ms=1_700_000_050_000)]
    trade1 = FakeTrade(
        [FakeOrderStatus("Filled", filled=3, remaining=0, avg_fill_price=100.33, order_id=11, perm_id=8001)],
        fills=fills,
    )
    trade1.order = type("Order", (), {"action": "BUY", "totalQuantity": 3})()
    trade1.contract = FakeContract("BHP", "AUD")

    trade2 = FakeTrade(
        [FakeOrderStatus("Filled", filled=1, remaining=0, avg_fill_price=50, order_id=12, perm_id=8002)],
        fills=other_fills,
    )
    trade2.order = type("Order", (), {"action": "SELL", "totalQuantity": 1})()
    trade2.contract = FakeContract("CSL", "AUD")

    fake_ib = FakeIB(
        connected=True,
        trades=[trade1, trade2],
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    since = 1_700_000_050_000
    trades = await trader.get_my_trades_async("BHP/AUD", since=since)

    assert len(trades) == 1
    t = trades[0]
    assert t["symbol"] == "BHP/AUD"
    assert t["side"] == "buy"
    assert t["price"] == 101.0
    assert t["amount"] == 1
    assert t["fee"] == 0.05
    assert t["liquidity"] == "removed"
    assert t["order_id"] == 11
    assert t["trade_id"] == 8001


@pytest.mark.asyncio
async def test_get_trades_from_timestamp_delegates_and_limits():
    fills = [FakeFill(price=10, size=1, timestamp_ms=1_700_000_200_000)]
    trade = FakeTrade([FakeOrderStatus("Filled", filled=1, remaining=0, order_id=21, perm_id=9001)], fills=fills)
    trade.order = type("Order", (), {"action": "BUY", "totalQuantity": 1})()
    trade.contract = FakeContract("AUD", "USD", sec_type="FX", exchange="IDEALPRO")

    fake_ib = FakeIB(connected=True, trades=[trade])
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    trades = await trader.get_trades_from_timestamp("AUD/USD", 1_700_000_100_000)

    assert len(trades) == 1
    assert trades[0]["symbol"] == "AUD/USD"
    assert trades[0]["price"] == 10


@pytest.mark.asyncio
async def test_fetch_ohlcv_maps_bars_and_clamps_limit():
    bars = [
        FakeBar(1_700_000_000_000, 100, 101, 99, 100.5, 1000),
        FakeBar(1_700_000_060_000, 101, 102, 100, 101.5, 900),
    ]
    fake_ib = FakeIB(
        connected=True,
        historical_data={"BHPAUD": bars},
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    ohlcv = await trader.fetch_ohlcv("BHP/AUD", timeframe="1m", limit=1)

    assert fake_ib.hist_calls[0]["params"]["barSizeSetting"] == "1 min"
    assert fake_ib.hist_calls[0]["params"]["durationStr"] == "1 M"
    assert ohlcv == [[1_700_000_060_000, 101, 102, 100, 101.5, 900]]


@pytest.mark.asyncio
async def test_fetch_ohlcv_unsupported_timeframe_raises():
    fake_ib = FakeIB(connected=True, historical_data={})
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    with pytest.raises(ValueError):
        await trader.fetch_ohlcv("BHP/AUD", timeframe="2m", limit=10)


@pytest.mark.asyncio
async def test_connect_respects_backoff_after_failure():
    clock = FakeClock()
    fake_ib = FakeIB(fail_connect=True)
    trader = IBTrader(
        ib_client=fake_ib,
        monotonic=clock,
        reconnect_backoff_base=2.0,
        reconnect_backoff_max=5.0,
    )

    with pytest.raises(RuntimeError):
        await trader.connect_async()
    assert trader._next_reconnect_time == pytest.approx(clock.now + 2.0)

    fake_ib.fail_connect = False
    connect_calls_before = len(fake_ib.connect_calls)
    with pytest.raises(RuntimeError):
        await trader.connect_async()
    assert len(fake_ib.connect_calls) == connect_calls_before

    clock.advance(3.0)
    await trader.connect_async()
    assert trader.connected is True
    assert trader._reconnect_failures == 0
    assert trader._next_reconnect_time == 0


@pytest.mark.asyncio
async def test_ping_failure_schedules_backoff():
    clock = FakeClock()
    fake_ib = FakeIB(connected=True, fail_ping=True)
    trader = IBTrader(ib_client=fake_ib, monotonic=clock, reconnect_backoff_base=1.0)
    trader.connected = True

    with pytest.raises(RuntimeError):
        await trader.connect_async()

    assert trader.connected is False
    assert trader._next_reconnect_time > clock.now


@pytest.mark.asyncio
async def test_fetch_order_book_shapes_top_of_book():
    fake_ib = FakeIB(
        connected=True,
        market_data={
            "BHPAUD": FakeTicker(price=101.0, bid=100.5, ask=101.5, bid_size=200, ask_size=120)
        },
    )
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    ob = await trader.fetch_order_book("BHP/AUD")

    assert ob["bids"] == [[100.5, 200]]
    assert ob["asks"] == [[101.5, 120]]
    assert ob["symbol"] == "BHP/AUD"
    assert "timestamp" in ob


@pytest.mark.asyncio
async def test_fetch_trades_returns_empty_list():
    fake_ib = FakeIB(connected=True)
    trader = IBTrader(ib_client=fake_ib)
    trader.connected = True

    trades = await trader.fetch_trades("BHP/AUD", limit=5)

    assert trades == []


@pytest.mark.asyncio
async def test_market_data_cache_throttles_requests():
    clock = FakeClock()
    fake_ib = FakeIB(
        connected=True,
        market_data={"BHPAUD": FakeTicker(price=101.0, bid=100.5, ask=101.5, bid_size=200, ask_size=120)},
    )
    trader = IBTrader(ib_client=fake_ib, monotonic=clock, market_data_cache_seconds=5.0)
    trader.connected = True

    await trader.get_market_data_async("BHP/AUD")
    await trader.get_market_data_async("BHP/AUD")

    assert fake_ib.req_mkt_data_calls == ["BHPAUD"]


@pytest.mark.asyncio
async def test_hist_rate_limit_enforced():
    bars = [FakeBar(1_700_000_000_000, 100, 101, 99, 100.5, 1000)]
    fake_ib = FakeIB(connected=True, historical_data={"BHPAUD": bars})
    trader = IBTrader(ib_client=fake_ib, hist_request_limit=1, hist_window_seconds=1000.0)
    trader.connected = True

    await trader.fetch_ohlcv("BHP/AUD", timeframe="1m", limit=1)
    with pytest.raises(RuntimeError):
        await trader.fetch_ohlcv("BHP/AUD", timeframe="1m", limit=1)
