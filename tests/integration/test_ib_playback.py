import os

import pytest

from trader_bot.ib_trader import IBTrader
from tests.ib_fakes import FakeClock, load_ib_fixture_bundle, make_fake_ib_from_bundle


IB_TEST_MODE = os.getenv("IB_TEST_MODE", "playback").lower()
IB_PLAYBACK_FIXTURE = os.getenv("IB_PLAYBACK_FIXTURE")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ib_playback_smoke_maps_core_paths():
    if IB_TEST_MODE == "live":
        pytest.skip("IB live mode not enabled for CI playback; set IB_TEST_MODE=playback")

    bundle = load_ib_fixture_bundle(IB_PLAYBACK_FIXTURE) if IB_PLAYBACK_FIXTURE else load_ib_fixture_bundle()
    fake_ib = make_fake_ib_from_bundle(bundle)
    clock = FakeClock()

    trader = IBTrader(
        ib_client=fake_ib,
        monotonic=clock,
        order_wait_timeout=0.2,
        order_poll_interval=0.05,
        equity_tick_size=0.01,
    )

    await trader.connect_async()

    md = await trader.get_market_data_async("BHP/AUD")
    assert md["venue"] == "IB"
    assert md["instrument_type"] == "STK"

    positions = await trader.get_positions_async()
    assert any(p["symbol"] == "BHP/AUD" for p in positions)

    order_result = await trader.place_order_async("BHP/AUD", "BUY", 10, prefer_maker=True)
    assert order_result["status"] in {"filled", "open", "submitted"}
    assert order_result["order_id"]

    trades = await trader.get_my_trades_async("BHP/AUD")
    assert trades and trades[0]["symbol"] == "BHP/AUD"

    ohlcv = await trader.fetch_ohlcv("BHP/AUD", timeframe="1m", limit=1)
    assert len(ohlcv) == 1
    assert ohlcv[0][1] <= ohlcv[0][2]

    open_orders = await trader.get_open_orders_async()
    assert open_orders and open_orders[0]["symbol"] == "CSL/AUD"

    assert fake_ib.req_mkt_data_calls
    assert fake_ib.hist_calls
