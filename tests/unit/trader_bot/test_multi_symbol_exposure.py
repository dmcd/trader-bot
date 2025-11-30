import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.strategy_runner import StrategyRunner


@pytest.mark.asyncio
async def test_runner_handles_multi_symbol_exposure(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "multi-exposure.db"))
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = runner.db.get_or_create_session(starting_balance=1000.0, bot_version="multi-exposure")
    runner.session = runner.db.get_session(runner.session_id)
    runner.daily_loss_pct = 50.0
    runner.risk_manager.start_of_day_equity = 1000.0

    runner._get_active_symbols = lambda: ["BTC/USD", "ETH/USD"]

    runner.bot.get_equity_async = AsyncMock(return_value=1000.0)
    positions = [
        {"symbol": "BTC/USD", "quantity": 0.1, "avg_price": None},
        {"symbol": "ETH/USD", "quantity": 2.0, "avg_price": None},
    ]
    runner.bot.get_positions_async = AsyncMock(return_value=positions)
    runner.bot.get_open_orders_async = AsyncMock(
        return_value=[
            {"symbol": "ETH/USD", "side": "buy", "price": 50.0, "amount": 1.0, "remaining": 1.0, "clientOrderId": f"{CLIENT_ORDER_PREFIX}1"}
        ]
    )

    market_map = {
        "BTC/USD": {"price": 30000.0, "bid": 29950.0, "ask": 30050.0},
        "ETH/USD": {"price": 2000.0, "bid": 1995.0, "ask": 2005.0},
    }

    async def fake_market_data(symbol):
        md = dict(market_map[symbol])
        md["_fetched_monotonic"] = 0.0
        return md

    runner.bot.get_market_data_async = AsyncMock(side_effect=fake_market_data)

    price_overrides_seen = {}

    def capture_exposure(price_overrides=None):
        nonlocal price_overrides_seen
        price_overrides_seen = price_overrides or {}
        return 0.0

    runner.risk_manager.get_total_exposure = MagicMock(side_effect=capture_exposure)

    symbols = runner._get_active_symbols()
    market_data = {}
    for sym in symbols:
        md = await runner.bot.get_market_data_async(sym)
        market_data[sym] = md
        runner.db.log_market_data(
            runner.session_id,
            sym,
            md.get("price"),
            md.get("bid"),
            md.get("ask"),
            md.get("volume", 0.0),
            spread_pct=md.get("spread_pct"),
            bid_size=md.get("bid_size"),
            ask_size=md.get("ask_size"),
            ob_imbalance=md.get("ob_imbalance"),
        )

    open_orders = await runner.bot.get_open_orders_async()
    runner.db.replace_open_orders(runner.session_id, open_orders)
    live_positions = await runner.bot.get_positions_async()
    runner.db.replace_positions(runner.session_id, live_positions)

    positions_dict = {}
    price_lookup = {}
    positions_data = runner.db.get_positions(runner.session_id)
    for pos in positions_data:
        sym = pos["symbol"]
        current_price = pos.get("avg_price") or 0
        recent = runner.db.get_recent_market_data(runner.session_id, sym, limit=1)
        if recent and recent[0].get("price"):
            current_price = recent[0]["price"]
        if market_data.get(sym) and market_data[sym].get("price"):
            current_price = market_data[sym]["price"]
        if current_price:
            positions_dict[sym] = {"quantity": pos["quantity"], "current_price": current_price}
    runner.risk_manager.update_positions(positions_dict)

    for sym, md in market_data.items():
        if md and md.get("price"):
            price_lookup[sym] = md["price"]
    runner.risk_manager.update_pending_orders(open_orders, price_lookup=price_lookup)

    price_overrides = {sym: md.get("price") for sym, md in market_data.items() if md and md.get("price")}
    price_overrides = price_overrides or None
    runner.risk_manager.get_total_exposure(price_overrides=price_overrides)

    assert runner.bot.get_market_data_async.call_count == 2
    assert set(price_overrides_seen.keys()) == {"BTC/USD", "ETH/USD"}
    assert price_overrides_seen["ETH/USD"] == 2000.0
