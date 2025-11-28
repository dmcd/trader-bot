import asyncio

import pytest

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.strategy_runner import StrategyRunner


class StubConn:
    def execute(self, *args, **kwargs):
        class Result:
            def fetchone(self_inner):
                return None
        return Result()


class StubDB:
    def __init__(self, plan_reason=None):
        self.plan_reason = plan_reason
        self.logged_trades = []
        self.conn = StubConn()

    def get_distinct_trade_symbols(self, session_id):
        return ["BTC/USD"]

    def get_positions(self, session_id):
        return []

    def get_open_trade_plans(self, session_id):
        return []

    def get_open_orders(self, session_id):
        return []

    def get_latest_trade_timestamp(self, session_id):
        return None

    def log_trade(self, session_id, symbol, action, quantity, price, fee, reason, liquidity="unknown", realized_pnl=0.0, trade_id=None, timestamp=None):
        self.logged_trades.append(
            {
                "session_id": session_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "reason": reason,
                "liquidity": liquidity,
                "realized_pnl": realized_pnl,
                "trade_id": trade_id,
                "timestamp": timestamp,
            }
        )

    def get_trade_plan_reason_by_order(self, session_id, order_id=None, client_order_id=None):
        return self.plan_reason


class StubBot:
    def __init__(self, trades):
        self.trades = trades
        self.called_symbols = []

    async def get_my_trades_async(self, symbol, since=None, limit=None):
        # Return trades once, then empty to stop paging
        self.called_symbols.append(symbol)
        if self.trades is None:
            return []
        trades, self.trades = self.trades, None
        return trades


def build_trade(trade_id, client_oid=None, order_id="order-1", side="buy", price=100.0, amount=1.0, fee=0.0):
    payload = {
        "id": trade_id,
        "order": order_id,
        "side": side,
        "price": price,
        "amount": amount,
        "fee": {"cost": fee},
        "timestamp": 1,
        "datetime": "2024-01-01T00:00:00Z",
    }
    if client_oid is not None:
        payload["clientOrderId"] = client_oid
    return payload


@pytest.mark.asyncio
async def test_sync_trades_ignores_missing_client_ids():
    trade_missing_oid = build_trade("t-missing", client_oid=None)
    trade_with_oid = build_trade("t-kept", client_oid=f"{CLIENT_ORDER_PREFIX}123")

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 1
    runner.telemetry_logger = None
    runner.db = StubDB()
    runner.bot = StubBot([trade_missing_oid, trade_with_oid])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None
    runner.order_reasons = {"order-1": "LLM signal"}

    await runner.sync_trades_from_exchange()

    assert [t["trade_id"] for t in runner.db.logged_trades] == ["t-kept"]


@pytest.mark.asyncio
async def test_sync_trades_uses_plan_reason_fallback():
    client_oid = f"{CLIENT_ORDER_PREFIX}abc"
    trade_with_reason = build_trade("t-plan", client_oid=client_oid, order_id="order-42")

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 2
    runner.telemetry_logger = None
    runner.db = StubDB(plan_reason="Plan reason")
    runner.bot = StubBot([trade_with_reason])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None
    runner.order_reasons = {}  # ensure cache miss

    await runner.sync_trades_from_exchange()

    assert runner.db.logged_trades[0]["reason"] == "Plan reason"


@pytest.mark.asyncio
async def test_sync_trades_skips_unattributed_trades():
    client_oid = f"{CLIENT_ORDER_PREFIX}zzz"
    trade_unknown = build_trade("t-unknown", client_oid=client_oid, order_id="order-99")

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 4
    runner.telemetry_logger = None
    runner.db = StubDB(plan_reason=None)
    runner.bot = StubBot([trade_unknown])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None
    runner.order_reasons = {}  # no cached reason

    await runner.sync_trades_from_exchange()

    assert runner.db.logged_trades == []
    # Ensure we don't loop forever on the same trade id
    assert "t-unknown" in runner.processed_trade_ids


@pytest.mark.asyncio
async def test_sync_trades_skips_invalid_symbols():
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 3
    runner.telemetry_logger = None

    class SymbolDB(StubDB):
        def get_distinct_trade_symbols(self, session_id):
            return ["USD", "BTC/USD"]

    runner.db = SymbolDB()
    runner.bot = StubBot([])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None

    await runner.sync_trades_from_exchange()

    assert runner.bot.called_symbols == ["BTC/USD"]


def test_filter_our_orders_only_keeps_client_prefixed():
    runner = StrategyRunner(execute_orders=False)
    ours = {"clientOrderId": f"{CLIENT_ORDER_PREFIX}001", "symbol": "BTC/USD"}
    foreign = {"clientOrderId": "OTHER123", "symbol": "BTC/USD"}
    missing = {"symbol": "BTC/USD"}

    filtered = runner._filter_our_orders([ours, foreign, missing])

    assert filtered == [ours]
