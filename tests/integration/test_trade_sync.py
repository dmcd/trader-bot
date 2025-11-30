import asyncio
import json
from datetime import datetime, timezone, timedelta

import pytest

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.database import TradingDatabase
from trader_bot.strategy_runner import StrategyRunner
from trader_bot.trading_context import TradingContext
from trader_bot.utils import get_client_order_id

pytestmark = pytest.mark.integration


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

    def close(self):
        # Mirror TradingDatabase interface for tests that clean up explicitly
        return None

    def get_distinct_trade_symbols(self, session_id, portfolio_id=None):
        return ["BTC/USD"]

    def get_positions(self, session_id, portfolio_id=None):
        return []

    def get_open_trade_plans(self, session_id, portfolio_id=None):
        return []

    def get_open_orders(self, session_id, portfolio_id=None):
        return []

    def get_latest_trade_timestamp(self, session_id, portfolio_id=None):
        return None

    def log_trade(self, session_id, symbol, action, quantity, price, fee, reason, liquidity="unknown", realized_pnl=0.0, trade_id=None, timestamp=None, portfolio_id=None):
        self.logged_trades.append(
            {
                "session_id": session_id,
                "portfolio_id": portfolio_id,
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

    def get_trade_plan_reason_by_order(self, session_id, order_id=None, client_order_id=None, portfolio_id=None):
        return self.plan_reason

    def get_processed_trade_ids(self, session_id, portfolio_id=None):
        return set()

    def record_processed_trade_ids(self, session_id, processed, portfolio_id=None):
        self.recorded = processed


class StubBot:
    def __init__(self, trades):
        self.trades = trades
        self.called_symbols = []
        self.open_orders = []

    async def get_my_trades_async(self, symbol, since=None, limit=None):
        # Return trades once, then empty to stop paging
        self.called_symbols.append(symbol)
        if self.trades is None:
            return []
        trades, self.trades = self.trades, None
        return trades

    async def get_open_orders_async(self):
        return self.open_orders


def build_trade(trade_id, client_oid=None, order_id="order-1", side="buy", price=100.0, amount=1.0, fee=0.0, info=None, liquidity=None):
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    payload = {
        "id": trade_id,
        "order": order_id,
        "side": side,
        "price": price,
        "amount": amount,
        "fee": {"cost": fee},
        "timestamp": ts,
        "datetime": "2024-01-01T00:00:00Z",
    }
    if client_oid is not None:
        payload["clientOrderId"] = client_oid
    if info is not None:
        payload["info"] = info
    if liquidity is not None:
        payload["liquidity"] = liquidity
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

    try:
        await runner.sync_trades_from_exchange()
    finally:
        runner.db.close()

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
async def test_sync_trades_respects_cutoff(monkeypatch):
    client_oid = f"{CLIENT_ORDER_PREFIX}old"
    # 2 hours old trade should be skipped when cutoff is 60 minutes
    old_ts = int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp() * 1000)
    old_trade = build_trade("t-old", client_oid=client_oid, order_id="order-old")
    old_trade["timestamp"] = old_ts

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 6
    runner.telemetry_logger = None
    runner.db = StubDB(plan_reason=None)
    runner.bot = StubBot([old_trade])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None
    runner.order_reasons = {"order-old": "LLM reason"}

    monkeypatch.setattr("trader_bot.strategy_runner.TRADE_SYNC_CUTOFF_MINUTES", 60)

    await runner.sync_trades_from_exchange()

    assert runner.db.logged_trades == []


@pytest.mark.asyncio
async def test_sync_trades_skips_invalid_symbols():
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 3
    runner.telemetry_logger = None

    class SymbolDB(StubDB):
        def get_distinct_trade_symbols(self, session_id, portfolio_id=None):
            return ["USD", "BTC/USD"]

    runner.db = SymbolDB()
    runner.bot = StubBot([])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None

    await runner.sync_trades_from_exchange()

    assert runner.bot.called_symbols == ["BTC/USD"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "liquidity_fields,expected",
    [
        ({"liquidity": "maker"}, "maker"),
        ({"info": {"fillLiquidity": "taker"}}, "taker"),
        ({"info": {"liquidityIndicator": "maker_or_cancel"}}, "maker_or_cancel"),
    ],
)
async def test_sync_trades_records_reported_liquidity(liquidity_fields, expected):
    client_oid = f"{CLIENT_ORDER_PREFIX}liq"
    trade_liq = build_trade("t-liq", client_oid=client_oid, **liquidity_fields)

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 5
    runner.telemetry_logger = None
    runner.db = StubDB(plan_reason="LLM reason")
    runner.bot = StubBot([trade_liq])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None
    runner.order_reasons = {}

    await runner.sync_trades_from_exchange()

    assert runner.db.logged_trades[0]["liquidity"] == expected


def test_filter_our_orders_only_keeps_client_prefixed():
    runner = StrategyRunner(execute_orders=False)
    ours = {"clientOrderId": f"{CLIENT_ORDER_PREFIX}001", "symbol": "BTC/USD"}
    foreign = {"clientOrderId": "OTHER123", "symbol": "BTC/USD"}
    missing = {"symbol": "BTC/USD"}

    filtered = runner._filter_our_orders([ours, foreign, missing])

    assert filtered == [ours]


def test_get_client_order_id_variants():
    variants = [
        {"clientOrderId": "A"},
        {"client_order_id": "B"},
        {"client_order": "C"},
        {"info": {"clientOrderId": "D"}},
        {"info": {"client_order_id": "E"}},
        {"info": {"client_order": "F"}},
    ]
    expected = ["A", "B", "C", "D", "E", "F"]
    assert [get_client_order_id(v) for v in variants] == expected


@pytest.mark.asyncio
async def test_reconcile_open_orders_removes_stale():
    runner = StrategyRunner(execute_orders=False)
    runner.session_id = 5
    runner.telemetry_logger = None

    class ReconDB(StubDB):
        def __init__(self):
            super().__init__()
            self.replaced = None

        def get_open_orders(self, session_id, portfolio_id=None):
            return [
                {"order_id": "live-1", "symbol": "BTC/USD"},
                {"order_id": "stale-1", "symbol": "BTC/USD"},
            ]

        def replace_open_orders(self, session_id, orders, portfolio_id=None):
            self.replaced = orders

    runner.db = ReconDB()
    runner.bot = StubBot(trades=[])
    runner.bot.open_orders = [{"order_id": "live-1", "clientOrderId": f"{CLIENT_ORDER_PREFIX}111", "symbol": "BTC/USD"}]

    await runner._reconcile_open_orders()

    assert runner.db.replaced == [{"order_id": "live-1", "clientOrderId": f"{CLIENT_ORDER_PREFIX}111", "symbol": "BTC/USD"}]


def test_trading_context_filters_foreign_open_orders():
    class CtxDB(StubDB):
        def get_session_stats(self, session_id):
            return {
                "created_at": "2024-01-01T00:00:00",
                "date": "2024-01-01",
                "starting_balance": 0,
                "net_pnl": 0,
                "total_fees": 0,
                "total_llm_cost": 0,
                "total_trades": 0,
            }

        def get_recent_trades(self, session_id, limit=50, portfolio_id=None):
            return []

        def get_recent_market_data(self, session_id, symbol, limit=20, before_timestamp=None, portfolio_id=None):
            return [{"price": 100}, {"price": 100}]

        def get_positions(self, session_id, portfolio_id=None):
            return []

        def get_open_orders(self, session_id, portfolio_id=None):
            return []

    db = CtxDB()
    ctx = TradingContext(db, session_id=1)
    ours = {"clientOrderId": f"{CLIENT_ORDER_PREFIX}xyz", "symbol": "BTC/USD", "amount": 1, "price": 100}
    foreign = {"clientOrderId": "OTHER999", "symbol": "BTC/USD", "amount": 1, "price": 100}

    summary = ctx.get_context_summary("BTC/USD", open_orders=[ours, foreign])
    parsed = json.loads(summary)

    assert len(parsed["open_orders"]) == 1
    assert parsed["open_orders"][0]["symbol"] == "BTC/USD"


@pytest.mark.asyncio
async def test_processed_trade_ids_persist_across_runs(test_db_path):
    db = TradingDatabase(db_path=str(test_db_path))
    session_id = db.get_or_create_session(starting_balance=0.0, bot_version="test-dedupe")

    orphan_trade = build_trade("t-repeat", client_oid=f"{CLIENT_ORDER_PREFIX}repeat", order_id="order-repeat")

    runner = StrategyRunner(execute_orders=False)
    runner.session_id = session_id
    runner.telemetry_logger = None
    runner.db = db
    runner.bot = StubBot([orphan_trade])
    runner._update_holdings_and_realized = lambda *args, **kwargs: 0.0
    runner._apply_fill_to_session_stats = lambda *args, **kwargs: None
    runner.order_reasons = {}

    try:
        await runner.sync_trades_from_exchange()

        assert db.get_trade_count(session_id) == 0
        assert "t-repeat" in db.get_processed_trade_ids(session_id)

        runner2 = StrategyRunner(execute_orders=False)
        runner2.session_id = session_id
        runner2.telemetry_logger = None
        runner2.db = db
        runner2.bot = StubBot([orphan_trade])
        runner2._update_holdings_and_realized = lambda *args, **kwargs: 0.0
        runner2._apply_fill_to_session_stats = lambda *args, **kwargs: None
        runner2.order_reasons = {"order-repeat": "Recovered reason"}

        await runner2.sync_trades_from_exchange()

        assert db.get_trade_count(session_id) == 0
        assert "t-repeat" in db.get_processed_trade_ids(session_id)
    finally:
        db.close()
