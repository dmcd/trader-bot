import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.services.resync_service import ResyncService
from trader_bot.utils import get_client_order_id


class StubDB:
    def __init__(self):
        self.positions = []
        self.orders = []
        self.replaced_positions = None
        self.replaced_orders = None
        self.health = {}
        self.replace_calls = 0
        self.replace_fail = False

    def get_positions(self, session_id):
        return self.positions

    def get_open_orders(self, session_id):
        return self.orders

    def replace_positions(self, session_id, positions):
        self.replaced_positions = positions

    def replace_open_orders(self, session_id, orders):
        self.replace_calls += 1
        if self.replace_fail:
            raise RuntimeError("replace failed")
        self.replaced_orders = orders

    def set_health_state(self, key, value, detail_str=None):
        self.health[key] = value


class StubBot:
    def __init__(self, positions=None, orders=None):
        self.positions = positions or []
        self.orders = orders or []

    async def get_positions_async(self):
        return self.positions

    async def get_open_orders_async(self):
        return self.orders


def test_filter_our_orders_only_keeps_prefixed():
    resync = ResyncService(db=None, bot=None, risk_manager=None, holdings_updater=None, session_stats_applier=None, logger=None)
    ours = {"clientOrderId": f"{CLIENT_ORDER_PREFIX}123"}
    foreign = {"clientOrderId": "X-1"}
    missing = {"id": 1}
    filtered = resync.filter_our_orders([ours, foreign, missing])
    assert filtered == [ours]


@pytest.mark.asyncio
async def test_reconcile_exchange_state_replaces_snapshots():
    db = StubDB()
    bot = StubBot(
        positions=[{"symbol": "BTC/USD", "quantity": 1, "avg_price": 100}],
        orders=[{"id": "1", "symbol": "BTC/USD", "clientOrderId": f"{CLIENT_ORDER_PREFIX}1"}],
    )
    risk = SimpleNamespace(update_positions=lambda *args, **kwargs: None, update_pending_orders=lambda *args, **kwargs: None)
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=risk,
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
        record_health_state=lambda key, val, detail=None: db.set_health_state(key, val),
    )
    resync.set_session(1)

    await resync.reconcile_exchange_state()

    assert db.replaced_positions == bot.positions
    assert db.replaced_orders == bot.orders
    assert db.health.get("restart_recovery") == "ok"


@pytest.mark.asyncio
async def test_reconcile_exchange_state_records_errors():
    db = StubDB()
    bot = StubBot()

    def record_health(key, val, detail=None):
        db.set_health_state(key, val, detail)

    async def boom():
        raise RuntimeError("boom")
    bot.get_positions_async = boom
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(update_positions=lambda *_: None, update_pending_orders=lambda *_: None),
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
        record_health_state=record_health,
    )
    resync.set_session(1)

    await resync.reconcile_exchange_state()
    assert db.health.get("restart_recovery") == "error"


@pytest.mark.asyncio
async def test_reconcile_open_orders_handles_exchange_failure():
    db = StubDB()

    async def boom():
        raise RuntimeError("no orders")

    bot = StubBot()
    bot.get_open_orders_async = boom
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(update_positions=lambda *_: None, update_pending_orders=lambda *_: None),
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_session(1)

    await resync.reconcile_open_orders()

    assert db.replace_calls == 0  # early exit, no refresh attempted


@pytest.mark.asyncio
async def test_reconcile_open_orders_replaces_stale_snapshot(caplog):
    db = StubDB()
    db.orders = [{"order_id": "old"}]
    bot = StubBot(orders=[{"order_id": "new", "clientOrderId": f"{CLIENT_ORDER_PREFIX}x"}])
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(update_positions=lambda *_: None, update_pending_orders=lambda *_: None),
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_session(1)

    with caplog.at_level("INFO"):
        await resync.reconcile_open_orders()

    assert db.replaced_orders == bot.orders
    assert any("stale open orders" in rec.message for rec in caplog.records)


class SyncStubDB:
    def __init__(self):
        self.logged = []
        self.recorded = None
        self.conn = SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(fetchone=lambda: None))

    def get_processed_trade_ids(self, session_id):
        return set()

    def get_latest_trade_timestamp(self, session_id):
        return None

    def log_trade(self, *args, **kwargs):
        self.logged.append((args, kwargs))

    def record_processed_trade_ids(self, session_id, processed):
        self.recorded = processed


class SyncStubBot:
    def __init__(self, trades):
        self.trades = trades
        self.calls = 0

    async def get_my_trades_async(self, symbol, since=None, limit=100):
        self.calls += 1
        return self.trades(self.calls)


@pytest.mark.asyncio
async def test_sync_trades_processes_and_records_ids():
    trade_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    trades = [{
        "id": "t1",
        "order": "o1",
        "side": "buy",
        "price": 100.0,
        "amount": 0.1,
        "fee": {"cost": 0.01},
        "timestamp": trade_ts,
        "clientOrderId": f"{CLIENT_ORDER_PREFIX}-123",
        "info": {"liquidity": "maker"},
    }]
    db = SyncStubDB()
    bot = SyncStubBot(lambda call: trades if call == 1 else [])
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
    )

    processed_ids: set[tuple[str, str | None]] = set()
    await resync.sync_trades_from_exchange(
        session_id=1,
        processed_trade_ids=processed_ids,
        order_reasons={"o1": "entry"},
        plan_reason_lookup=lambda *_: None,
        get_symbols=lambda: {"BTC/USD"},
    )

    assert db.logged
    assert db.recorded == {("t1", trades[0]["clientOrderId"])}
    assert "t1" in processed_ids


@pytest.mark.asyncio
async def test_sync_trades_paginates_and_skips_duplicates():
    trade_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    def paged_trades(call_count):
        if call_count == 1:
            # 100 identical trades forces pagination; duplicates should be skipped after first
            return [
                {
                    "id": "dup",
                    "order": "o2",
                    "side": "sell",
                    "price": 200.0,
                    "amount": 0.2,
                    "fee": {"cost": 0.02},
                    "timestamp": trade_ts,
                    "clientOrderId": f"{CLIENT_ORDER_PREFIX}-999",
                    "info": {"fillLiquidity": "taker"},
                }
            ] * 100
        return []

    db = SyncStubDB()
    db.record_processed_trade_ids = lambda session_id, processed: setattr(db, "recorded", processed) or (_ for _ in ()).throw(RuntimeError("persist failed"))
    bot = SyncStubBot(paged_trades)
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
    )

    processed_ids: set[tuple[str, str | None]] = set()
    await resync.sync_trades_from_exchange(
        session_id=2,
        processed_trade_ids=processed_ids,
        order_reasons={"o2": "exit"},
        plan_reason_lookup=lambda *_: None,
        get_symbols=lambda: {"BTC/USD"},
    )

    # logged once despite duplicates, pagination handled, processed set recorded
    assert db.logged
    assert len(db.logged) == 1
    assert "dup" in processed_ids
    assert db.recorded == {("dup", f"{CLIENT_ORDER_PREFIX}-999")}


@pytest.mark.asyncio
async def test_sync_trades_noop_when_session_missing():
    db = SyncStubDB()

    def trades(_):
        raise AssertionError("should not fetch trades")

    bot = SyncStubBot(trades)
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        session_stats_applier=lambda *args, **kwargs: None,
    )

    await resync.sync_trades_from_exchange(
        session_id=0,
        processed_trade_ids=set(),
        order_reasons={},
        plan_reason_lookup=lambda *_: None,
        get_symbols=lambda: set(),
    )

    assert db.logged == []
