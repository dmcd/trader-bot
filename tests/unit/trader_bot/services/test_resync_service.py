import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.services.plan_monitor import PlanMonitor, PlanMonitorConfig
from trader_bot.services.resync_service import ResyncService
from trader_bot.utils import get_client_order_id


class StubDB:
    def __init__(self):
        self.positions = []
        self.orders = []
        self.open_plans = []
        self.replaced_positions = None
        self.replaced_orders = None
        self.health = {}
        self.replace_calls = 0
        self.replace_fail = False
        self.closed_plans = []

    def get_positions_for_portfolio(self, portfolio_id):
        return self.positions

    def get_open_orders_for_portfolio(self, portfolio_id):
        return self.orders

    def get_open_trade_plans_for_portfolio(self, portfolio_id):
        return self.open_plans

    def replace_positions_for_portfolio(self, portfolio_id, positions):
        self.replaced_positions = positions

    def replace_open_orders_for_portfolio(self, portfolio_id, orders):
        self.replace_calls += 1
        if self.replace_fail:
            raise RuntimeError("replace failed")
        self.replaced_orders = orders

    def update_trade_plan_status(self, plan_id, *, status, closed_at=None, reason=None):
        self.closed_plans.append((plan_id, status, closed_at, reason))

    def log_trade_for_portfolio(self, *args, **kwargs):
        return None

    def update_trade_plan_prices(self, *args, **kwargs):
        return None

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


class TrackingRisk:
    def __init__(self):
        self.positions = None
        self.pending_orders = None

    def update_positions(self, positions):
        self.positions = positions

    def update_pending_orders(self, orders, price_lookup=None):
        self.pending_orders = orders

    def get_total_exposure(self):
        return 0.0


def test_filter_our_orders_only_keeps_prefixed():
    resync = ResyncService(db=None, bot=None, risk_manager=None, holdings_updater=None, portfolio_stats_applier=None, logger=None)
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
        portfolio_stats_applier=lambda *args, **kwargs: None,
        record_health_state=lambda key, val, detail=None: db.set_health_state(key, val),
    )
    resync.set_portfolio(1)

    await resync.reconcile_exchange_state()

    assert db.replaced_positions == bot.positions
    assert db.replaced_orders == bot.orders
    assert db.health.get("restart_recovery") == "ok"


def test_bootstrap_snapshots_restores_portfolio_state():
    db = StubDB()
    db.positions = [{"symbol": "ETH/USD", "quantity": 2.0, "avg_price": 1800.0}]
    db.orders = [{"id": "o1", "symbol": "ETH/USD", "clientOrderId": f"{CLIENT_ORDER_PREFIX}1"}]
    risk = TrackingRisk()
    resync = ResyncService(
        db=db,
        bot=None,
        risk_manager=risk,
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_portfolio(7)

    state = resync.bootstrap_snapshots()

    assert state["positions"] == db.positions
    assert state["open_orders"] == db.orders
    assert risk.positions == {"ETH/USD": {"quantity": 2.0, "current_price": 1800.0}}
    assert risk.pending_orders == db.orders
    assert db.replaced_positions is None
    assert db.replaced_orders is None


def test_bootstrap_snapshots_requires_portfolio():
    resync = ResyncService(
        db=StubDB(),
        bot=None,
        risk_manager=TrackingRisk(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError):
        resync.bootstrap_snapshots()


@pytest.mark.asyncio
async def test_bootstrap_carries_open_plans_and_preserves_monitor_state():
    db = StubDB()
    db.positions = [{"symbol": "ETH/USD", "quantity": 1.0, "avg_price": 1800.0}]
    db.open_plans = [
        {
            "id": 5,
            "symbol": "ETH/USD",
            "side": "BUY",
            "entry_price": 1800.0,
            "stop_price": 1700.0,
            "target_price": 2000.0,
            "size": 1.0,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    risk = TrackingRisk()
    resync = ResyncService(
        db=db,
        bot=None,
        risk_manager=risk,
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
        portfolio_id=3,
    )

    state = resync.bootstrap_snapshots()

    assert state["open_plans"] == db.open_plans
    bot = AsyncMock()
    bot.place_order_async = AsyncMock(return_value={"order_id": "1", "liquidity": "taker"})
    monitor = PlanMonitor(
        db=db,
        bot=bot,
        cost_tracker=MagicMock(calculate_trade_fee=lambda *args, **kwargs: 0.0),
        risk_manager=risk,
        prefer_maker=lambda symbol: False,
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
        portfolio_id=3,
    )
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.02)
    await monitor.monitor(
        price_lookup={"ETH/USD": 1810.0},
        open_orders=[],
        config=config,
        now=datetime.now(timezone.utc),
        portfolio_id=None,
    )

    assert db.closed_plans == []


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
        portfolio_stats_applier=lambda *args, **kwargs: None,
        record_health_state=record_health,
    )
    resync.set_portfolio(1)

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
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_portfolio(1)

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
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_portfolio(1)

    with caplog.at_level("INFO"):
        await resync.reconcile_open_orders()

    assert db.replaced_orders == bot.orders
    assert any("stale open orders" in rec.message for rec in caplog.records)


class SyncStubDB:
    def __init__(self):
        self.logged = []
        self.recorded = None
        self.conn = SimpleNamespace(execute=lambda *args, **kwargs: SimpleNamespace(fetchone=lambda: None))

    def get_processed_trade_entries_for_portfolio(self, portfolio_id):
        return set()

    def get_latest_trade_timestamp_for_portfolio(self, portfolio_id):
        return None

    def log_trade_for_portfolio(self, *args, **kwargs):
        self.logged.append((args, kwargs))

    def record_processed_trade_ids_for_portfolio(self, portfolio_id, processed):
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
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_portfolio(1)

    processed_ids: set[str] = set()
    await resync.sync_trades_from_exchange(
        processed_trade_ids=processed_ids,
        order_reasons={"o1": "entry"},
        plan_reason_lookup=lambda *_: None,
        get_symbols=lambda: {"BTC/USD"},
    )

    assert db.logged
    assert db.recorded == {("t1", trades[0]["clientOrderId"])}
    assert ResyncService._trade_key("t1") in processed_ids
    assert ResyncService._client_key(trades[0]["clientOrderId"]) in processed_ids


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
    db.record_processed_trade_ids_for_portfolio = lambda portfolio_id, processed: setattr(db, "recorded", processed) or (_ for _ in ()).throw(RuntimeError("persist failed"))
    bot = SyncStubBot(paged_trades)
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_portfolio(2)

    processed_ids: set[str] = set()
    await resync.sync_trades_from_exchange(
        processed_trade_ids=processed_ids,
        order_reasons={"o2": "exit"},
        plan_reason_lookup=lambda *_: None,
        get_symbols=lambda: {"BTC/USD"},
    )

    # logged once despite duplicates, pagination handled, processed set recorded
    assert db.logged
    assert len(db.logged) == 1
    assert ResyncService._trade_key("dup") in processed_ids
    assert ResyncService._client_key(f"{CLIENT_ORDER_PREFIX}-999") in processed_ids
    assert db.recorded == {("dup", f"{CLIENT_ORDER_PREFIX}-999")}


@pytest.mark.asyncio
async def test_sync_trades_skips_persisted_entries_without_logging():
    trade_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    persisted_trade = ("old-trade", f"{CLIENT_ORDER_PREFIX}-persisted")

    trades = [
        {
            "id": persisted_trade[0],
            "order": "ord-old",
            "side": "buy",
            "price": 50.0,
            "amount": 0.5,
            "fee": {"cost": 0.0},
            "timestamp": trade_ts,
            "clientOrderId": persisted_trade[1],
        }
    ]

    db = SyncStubDB()
    db.get_processed_trade_entries_for_portfolio = lambda *_: {persisted_trade}
    bot = SyncStubBot(lambda call: trades if call == 1 else [])
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )
    resync.set_portfolio(3)

    processed_ids: set[str] = set()
    await resync.sync_trades_from_exchange(
        processed_trade_ids=processed_ids,
        order_reasons={"ord-old": "entry"},
        plan_reason_lookup=lambda *_: "entry",
        get_symbols=lambda: {"BTC/USD"},
    )

    # Already persisted; no logging and no re-persist call
    assert db.logged == []
    assert db.recorded is None
    assert ResyncService._trade_key(persisted_trade[0]) in processed_ids
    assert ResyncService._client_key(persisted_trade[1]) in processed_ids

@pytest.mark.asyncio
async def test_sync_trades_requires_portfolio():
    db = SyncStubDB()
    bot = SyncStubBot(lambda *_: [])
    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=SimpleNamespace(),
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError):
        await resync.sync_trades_from_exchange(
            processed_trade_ids=set(),
            order_reasons={},
            plan_reason_lookup=lambda *_: None,
            get_symbols=lambda: set(),
        )


@pytest.mark.asyncio
async def test_reconcile_open_orders_requires_portfolio_scope():
    resync = ResyncService(
        db=StubDB(),
        bot=StubBot(),
        risk_manager=SimpleNamespace(update_positions=lambda *_: None, update_pending_orders=lambda *_: None),
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError):
        await resync.reconcile_open_orders()
