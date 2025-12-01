import logging
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from trader_bot.services.plan_monitor import PlanMonitor, PlanMonitorConfig


class StubDB:
    def __init__(self, plans):
        self.plans = plans
        self.updated_prices = []
        self.closed = []
        self.logged_trades = []

    def get_open_trade_plans_for_portfolio(self, portfolio_id):
        return self.plans

    def update_trade_plan_prices(self, plan_id, *, stop_price=None, target_price=None, reason=None, **kwargs):
        record = {
            "plan_id": plan_id,
            "stop_price": stop_price,
            "target_price": target_price,
            "reason": reason,
        }
        record.update(kwargs)
        self.updated_prices.append(record)

    def update_trade_plan_status(self, plan_id, *, status, closed_at, reason):
        self.closed.append((plan_id, status, closed_at, reason))

    def log_trade_for_portfolio(self, *args, **kwargs):
        self.logged_trades.append((args, kwargs))


class StubRiskManager:
    def __init__(self, positions=None):
        self.positions = positions or {}

    def get_total_exposure(self):
        return 0.0


def _monitor_with(db, risk_manager=None, bot=None):
    bot = bot or AsyncMock()
    bot.place_order_async = AsyncMock(return_value={"order_id": "1", "liquidity": "taker"})
    cost_tracker = MagicMock()
    cost_tracker.calculate_trade_fee.return_value = 0.0
    risk_manager = risk_manager or StubRiskManager()
    return PlanMonitor(
        db=db,
        bot=bot,
        cost_tracker=cost_tracker,
        risk_manager=risk_manager,
        prefer_maker=lambda symbol: False,
        holdings_updater=lambda *args, **kwargs: 0.0,
        portfolio_stats_applier=lambda *args, **kwargs: None,
    ), bot


@pytest.mark.asyncio
async def test_trails_sell_stop_to_breakeven_without_closing():
    now = datetime.now(timezone.utc)
    db = StubDB(
        plans=[
            {
                "id": 10,
                "symbol": "ETH/USD",
                "side": "SELL",
                "entry_price": 100.0,
                "stop_price": 110.0,
                "target_price": None,
                "size": 0.2,
                "opened_at": now.isoformat(),
            }
        ]
    )
    monitor, bot = _monitor_with(db, risk_manager=StubRiskManager({"ETH/USD": {"quantity": -0.2}}))
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.05)

    await monitor.monitor(
        price_lookup={"ETH/USD": 85.0},
        open_orders=[],
        config=config,
        now=now,
        portfolio_id=1,
    )

    assert db.updated_prices == [
        {
            "plan_id": 10,
            "stop_price": 100.0,
            "target_price": None,
            "reason": "Trailed stop to breakeven",
        }
    ]
    bot.place_order_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_skips_when_price_missing():
    now = datetime.now(timezone.utc)
    db = StubDB(
        plans=[
            {
                "id": 11,
                "symbol": "SOL/USD",
                "side": "BUY",
                "entry_price": 50.0,
                "stop_price": 45.0,
                "target_price": 60.0,
                "size": 1.0,
                "opened_at": now.isoformat(),
            }
        ]
    )
    monitor, bot = _monitor_with(db, risk_manager=StubRiskManager({"SOL/USD": {"quantity": 1.0}}))
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.01)

    await monitor.monitor(
        price_lookup={},
        open_orders=[],
        config=config,
        now=now,
        portfolio_id=1,
    )

    bot.place_order_async.assert_not_awaited()
    assert not db.closed
    assert not db.updated_prices


@pytest.mark.asyncio
async def test_closes_when_flat_and_no_symbol_orders():
    now = datetime.now(timezone.utc)
    db = StubDB(
        plans=[
            {
                "id": 12,
                "symbol": "BTC/USD",
                "side": "BUY",
                "entry_price": 25_000.0,
                "stop_price": 24_000.0,
                "target_price": 26_000.0,
                "size": 0.05,
                "opened_at": now.isoformat(),
            }
        ]
    )
    monitor, bot = _monitor_with(db, risk_manager=StubRiskManager())
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.02)

    await monitor.monitor(
        price_lookup={"BTC/USD": 25_100.0},
        open_orders=[{"symbol": "ETH/USD", "id": "ignored"}],
        config=config,
        now=now,
        portfolio_id=99,
    )

    bot.place_order_async.assert_awaited_once()
    assert db.closed
    assert db.logged_trades


@pytest.mark.asyncio
async def test_plan_monitor_requests_marketable_exit():
    now = datetime.now(timezone.utc)
    db = StubDB(
        plans=[
            {
                "id": 13,
                "symbol": "BHP/AUD",
                "side": "BUY",
                "entry_price": 100.0,
                "stop_price": 95.0,
                "target_price": 110.0,
                "size": 5.0,
                "opened_at": now.isoformat(),
            }
        ]
    )
    bot = AsyncMock()
    bot.place_order_async = AsyncMock(return_value={"order_id": "13", "liquidity": "taker"})
    monitor, _ = _monitor_with(db, risk_manager=StubRiskManager({"BHP/AUD": {"quantity": 5.0}}), bot=bot)
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.02)

    await monitor.monitor(
        price_lookup={"BHP/AUD": 111.0},
        open_orders=[],
        config=config,
        now=now,
        portfolio_id=1,
    )

    bot.place_order_async.assert_awaited_once()
    kwargs = bot.place_order_async.call_args.kwargs
    assert kwargs["force_market"] is True
    assert kwargs["prefer_maker"] is False


@pytest.mark.asyncio
async def test_handles_db_failure_gracefully(caplog):
    class FailingDB(StubDB):
        def get_open_trade_plans_for_portfolio(self, portfolio_id):
            raise RuntimeError("db unavailable")

    monitor, _ = _monitor_with(FailingDB(plans=[]))
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.01)

    with caplog.at_level(logging.WARNING):
        await monitor.monitor(
            price_lookup={"BTC/USD": 100.0},
            open_orders=[],
            config=config,
            now=datetime.now(timezone.utc),
            portfolio_id=1,
        )

    assert any("Monitor trade plans failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_plans_not_forced_flat_at_day_boundary():
    opened = datetime(2024, 1, 1, 14, tzinfo=timezone.utc)
    db = StubDB(
        plans=[
            {
                "id": 21,
                "symbol": "BTC/USD",
                "side": "BUY",
                "entry_price": 100.0,
                "stop_price": 95.0,
                "target_price": 120.0,
                "size": 0.25,
                "opened_at": opened.isoformat(),
            }
        ]
    )
    monitor, bot = _monitor_with(db, risk_manager=StubRiskManager({"BTC/USD": {"quantity": 0.25}}))
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.02)

    after_cutoff = opened + timedelta(days=1)
    after_cutoff = after_cutoff.replace(hour=21)

    await monitor.monitor(price_lookup={"BTC/USD": 101.0}, open_orders=[], config=config, now=after_cutoff, portfolio_id=1)
    assert bot.place_order_async.await_count == 0
    assert not db.closed


@pytest.mark.asyncio
async def test_overnight_widen_applies_once_with_restart_metadata():
    opened = datetime(2024, 1, 1, 22, tzinfo=timezone.utc)
    now = opened + timedelta(days=1, hours=1)
    db = StubDB(
        plans=[
            {
                "id": 30,
                "symbol": "BTC/USD",
                "side": "BUY",
                "entry_price": 100.0,
                "stop_price": 95.0,
                "target_price": 110.0,
                "size": 0.5,
                "opened_at": opened.isoformat(),
                "version": 1,
            }
        ]
    )
    monitor, _ = _monitor_with(db, risk_manager=StubRiskManager({"BTC/USD": {"quantity": 0.5}}))
    config = PlanMonitorConfig(
        max_plan_age_minutes=None,
        trail_to_breakeven_pct=0.02,
        overnight_widen_enabled=True,
        overnight_widen_pct=0.02,
        overnight_widen_abs=0.0,
        overnight_widen_max_pct=0.1,
        portfolio_day_timezone="UTC",
    )

    await monitor.monitor(
        price_lookup={"BTC/USD": 100.0},
        open_orders=[],
        config=config,
        now=now,
        portfolio_id=1,
    )

    assert len(db.updated_prices) == 1
    record = db.updated_prices[0]
    assert record["stop_price"] == pytest.approx(93.0)
    assert record["target_price"] == pytest.approx(110.0)  # clamped by max widen pct
    assert record["widen_version"] == 2
    assert record.get("widened_at", "").startswith(now.date().isoformat())

    # Persist metadata for the next monitor run and ensure widen is only applied once per day
    db.plans[0]["stop_price"] = record["stop_price"]
    db.plans[0]["target_price"] = record["target_price"]
    db.plans[0]["version"] = record["widen_version"]
    db.plans[0]["overnight_widened_at"] = record["widened_at"]
    db.updated_prices = []

    await monitor.monitor(
        price_lookup={"BTC/USD": 101.0},
        open_orders=[],
        config=config,
        now=now,
        portfolio_id=1,
    )

    assert db.updated_prices == []


@pytest.mark.asyncio
async def test_overnight_widen_respects_opt_out():
    opened = datetime(2024, 2, 1, 23, tzinfo=timezone.utc)
    db = StubDB(
        plans=[
            {
                "id": 31,
                "symbol": "ETH/USD",
                "side": "SELL",
                "entry_price": 2000.0,
                "stop_price": 2100.0,
                "target_price": 1800.0,
                "size": 0.2,
                "opened_at": opened.isoformat(),
            }
        ]
    )
    monitor, _ = _monitor_with(db, risk_manager=StubRiskManager({"ETH/USD": {"quantity": -0.2}}))
    config = PlanMonitorConfig(
        max_plan_age_minutes=None,
        trail_to_breakeven_pct=0.02,
        overnight_widen_enabled=False,
        overnight_widen_pct=0.05,
        overnight_widen_abs=5.0,
        overnight_widen_max_pct=0.2,
        portfolio_day_timezone="UTC",
    )

    await monitor.monitor(
        price_lookup={"ETH/USD": 1990.0},
        open_orders=[],
        config=config,
        now=opened + timedelta(days=1),
        portfolio_id=1,
    )

    assert db.updated_prices == []


def test_rearm_after_restart_applies_widen_policy():
    opened = datetime(2024, 3, 1, 21, tzinfo=timezone.utc)
    plan = {
        "id": 40,
        "symbol": "SOL/USD",
        "side": "BUY",
        "entry_price": 100.0,
        "stop_price": 92.0,
        "target_price": 120.0,
        "size": 1.0,
        "opened_at": opened.isoformat(),
        "version": 1,
    }
    db = StubDB(plans=[plan])
    monitor, _ = _monitor_with(db, risk_manager=StubRiskManager({"SOL/USD": {"quantity": 1.0}}))
    monitor.portfolio_id = 5
    config = PlanMonitorConfig(
        max_plan_age_minutes=None,
        trail_to_breakeven_pct=0.02,
        overnight_widen_enabled=True,
        overnight_widen_pct=0.01,
        overnight_widen_abs=0.5,
        overnight_widen_max_pct=0.1,
        auto_rearm_on_restart=True,
        portfolio_day_timezone="UTC",
    )

    updated = monitor.rearm_after_restart([plan], config=config, now=opened + timedelta(days=1))

    assert len(db.updated_prices) == 1
    widened = db.updated_prices[0]
    assert widened["stop_price"] < plan["stop_price"]
    assert updated[0]["stop_price"] == widened["stop_price"]


@pytest.mark.asyncio
async def test_monitor_requires_portfolio_id():
    monitor, _ = _monitor_with(StubDB(plans=[]))
    config = PlanMonitorConfig(max_plan_age_minutes=None, trail_to_breakeven_pct=0.01)

    with pytest.raises(ValueError):
        await monitor.monitor(price_lookup={}, open_orders=[], config=config, now=datetime.now(timezone.utc), portfolio_id=None)
