import logging
from datetime import datetime, timedelta, timezone

import pytest

from trader_bot.config import CLIENT_ORDER_PREFIX
from trader_bot.database import TradingDatabase
from trader_bot.risk_manager import RiskManager
from trader_bot.services.plan_monitor import PlanMonitor, PlanMonitorConfig
from trader_bot.services.portfolio_tracker import PortfolioTracker
from trader_bot.services.resync_service import ResyncService

pytestmark = pytest.mark.integration


class StubBot:
    def __init__(self, trades=None):
        self.trades = list(trades or [])
        self.calls = 0

    async def get_my_trades_async(self, symbol, since=None, limit=100):
        self.calls += 1
        if self.calls > 1:
            return []
        trades = list(self.trades)
        self.trades = []
        return trades

    async def get_open_orders_async(self):
        return []


class StubCostTracker:
    def calculate_trade_fee(self, *args, **kwargs):
        return 0.0


@pytest.mark.asyncio
async def test_multi_day_position_survives_rollover(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "multi-day.db"))
    db = TradingDatabase()
    portfolio_id, _ = db.ensure_active_portfolio(name="multi-day", bot_version="multi-day")

    now = datetime.now(timezone.utc)
    day1 = now - timedelta(hours=30)
    day2 = day1 + timedelta(days=1, hours=1)

    db.replace_positions_for_portfolio(
        portfolio_id,
        [{"symbol": "ETH/USD", "quantity": 1.0, "avg_price": 1800.0, "current_price": 1820.0}],
    )
    plan_id = db.create_trade_plan_for_portfolio(
        portfolio_id,
        "ETH/USD",
        "BUY",
        entry_price=1800.0,
        stop_price=1700.0,
        target_price=1950.0,
        size=1.0,
        reason="swing",
        entry_order_id="order-plan",
        entry_client_order_id=f"{CLIENT_ORDER_PREFIX}-plan",
    )
    db.conn.execute("UPDATE trade_plans SET opened_at = ? WHERE id = ?", (day1.isoformat(), plan_id))
    db.conn.commit()

    tracker = PortfolioTracker(db, portfolio_id=portfolio_id, logger=logging.getLogger("test"))
    tracker.holdings["ETH/USD"] = {"qty": 1.0, "avg_cost": 1820.0}
    risk = RiskManager(base_currency="USD")
    risk.set_portfolio(portfolio_id)

    trade = {
        "id": "t-day2",
        "order": "order-plan",
        "side": "sell",
        "price": 1900.0,
        "amount": 0.2,
        "fee": {"cost": 1.0},
        "timestamp": int(day2.timestamp() * 1000),
        "datetime": day2.isoformat(),
        "clientOrderId": f"{CLIENT_ORDER_PREFIX}-plan",
    }
    bot = StubBot([trade])

    resync = ResyncService(
        db=db,
        bot=bot,
        risk_manager=risk,
        holdings_updater=tracker.update_holdings_and_realized,
        portfolio_stats_applier=tracker.apply_fill_to_portfolio_stats,
        portfolio_id=portfolio_id,
    )

    state = resync.bootstrap_snapshots()
    assert state["positions"]
    assert risk.positions["ETH/USD"]["quantity"] == pytest.approx(1.0)

    plan_monitor = PlanMonitor(
        db=db,
        bot=None,
        cost_tracker=StubCostTracker(),
        risk_manager=risk,
        prefer_maker=lambda _symbol: False,
        holdings_updater=tracker.update_holdings_and_realized,
        portfolio_stats_applier=tracker.apply_fill_to_portfolio_stats,
        portfolio_id=portfolio_id,
    )
    config = PlanMonitorConfig(
        max_plan_age_minutes=None,
        trail_to_breakeven_pct=0.0,
        overnight_widen_enabled=True,
        overnight_widen_pct=0.01,
        overnight_widen_abs=10.0,
        overnight_widen_max_pct=0.05,
        auto_rearm_on_restart=True,
        portfolio_day_timezone="Australia/Sydney",
    )
    widened = plan_monitor.rearm_after_restart(state["open_plans"], config=config, now=day2)
    assert widened and widened[0]["overnight_widened_at"]

    plan_row = db.get_open_trade_plans_for_portfolio(portfolio_id)[0]
    assert plan_row["stop_price"] == pytest.approx(1710.0)  # clamped by 5% cap
    assert plan_row["target_price"] == pytest.approx(1890.0)
    assert plan_row["overnight_widen_version"] == plan_row["version"]

    processed_ids: set[str] = set()
    await resync.sync_trades_from_exchange(
        processed_trade_ids=processed_ids,
        order_reasons={"order-plan": "entry"},
        plan_reason_lookup=lambda pid, oid, coid: db.get_trade_plan_reason_by_order_for_portfolio(
            pid, order_id=oid, client_order_id=coid
        ),
        get_symbols=lambda: {"ETH/USD"},
    )

    assert db.get_trade_count_for_portfolio(portfolio_id) == 1
    persisted_pairs = db.get_processed_trade_entries_for_portfolio(portfolio_id)
    assert ("t-day2", f"{CLIENT_ORDER_PREFIX}-plan") in persisted_pairs
    assert tracker.portfolio_stats["total_trades"] == 1
    assert tracker.holdings["ETH/USD"]["qty"] == pytest.approx(0.8)
    assert risk.get_total_exposure() > 0.0

    db.close()
