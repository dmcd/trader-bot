import logging
from types import SimpleNamespace

import pytest

from trader_bot.services.portfolio_tracker import PortfolioTracker


class FakeDB:
    def __init__(self):
        self.cached_stats = None
        self.session_totals = None
        self.trades = []
        self.session_stats_row = {"total_llm_cost": 0.0}

    def set_session_stats_cache(self, session_id, stats):
        self.cached_stats = (session_id, stats.copy())

    def get_session_stats(self, session_id):
        return self.session_stats_row

    def update_session_totals(self, session_id, **kwargs):
        self.session_totals = (session_id, kwargs)

    def get_trades_for_session(self, session_id):
        return list(self.trades)


def test_apply_fill_updates_holdings_and_stats():
    db = FakeDB()
    tracker = PortfolioTracker(db, session_id=1, logger=logging.getLogger("test"))

    # Buy 1 @ 100, then sell 0.5 @ 110 with $1 fee
    tracker.update_holdings_and_realized("BTC/USD", "BUY", 1.0, 100.0, 0.0)
    realized = tracker.update_holdings_and_realized("BTC/USD", "SELL", 0.5, 110.0, 1.0)
    tracker.apply_fill_to_session_stats(order_id="abc", actual_fee=1.0, realized_pnl=realized, estimated_fee_map={"abc": 0.2})

    assert tracker.holdings["BTC/USD"]["qty"] == pytest.approx(0.5)
    assert tracker.session_stats["total_trades"] == 1
    assert tracker.session_stats["total_fees"] == pytest.approx(1.0)
    assert tracker.session_stats["gross_pnl"] == pytest.approx(5.0)
    # Cache write captured
    assert db.cached_stats[0] == 1


def test_rebuild_session_stats_from_trades_applies_fees_and_realized():
    db = FakeDB()
    db.trades = [
        {"symbol": "ETH/USD", "action": "BUY", "quantity": 1.0, "price": 2000.0, "fee": {"cost": 0.5}},
        {"symbol": "ETH/USD", "action": "SELL", "quantity": 1.0, "price": 2100.0, "fee": {"cost": 0.5}},
    ]
    db.session_stats_row = {"total_llm_cost": 2.0}
    tracker = PortfolioTracker(db, session_id=7, logger=logging.getLogger("test"))

    stats = tracker.rebuild_session_stats_from_trades()

    assert stats["total_trades"] == 2
    assert stats["total_fees"] == pytest.approx(1.0)
    assert stats["gross_pnl"] == pytest.approx(100.0)
    assert stats["total_llm_cost"] == pytest.approx(2.0)
    assert tracker.holdings.get("ETH/USD", {}).get("qty") == pytest.approx(0.0)
    assert db.session_totals[0] == 7


def test_apply_exchange_trades_for_rebuild_skips_bad_entries(caplog):
    caplog.set_level(logging.WARNING)
    db = FakeDB()
    tracker = PortfolioTracker(db, session_id=3, logger=logging.getLogger("test"))
    valid_trade = {"symbol": "BTC/USD", "side": "buy", "amount": 1, "price": 100.0, "fee": {"cost": 0.1}}
    malformed_trade = {"symbol": "BTC/USD", "side": "sell", "amount": -1, "price": 100.0}

    stats = tracker.apply_exchange_trades_for_rebuild([valid_trade, malformed_trade])

    assert stats["total_trades"] == 1
    assert "Skipped 1 malformed trades" in "\n".join(caplog.messages)


def test_extract_fee_cost_aggregates_sequences():
    total = PortfolioTracker.extract_fee_cost([{"cost": 1.0}, {"cost": "0.5"}, None])
    assert total == pytest.approx(1.5)
    assert PortfolioTracker.extract_fee_cost("bad") == 0.0


def test_load_holdings_resets_before_rebuild():
    db = FakeDB()
    db.trades = [
        {"symbol": "BTC/USD", "action": "BUY", "quantity": 1.0, "price": 100.0},
        {"symbol": "BTC/USD", "action": "SELL", "quantity": 1.0, "price": 110.0},
    ]
    tracker = PortfolioTracker(db, session_id=9, logger=logging.getLogger("test"))
    tracker.holdings["OLD"] = {"qty": 5.0, "avg_cost": 10.0}

    tracker.load_holdings_from_db()

    assert "OLD" not in tracker.holdings
    assert tracker.holdings["BTC/USD"]["qty"] == pytest.approx(0.0)


def test_apply_exchange_trades_for_rebuild_accumulates_fees():
    db = FakeDB()
    tracker = PortfolioTracker(db, session_id=4, logger=logging.getLogger("test"))
    trades = [
        {"symbol": "BTC/USD", "side": "buy", "amount": 1, "price": 100.0, "fee": [{"cost": 0.1}, {"cost": 0.2}]},
        {"symbol": "BTC/USD", "side": "sell", "amount": 1, "price": 110.0, "fee": {"cost": 0.05}},
    ]

    stats = tracker.apply_exchange_trades_for_rebuild(trades)

    assert stats["total_trades"] == 2
    assert stats["gross_pnl"] == pytest.approx(10.0)
    assert stats["total_fees"] == pytest.approx(0.35)
