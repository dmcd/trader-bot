from datetime import datetime, timedelta

import pytest

from trader_bot.database import TradingDatabase


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "db.sqlite"
    db = TradingDatabase(str(db_path))
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="test")
    try:
        yield db, session_id
    finally:
        db.close()


def test_health_state_roundtrip(db_session):
    db, _ = db_session
    db.set_health_state("circuit", "open", detail='{"reason":"risk"}')
    db.set_health_state("circuit", "closed", detail=None)

    state = db.get_health_state()

    assert len(state) == 1
    assert state[0]["value"] == "closed"
    assert state[0]["detail"] is None


def test_prune_commands_retains_pending(db_session):
    db, _ = db_session
    cursor = db.conn.cursor()
    old_ts = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO commands (command, status, created_at, executed_at) VALUES (?, 'executed', ?, ?)",
        ("OLD_EXEC", old_ts, old_ts),
    )
    cursor.execute(
        "INSERT INTO commands (command, status, created_at) VALUES (?, 'pending', ?)",
        ("STILL_PENDING", old_ts),
    )
    db.conn.commit()

    db.prune_commands(retention_days=7)
    cursor.execute("SELECT command, status FROM commands")
    remaining = {row["command"]: row["status"] for row in cursor.fetchall()}

    assert remaining == {"STILL_PENDING": "pending"}


def test_trade_plan_crud_and_versioning(db_session):
    db, session_id = db_session
    plan_id = db.create_trade_plan(
        session_id,
        symbol="BTC/USD",
        side="long",
        entry_price=100.0,
        stop_price=90.0,
        target_price=120.0,
        size=1.0,
        reason="entry",
        entry_client_order_id="cid-1",
    )
    db.update_trade_plan_prices(plan_id, stop_price=95.0, target_price=125.0, reason="tighten")
    db.update_trade_plan_size(plan_id, size=0.5, reason="partial")

    open_plans = db.get_open_trade_plans(session_id)
    assert len(open_plans) == 1
    assert open_plans[0]["version"] == 3
    assert open_plans[0]["size"] == 0.5
    assert open_plans[0]["reason"] == "partial"

    db.update_trade_plan_status(plan_id, "closed", closed_at="2024-01-01T00:00:00Z", reason="exit")
    assert db.get_open_trade_plans(session_id) == []
    reason = db.get_trade_plan_reason_by_order(session_id, client_order_id="cid-1")
    assert reason == "exit"


def test_equity_snapshot_logging_and_pruning(db_session):
    db, session_id = db_session
    db.log_equity_snapshot(session_id, 100.0)
    db.log_equity_snapshot(session_id, 150.0)

    assert db.get_latest_equity(session_id) == 150.0
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM equity_snapshots WHERE equity < 150")
    db.conn.commit()
    assert db.get_latest_equity(session_id) == 150.0


def test_get_open_orders_handles_empty_table(db_session):
    db, session_id = db_session
    assert db.get_open_orders(session_id) == []


def test_log_llm_trace_handles_bad_json(db_session):
    db, session_id = db_session
    cyclical = []
    cyclical.append(cyclical)

    trace_id = db.log_llm_trace(session_id, "prompt", "resp", decision_json="{}", market_context=cyclical)
    cursor = db.conn.cursor()
    cursor.execute("SELECT market_context FROM llm_traces WHERE id = ?", (trace_id,))
    stored = cursor.fetchone()["market_context"]

    assert stored == str(cyclical)
