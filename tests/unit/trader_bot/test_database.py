from datetime import datetime, timedelta

import pytest

from trader_bot.database import TradingDatabase


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "db.sqlite"
    db = TradingDatabase(str(db_path))
    session_id = db.get_or_create_session(starting_balance=5000.0, bot_version="test-version")
    try:
        yield db, session_id
    finally:
        db.close()


def _insert_market_data_rows(db, session_id, old_ts, recent_ts):
    cursor = db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO market_data (session_id, timestamp, symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, old_ts.isoformat(), "BTC/USD", 10.0, 9.5, 10.5, 1.0, None, None, None, None),
    )
    cursor.execute(
        """
        INSERT INTO market_data (session_id, timestamp, symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, recent_ts.isoformat(), "BTC/USD", 11.0, 10.5, 11.5, 2.0, None, None, None, None),
    )
    db.conn.commit()


def _insert_llm_traces(db, session_id, old_ts, recent_ts):
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO llm_traces (session_id, timestamp, prompt, response, decision_json, market_context) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, old_ts.isoformat(), "old", "resp", "{}", "{}"),
    )
    cursor.execute(
        "INSERT INTO llm_traces (session_id, timestamp, prompt, response, decision_json, market_context) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, recent_ts.isoformat(), "new", "resp", "{}", "{}"),
    )
    db.conn.commit()


def test_log_and_fetch_entities(db_session):
    db, session_id = db_session
    db.log_trade(session_id, "BTC/USD", "BUY", 0.1, 20000.0, fee=1.0, reason="test")
    db.log_llm_call(session_id, input_tokens=10, output_tokens=5, cost=0.001, decision="{}")
    db.log_market_data(session_id, "BTC/USD", price=20000.0, bid=19990.0, ask=20010.0, volume=1000.0)
    db.log_equity_snapshot(session_id, equity=5050.0)

    trades = db.get_recent_trades(session_id, limit=5)
    assert len(trades) == 1

    stats = db.get_session_stats(session_id)
    assert stats["total_trades"] == 1
    assert stats["total_fees"] >= 1.0

    latest_equity = db.get_latest_equity(session_id)
    assert latest_equity == 5050.0

    positions = db.get_net_positions_from_trades(session_id)
    assert positions["BTC/USD"] == pytest.approx(0.1)


def test_start_of_day_equity_persistence(db_session):
    db, session_id = db_session
    baseline = 1234.5
    db.set_start_of_day_equity(session_id, baseline)

    stored = db.get_start_of_day_equity(session_id)
    assert stored == baseline


def test_log_and_fetch_ohlcv(db_session):
    db, session_id = db_session
    bars = [
        {"timestamp": 1_000_000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
        {"timestamp": 1_000_060, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 20},
    ]
    db.log_ohlcv_batch(session_id, "BTC/USD", "1m", bars)
    fetched = db.get_recent_ohlcv(session_id, "BTC/USD", "1m", limit=5)

    assert len(fetched) == 2
    assert fetched[0]["close"] == pytest.approx(2.0)


def test_prune_ohlcv_retains_latest_rows(db_session):
    db, session_id = db_session
    bars = []
    for idx in range(6):
        bars.append(
            {
                "timestamp": 1_000_000 + (idx * 60),
                "open": float(idx),
                "high": float(idx) + 1,
                "low": float(idx),
                "close": float(idx) + 0.5,
                "volume": idx + 1,
            }
        )
    db.log_ohlcv_batch(session_id, "ETH/USD", "5m", bars)
    db.prune_ohlcv(session_id, "ETH/USD", "5m", retain=3)

    remaining = db.get_recent_ohlcv(session_id, "ETH/USD", "5m", limit=10)
    assert len(remaining) == 3
    assert remaining[0]["close"] == pytest.approx(5.5)
    assert remaining[-1]["close"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "setup_fn, prune_call, count_query",
    [
        (
            lambda db, session_id: _insert_market_data_rows(
                db,
                session_id,
                datetime.now() - timedelta(minutes=10),
                datetime.now(),
            ),
            lambda db, session_id: db.prune_market_data(session_id, retention_minutes=5),
            "SELECT COUNT(*) as cnt FROM market_data WHERE session_id = ?",
        ),
        (
            lambda db, session_id: _insert_llm_traces(
                db,
                session_id,
                datetime.now() - timedelta(days=10),
                datetime.now(),
            ),
            lambda db, session_id: db.prune_llm_traces(session_id, retention_days=7),
            "SELECT COUNT(*) as cnt FROM llm_traces WHERE session_id = ?",
        ),
    ],
)
def test_prune_tables_drop_old_rows(db_session, setup_fn, prune_call, count_query):
    db, session_id = db_session
    setup_fn(db, session_id)

    prune_call(db, session_id)

    cursor = db.conn.cursor()
    cursor.execute(count_query, (session_id,))
    assert cursor.fetchone()["cnt"] == 1


def test_prune_commands_drops_old_executed(db_session):
    db, _ = db_session
    cursor = db.conn.cursor()
    old_ts = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    recent_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO commands (command, status, created_at, executed_at) VALUES (?, 'executed', ?, ?)",
        ("OLD", old_ts, old_ts),
    )
    cursor.execute(
        "INSERT INTO commands (command, status, created_at, executed_at) VALUES (?, 'executed', ?, ?)",
        ("RECENT", recent_ts, recent_ts),
    )
    db.conn.commit()

    db.prune_commands(retention_days=7)
    cursor.execute("SELECT command FROM commands")
    remaining = {row["command"] for row in cursor.fetchall()}
    assert remaining == {"RECENT"}


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


def test_health_state_roundtrip(db_session):
    db, _ = db_session
    db.set_health_state("circuit", "open", detail='{"reason":"risk"}')
    db.set_health_state("circuit", "closed", detail=None)

    state = db.get_health_state()

    assert len(state) == 1
    assert state[0]["value"] == "closed"
    assert state[0]["detail"] is None


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


def test_multiple_sessions_created_per_version(db_session):
    db, first_session = db_session
    next_session = db.get_or_create_session(starting_balance=6000.0, bot_version="test-version")

    cursor = db.conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM sessions WHERE bot_version = ?", ("test-version",))
    row = cursor.fetchone()

    assert first_session != next_session
    assert row["count"] == 2
    assert db.get_session_id_by_version("test-version") == next_session


def test_session_creation_does_not_reuse_on_restart(tmp_path):
    db_path = tmp_path / "restart.db"
    original = TradingDatabase(str(db_path))
    first_session = original.get_or_create_session(starting_balance=5000.0, bot_version="test-version")
    original.close()

    reopened = TradingDatabase(str(db_path))
    new_session = reopened.get_or_create_session(starting_balance=7000.0, bot_version="test-version")
    reopened.close()

    assert first_session != new_session
