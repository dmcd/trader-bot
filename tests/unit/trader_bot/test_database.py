from datetime import datetime, timedelta, timezone
import sqlite3

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


def test_session_base_currency_round_trip(tmp_path):
    db_path = tmp_path / "base_ccy.db"
    db = TradingDatabase(str(db_path))
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="aud", base_currency="aud")

    session = db.get_session(session_id)
    assert session["base_currency"] == "AUD"
    db.close()


def test_base_currency_column_present(tmp_path):
    db_path = tmp_path / "fresh.db"
    db = TradingDatabase(str(db_path))
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(sessions)")}
    assert "base_currency" in cols

    session_id = db.get_or_create_session(starting_balance=200.0, bot_version="new", base_currency=None)
    session = db.get_session(session_id)
    assert session["base_currency"] is None
    db.close()


def test_portfolio_creation_and_lookup(tmp_path):
    db_path = tmp_path / "portfolio.db"
    db = TradingDatabase(str(db_path))

    portfolio = db.get_or_create_portfolio("main", base_currency="aud", bot_version="v1")

    assert portfolio["id"] > 0
    assert portfolio["name"] == "main"
    assert portfolio["base_currency"] == "AUD"
    assert portfolio["bot_version"] == "v1"
    assert db.get_portfolio(portfolio["id"])["name"] == "main"
    db.close()


def test_portfolio_updates_metadata_when_reused(tmp_path):
    db_path = tmp_path / "portfolio-reuse.db"
    db = TradingDatabase(str(db_path))
    first = db.get_or_create_portfolio("swing", base_currency="usd", bot_version="v1")

    reused = db.get_or_create_portfolio("swing", base_currency="aud", bot_version="v2")

    assert reused["id"] == first["id"]
    assert reused["base_currency"] == "AUD"
    assert reused["bot_version"] == "v2"
    db.close()


@pytest.mark.parametrize(
    "table_name",
    [
        "trades",
        "processed_trades",
        "llm_calls",
        "llm_traces",
        "market_data",
        "ohlcv_bars",
        "equity_snapshots",
        "indicators",
        "positions",
        "open_orders",
    ],
)
def test_portfolio_column_added_to_core_tables(tmp_path, table_name):
    db_path = tmp_path / f"{table_name}.db"
    db = TradingDatabase(str(db_path))
    cols = {row["name"] for row in db.conn.execute(f"PRAGMA table_info({table_name})")}
    assert "portfolio_id" in cols
    db.close()


def test_trade_plan_portfolio_column_and_index(tmp_path):
    db_path = tmp_path / "plans.db"
    db = TradingDatabase(str(db_path))
    db.ensure_trade_plans_table()
    cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(trade_plans)")}
    assert "portfolio_id" in cols
    indexes = {row["name"] for row in db.conn.execute("PRAGMA index_list(trade_plans)")}
    assert "idx_trade_plans_portfolio_symbol_status" in indexes
    db.close()


def test_portfolio_indexes_created(tmp_path):
    db_path = tmp_path / "portfolio-index.db"
    db = TradingDatabase(str(db_path))
    ohlcv_indexes = {row["name"] for row in db.conn.execute("PRAGMA index_list(ohlcv_bars)")}
    market_indexes = {row["name"] for row in db.conn.execute("PRAGMA index_list(market_data)")}
    assert "idx_ohlcv_portfolio_symbol_tf_ts" in ohlcv_indexes
    assert "idx_market_data_portfolio_symbol_ts" in market_indexes
    db.close()


def test_run_id_columns_and_indexes(tmp_path):
    db_path = tmp_path / "runid.db"
    db = TradingDatabase(str(db_path))
    call_cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(llm_calls)")}
    trace_cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(llm_traces)")}
    assert "run_id" in call_cols
    assert "run_id" in trace_cols
    call_indexes = {row["name"] for row in db.conn.execute("PRAGMA index_list(llm_calls)")}
    trace_indexes = {row["name"] for row in db.conn.execute("PRAGMA index_list(llm_traces)")}
    assert "idx_llm_calls_run_ts" in call_indexes
    assert "idx_llm_traces_run_ts" in trace_indexes
    db.close()


def test_portfolio_stats_cache_roundtrip(tmp_path):
    db_path = tmp_path / "portfolio-stats.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("stats", base_currency="usd", bot_version="v1")

    db.set_portfolio_stats_cache(portfolio["id"], {"total_trades": 3, "gross_pnl": 10.0, "total_fees": 1.0})
    cached = db.get_portfolio_stats_cache(portfolio["id"])

    assert cached["total_trades"] == 3
    assert cached["gross_pnl"] == pytest.approx(10.0)
    assert cached["total_fees"] == pytest.approx(1.0)
    db.close()


def test_processed_trades_store_portfolio(tmp_path):
    db_path = tmp_path / "portfolio-processed.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("proc", base_currency="usd", bot_version="v1")
    session_id = db.get_or_create_session(1000.0, "v1", portfolio_id=portfolio["id"])

    db.record_processed_trade_ids(session_id, [("tid-1", "cid-1")], portfolio_id=portfolio["id"])
    row = db.conn.execute("SELECT portfolio_id FROM processed_trades WHERE trade_id = 'tid-1'").fetchone()

    assert row["portfolio_id"] == portfolio["id"]
    db.close()


def test_trades_can_be_fetched_by_portfolio(tmp_path):
    db_path = tmp_path / "portfolio-trades.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("p1", base_currency="usd", bot_version="v1")
    session_id = db.get_or_create_session(1000.0, "v1", portfolio_id=portfolio["id"])
    db.log_trade(session_id, "BTC/USD", "BUY", 0.1, 20000.0, fee=0.0, portfolio_id=portfolio["id"])

    trades = db.get_recent_trades(portfolio_id=portfolio["id"], limit=5)

    assert trades and trades[0]["portfolio_id"] == portfolio["id"]
    db.close()


def test_session_portfolios_view_exists(tmp_path):
    db_path = tmp_path / "session-portfolio-view.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("viewp", base_currency="usd", bot_version="v1")
    session_id = db.get_or_create_session(1000.0, "v1", portfolio_id=portfolio["id"])
    row = db.conn.execute("SELECT portfolio_id FROM session_portfolios WHERE session_id = ?", (session_id,)).fetchone()

    assert row["portfolio_id"] == portfolio["id"]
    db.close()


def test_session_portfolio_backfill(tmp_path):
    db_path = tmp_path / "session-backfill.db"
    db = TradingDatabase(str(db_path))
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (date, bot_version, starting_balance, base_currency) VALUES (?, ?, ?, ?)",
        ("2024-01-01", "v1", 1000.0, "USD"),
    )
    session_id = cursor.lastrowid
    db.conn.commit()

    updated = db.backfill_session_portfolios()
    portfolio_id = db.get_session_portfolio_id(session_id)

    assert updated == 1
    assert portfolio_id is not None
    portfolio = db.get_portfolio(portfolio_id)
    assert portfolio["bot_version"] == "v1"
    db.close()


def test_log_trade_sets_portfolio_id(tmp_path):
    db_path = tmp_path / "portfolio-trade.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("active", base_currency="usd", bot_version="v1")
    session_id = db.get_or_create_session(starting_balance=1000.0, bot_version="v1", base_currency="usd", portfolio_id=portfolio["id"])

    db.log_trade(session_id, "BTC/USD", "BUY", 0.1, 20000.0, fee=0.5, portfolio_id=portfolio["id"])
    row = db.conn.execute("SELECT portfolio_id FROM trades WHERE session_id = ?", (session_id,)).fetchone()

    assert row["portfolio_id"] == portfolio["id"]
    db.close()


def test_market_data_preserves_integer_sizes(db_session):
    db, session_id = db_session
    db.log_market_data(
        session_id,
        "BHP/AUD",
        price=100.0,
        bid=99.5,
        ask=100.5,
        volume=200,
        spread_pct=0.5,
        bid_size=150,
        ask_size=120,
        ob_imbalance=0.1,
    )

    rows = db.get_recent_market_data(session_id, "BHP/AUD", limit=1)
    assert rows[0]["volume"] == 200
    assert isinstance(rows[0]["volume"], int)
    assert rows[0]["bid_size"] == 150
    assert rows[0]["ask_size"] == 120


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


def test_portfolio_day_updates_from_equity_snapshots(tmp_path):
    db_path = tmp_path / "portfolio-days.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio(name="swing", base_currency="AUD", bot_version="v1")
    session_id = db.get_or_create_portfolio_session(
        portfolio_id=portfolio["id"],
        starting_balance=1000.0,
        bot_version="v1",
    )

    tz_name = "Australia/Melbourne"
    first_ts = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)  # midnight local on Jan 2
    db.log_equity_snapshot(
        session_id,
        equity=1000.0,
        portfolio_id=portfolio["id"],
        timestamp=first_ts,
        timezone_name=tz_name,
    )
    second_ts = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)  # still Jan 2 locally
    db.log_equity_snapshot(
        session_id,
        equity=1100.0,
        portfolio_id=portfolio["id"],
        timestamp=second_ts,
        timezone_name=tz_name,
    )

    cursor = db.conn.cursor()
    row = cursor.execute(
        """
        SELECT date, timezone, start_equity, end_equity, gross_pnl, net_pnl
        FROM portfolio_days
        WHERE portfolio_id = ?
        """,
        (portfolio["id"],),
    ).fetchone()

    assert row["date"] == "2024-01-02"
    assert row["timezone"] == tz_name
    assert row["start_equity"] == pytest.approx(1000.0)
    assert row["end_equity"] == pytest.approx(1100.0)
    assert row["gross_pnl"] == pytest.approx(100.0)
    assert row["net_pnl"] == pytest.approx(100.0)

    third_ts = datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)  # Jan 3 locally
    db.log_equity_snapshot(
        session_id,
        equity=1050.0,
        portfolio_id=portfolio["id"],
        timestamp=third_ts,
        timezone_name=tz_name,
    )

    rows = cursor.execute(
        """
        SELECT date, start_equity, end_equity
        FROM portfolio_days
        WHERE portfolio_id = ?
        ORDER BY date
        """,
        (portfolio["id"],),
    ).fetchall()

    assert len(rows) == 2
    assert rows[0]["date"] == "2024-01-02"
    assert rows[0]["start_equity"] == pytest.approx(1000.0)
    assert rows[0]["end_equity"] == pytest.approx(1100.0)
    assert rows[1]["date"] == "2024-01-03"
    assert rows[1]["start_equity"] == pytest.approx(1050.0)
    assert rows[1]["end_equity"] == pytest.approx(1050.0)


def test_get_open_orders_handles_empty_table(db_session):
    db, session_id = db_session
    assert db.get_open_orders(session_id) == []


def test_replace_positions_removes_legacy_session_rows(tmp_path):
    db_path = tmp_path / "positions-clean.db"
    db = TradingDatabase(str(db_path))
    session_id = db.get_or_create_session(starting_balance=5000.0, bot_version="legacy")
    portfolio = db.get_or_create_portfolio("swing", base_currency="USD", bot_version="v1")

    db.replace_positions(session_id, [{"symbol": "ETH/USD", "quantity": 2.0, "avg_price": 2000.0}])
    db.replace_positions(
        session_id,
        [{"symbol": "BTC/USD", "quantity": 1.0, "avg_price": 10000.0}],
        portfolio_id=portfolio["id"],
    )

    positions = db.get_positions(session_id)
    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTC/USD"
    assert positions[0]["portfolio_id"] == portfolio["id"]
    db.close()


def test_replace_open_orders_removes_legacy_session_rows(tmp_path):
    db_path = tmp_path / "orders-clean.db"
    db = TradingDatabase(str(db_path))
    session_id = db.get_or_create_session(starting_balance=5000.0, bot_version="legacy")
    portfolio = db.get_or_create_portfolio("swing", base_currency="USD", bot_version="v1")

    db.replace_open_orders(
        session_id,
        [{"order_id": "old", "symbol": "ETH/USD", "side": "sell", "price": 2100, "amount": 1, "remaining": 1}],
    )
    db.replace_open_orders(
        session_id,
        [{"order_id": "fresh", "symbol": "BTC/USD", "side": "buy", "price": 10100, "amount": 1, "remaining": 1}],
        portfolio_id=portfolio["id"],
    )

    orders = db.get_open_orders(session_id)
    assert len(orders) == 1
    assert orders[0]["order_id"] == "fresh"
    assert orders[0]["portfolio_id"] == portfolio["id"]
    db.close()


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


def test_get_or_create_portfolio_session_reuses_latest(tmp_path):
    db_path = tmp_path / "portfolio.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio(name="swing", base_currency="usd", bot_version="v1")

    first = db.get_or_create_portfolio_session(portfolio_id=portfolio["id"], starting_balance=1200.0, bot_version="v1")
    second = db.get_or_create_portfolio_session(portfolio_id=portfolio["id"], starting_balance=3300.0, bot_version="v1")

    cursor = db.conn.cursor()
    row = cursor.execute("SELECT COUNT(*) AS cnt FROM sessions WHERE portfolio_id = ?", (portfolio["id"],)).fetchone()

    assert first == second
    assert row["cnt"] == 1
    db.close()


def test_log_estimated_fee_persists_row(db_session):
    db, session_id = db_session
    db.log_estimated_fee(session_id, order_id="abc", estimated_fee=1.23, symbol="BTC/USD", action="BUY")

    cursor = db.conn.cursor()
    row = cursor.execute("SELECT estimated_fee, action, order_id FROM estimated_fees WHERE session_id = ?", (session_id,)).fetchone()
    assert row["estimated_fee"] == pytest.approx(1.23)
    assert row["action"] == "BUY"
    assert row["order_id"] == "abc"


def test_llm_trace_roundtrip_and_prune(db_session):
    db, session_id = db_session
    trace_id = db.log_llm_trace(session_id, prompt="p", response="r", decision_json="{}", market_context={"a": 1}, run_id="run-1")
    db.update_llm_trace_execution(trace_id, {"result": "ok"})
    traces = db.get_recent_llm_traces(session_id, limit=5)
    assert traces and traces[0]["id"] == trace_id
    assert '"result": "ok"' in (traces[0]["execution_result"] or "")
    cursor = db.conn.cursor()
    row = cursor.execute("SELECT run_id FROM llm_traces WHERE id = ?", (trace_id,)).fetchone()
    assert row["run_id"] == "run-1"

    old_ts = (datetime.now() - timedelta(days=10)).isoformat()
    db.conn.execute(
        "INSERT INTO llm_traces (session_id, timestamp, prompt, response, decision_json) VALUES (?, ?, ?, ?, ?)",
        (session_id, old_ts, "old", "resp", "{}"),
    )
    db.conn.commit()

    db.prune_llm_traces(session_id, retention_days=7)
    remaining = db.get_recent_llm_traces(session_id, limit=10)
    assert all(t["timestamp"] >= old_ts for t in remaining)


def test_recent_llm_stats_counts_flags(db_session):
    db, session_id = db_session
    db.log_llm_call(session_id, input_tokens=1, output_tokens=1, cost=0.0, decision="schema_error_missing")
    db.log_llm_call(session_id, input_tokens=1, output_tokens=1, cost=0.0, decision="clamped_size")
    db.log_llm_call(session_id, input_tokens=1, output_tokens=1, cost=0.0, decision="ok")

    stats = db.get_recent_llm_stats(session_id, limit=10)
    assert stats["total"] == 3
    assert stats["schema_errors"] == 1
    assert stats["clamped"] == 1


def test_session_stats_cache_upsert_and_merge(db_session):
    db, session_id = db_session
    db.set_session_stats_cache(session_id, {"total_trades": 2, "gross_pnl": 5.0})
    db.set_session_stats_cache(session_id, {"total_fees": 1.5})

    cached = db.get_session_stats_cache(session_id)
    assert cached["total_trades"] == 2
    assert cached["gross_pnl"] == pytest.approx(5.0)
    assert cached["total_fees"] == pytest.approx(1.5)


def test_session_stats_prefers_portfolio_cache(tmp_path):
    db_path = tmp_path / "portfolio-stats.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("swing", base_currency="USD", bot_version="v1")
    session_id = db.get_or_create_portfolio_session(
        portfolio_id=portfolio["id"],
        starting_balance=5000.0,
        bot_version="v1",
        base_currency="USD",
    )
    db.set_session_stats_cache(session_id, {"total_trades": 1, "gross_pnl": 10.0, "total_fees": 1.0})
    db.set_portfolio_stats_cache(
        portfolio["id"],
        {
            "total_trades": 7,
            "gross_pnl": 120.0,
            "total_fees": 5.0,
            "total_llm_cost": 2.5,
            "exposure_notional": 1500.0,
        },
    )

    stats = db.get_session_stats(session_id)

    assert stats["portfolio_id"] == portfolio["id"]
    assert stats["total_trades"] == 7
    assert stats["gross_pnl"] == pytest.approx(120.0)
    assert stats["total_fees"] == pytest.approx(5.0)
    assert stats["total_llm_cost"] == pytest.approx(2.5)
    assert stats["exposure_notional"] == pytest.approx(1500.0)
    db.close()


def test_command_lifecycle_roundtrip(db_session):
    db, _ = db_session
    db.create_command("DO_THIS")
    pending = db.get_pending_commands()
    assert pending and pending[0]["command"] == "DO_THIS"

    db.mark_command_executed(pending[0]["id"])
    executed = db.conn.execute("SELECT status FROM commands WHERE id = ?", (pending[0]["id"],)).fetchone()
    assert executed["status"] == "executed"

    db.create_command("CANCEL_ME")
    db.clear_old_commands()
    statuses = {row["command"]: row["status"] for row in db.conn.execute("SELECT command, status FROM commands")}
    assert statuses["CANCEL_ME"] == "cancelled"


def test_trade_plan_reason_lookup_and_counts(db_session):
    db, session_id = db_session
    plan_id = db.create_trade_plan(
        session_id,
        symbol="BTC/USD",
        side="long",
        entry_price=100.0,
        stop_price=None,
        target_price=None,
        size=1.0,
        reason="entry",
        entry_order_id="OID-1",
        entry_client_order_id="CID-1",
    )
    reason_by_order = db.get_trade_plan_reason_by_order(session_id, order_id="OID-1")
    reason_by_client = db.get_trade_plan_reason_by_order(session_id, client_order_id="CID-1")
    count = db.count_open_trade_plans_for_symbol(session_id, "BTC/USD")

    assert reason_by_order == "entry"
    assert reason_by_client == "entry"
    assert count == 1


def test_portfolio_days_upsert(tmp_path):
    db_path = tmp_path / "portfolio-days.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("swing", base_currency="usd", bot_version="v1")

    first = db.ensure_portfolio_day(portfolio["id"], datetime(2024, 1, 1).date(), timezone="AEST")
    second = db.ensure_portfolio_day(portfolio["id"], datetime(2024, 1, 1).date(), timezone="AEST")

    assert first["id"] == second["id"]
    assert first["portfolio_id"] == portfolio["id"]
    idx = {row["name"] for row in db.conn.execute("PRAGMA index_list(portfolio_days)")}
    assert "idx_portfolio_days_portfolio_date" in idx
    db.close()
