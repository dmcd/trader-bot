from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from trader_bot.database import TradingDatabase


@pytest.fixture
def db_session(tmp_path):
    db_path = tmp_path / "db.sqlite"
    db = TradingDatabase(str(db_path))
    portfolio_id, _ = db.ensure_active_portfolio(name="test-portfolio", bot_version="test-version", base_currency="USD")
    try:
        yield db, portfolio_id
    finally:
        db.close()


def _insert_market_data_rows(db, portfolio_id, old_ts, recent_ts):
    cursor = db.conn.cursor()
    cursor.execute(
        """
        INSERT INTO market_data (portfolio_id, timestamp, symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (portfolio_id, old_ts.isoformat(), "BTC/USD", 10.0, 9.5, 10.5, 1.0, None, None, None, None),
    )
    cursor.execute(
        """
        INSERT INTO market_data (portfolio_id, timestamp, symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (portfolio_id, recent_ts.isoformat(), "BTC/USD", 11.0, 10.5, 11.5, 2.0, None, None, None, None),
    )
    db.conn.commit()


def _insert_llm_traces(db, portfolio_id, old_ts, recent_ts):
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO llm_traces (portfolio_id, timestamp, prompt, response, decision_json, market_context) VALUES (?, ?, ?, ?, ?, ?)",
        (portfolio_id, old_ts.isoformat(), "old", "resp", "{}", "{}"),
    )
    cursor.execute(
        "INSERT INTO llm_traces (portfolio_id, timestamp, prompt, response, decision_json, market_context) VALUES (?, ?, ?, ?, ?, ?)",
        (portfolio_id, recent_ts.isoformat(), "new", "resp", "{}", "{}"),
    )
    db.conn.commit()


def test_log_and_fetch_entities(db_session):
    db, portfolio_id = db_session
    db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "BUY", 0.1, 20000.0, fee=1.0, reason="test")
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=10, output_tokens=5, cost=0.001, decision="{}")
    db.log_market_data_for_portfolio(portfolio_id, "BTC/USD", price=20000.0, bid=19990.0, ask=20010.0, volume=1000.0)
    db.log_equity_snapshot_for_portfolio(portfolio_id, equity=5050.0)

    trades = db.get_recent_trades_for_portfolio(portfolio_id, limit=5)
    assert len(trades) == 1

    stats = db.get_portfolio_stats(portfolio_id)
    assert stats["total_trades"] == 1
    assert stats["total_fees"] >= 1.0

    latest_equity = db.get_latest_equity_for_portfolio(portfolio_id)
    assert latest_equity == 5050.0

    positions = db.get_net_positions_from_trades_for_portfolio(portfolio_id)
    assert positions["BTC/USD"] == pytest.approx(0.1)


def test_portfolio_base_currency_round_trip(tmp_path):
    db_path = tmp_path / "base_ccy.db"
    db = TradingDatabase(str(db_path))
    portfolio_id, _ = db.ensure_active_portfolio(name="aud-portfolio", base_currency="aud", bot_version="aud")
    portfolio = db.get_portfolio(portfolio_id)
    assert portfolio["base_currency"] == "AUD"
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


def test_ensure_active_portfolio_returns_run_id(tmp_path):
    db_path = tmp_path / "active-portfolio.db"
    db = TradingDatabase(str(db_path))

    portfolio_id, run_id = db.ensure_active_portfolio(name="active", base_currency="usd", bot_version="v1")
    assert portfolio_id > 0
    assert run_id.startswith("v1-") or run_id.startswith("active-")

    reused_portfolio_id, provided_run = db.ensure_active_portfolio(name="active", base_currency="usd", bot_version="v1", run_id="custom-run")
    assert reused_portfolio_id == portfolio_id
    assert provided_run == "custom-run"
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
    assert "overnight_widened_at" in cols
    assert "overnight_widen_version" in cols
    assert "last_widened_stop_price" in cols
    assert "last_widened_target_price" in cols
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


def test_latest_run_metadata_for_portfolio(db_session):
    db, portfolio_id = db_session
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=1, output_tokens=1, cost=0.1, decision="ok", run_id="run-1")
    first = db.get_latest_run_metadata_for_portfolio(portfolio_id)
    assert first["run_id"] == "run-1"
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=1, output_tokens=1, cost=0.1, decision="ok", run_id="run-2")
    latest = db.get_latest_run_metadata_for_portfolio(portfolio_id)
    assert latest["run_id"] == "run-2"
    assert latest["source"] == "llm_calls"


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

    db.record_processed_trade_ids_for_portfolio(portfolio["id"], [("tid-1", "cid-1")])
    row = db.conn.execute("SELECT portfolio_id FROM processed_trades WHERE trade_id = 'tid-1'").fetchone()

    assert row["portfolio_id"] == portfolio["id"]
    db.close()


def test_trades_can_be_fetched_by_portfolio(tmp_path):
    db_path = tmp_path / "portfolio-trades.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("p1", base_currency="usd", bot_version="v1")
    db.log_trade_for_portfolio(portfolio["id"], "BTC/USD", "BUY", 0.1, 20000.0, fee=0.0)

    trades = db.get_recent_trades_for_portfolio(portfolio["id"], limit=5)

    assert trades and trades[0]["portfolio_id"] == portfolio["id"]
    db.close()



def test_portfolio_first_helpers_do_not_require_legacy_session_rows(tmp_path):
    db_path = tmp_path / "portfolio-first.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("pf", base_currency="usd", bot_version="v1")

    db.log_trade_for_portfolio(portfolio["id"], "BTC/USD", "BUY", 0.2, 20000.0, fee=1.5, reason="entry", realized_pnl=5.0, trade_id="tid-portfolio")
    db.log_market_data_for_portfolio(portfolio["id"], "BTC/USD", price=20000.0, bid=19990.0, ask=20010.0, volume=500)
    db.log_llm_call_for_portfolio(portfolio["id"], input_tokens=5, output_tokens=5, cost=0.05, decision="ok", run_id="run-p")
    trace_id = db.log_llm_trace_for_portfolio(portfolio["id"], prompt="p", response="r", decision_json="{}", market_context={"a": 1}, run_id="run-p")
    db.update_llm_trace_execution(trace_id, {"result": "done"})
    db.record_processed_trade_ids_for_portfolio(portfolio["id"], [("tid-portfolio", "cid-portfolio")])
    bars = [
        {"timestamp": 1_000_000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
        {"timestamp": 1_000_060, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 20},
    ]
    db.log_ohlcv_batch_for_portfolio(portfolio["id"], "BTC/USD", "1m", bars)
    db.log_equity_snapshot_for_portfolio(portfolio["id"], equity=1500.0, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc))
    db.replace_positions_for_portfolio(portfolio["id"], [{"symbol": "BTC/USD", "quantity": 0.2, "avg_price": 20000.0}])
    db.replace_open_orders_for_portfolio(portfolio["id"], [{"order_id": "oid-1", "symbol": "BTC/USD", "side": "buy", "price": 19900.0, "amount": 0.1, "remaining": 0.1, "status": "open"}])

    trades = db.get_recent_trades_for_portfolio(portfolio["id"], limit=2)
    market = db.get_recent_market_data_for_portfolio(portfolio["id"], "BTC/USD", limit=1)
    traces = db.get_recent_llm_traces_for_portfolio(portfolio["id"], limit=2)
    stats = db.get_portfolio_stats(portfolio["id"])
    processed = db.get_processed_trade_ids_for_portfolio(portfolio["id"])

    assert trades and trades[0]["portfolio_id"] == portfolio["id"]
    assert market and market[0]["portfolio_id"] == portfolio["id"]
    assert traces and traces[0]["id"] == trace_id
    assert stats["total_trades"] == 1
    assert stats["gross_pnl"] == pytest.approx(5.0)
    assert "tid-portfolio" in processed
    assert len(db.get_recent_ohlcv_for_portfolio(portfolio["id"], "BTC/USD", "1m", limit=5)) == 2
    assert db.get_latest_equity_for_portfolio(portfolio["id"]) == pytest.approx(1500.0)
    assert db.get_positions_for_portfolio(portfolio["id"])[0]["symbol"] == "BTC/USD"
    assert db.get_open_orders_for_portfolio(portfolio["id"])[0]["order_id"] == "oid-1"
    db.close()


def test_market_data_preserves_integer_sizes(db_session):
    db, portfolio_id = db_session
    db.log_market_data_for_portfolio(
        portfolio_id,
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

    rows = db.get_recent_market_data_for_portfolio(portfolio_id, "BHP/AUD", limit=1)
    assert rows[0]["volume"] == 200
    assert isinstance(rows[0]["volume"], int)
    assert rows[0]["bid_size"] == 150
    assert rows[0]["ask_size"] == 120


def test_log_and_fetch_ohlcv(db_session):
    db, portfolio_id = db_session
    bars = [
        {"timestamp": 1_000_000, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
        {"timestamp": 1_000_060, "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 20},
    ]
    db.log_ohlcv_batch_for_portfolio(portfolio_id, "BTC/USD", "1m", bars)
    fetched = db.get_recent_ohlcv_for_portfolio(portfolio_id, "BTC/USD", "1m", limit=5)

    assert len(fetched) == 2
    assert fetched[0]["close"] == pytest.approx(2.0)


def test_prune_ohlcv_retains_latest_rows(db_session):
    db, portfolio_id = db_session
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
    db.log_ohlcv_batch_for_portfolio(portfolio_id, "ETH/USD", "5m", bars)
    db.prune_ohlcv_for_portfolio(portfolio_id, "ETH/USD", "5m", retain=3)

    remaining = db.get_recent_ohlcv_for_portfolio(portfolio_id, "ETH/USD", "5m", limit=10)
    assert len(remaining) == 3
    assert remaining[0]["close"] == pytest.approx(5.5)
    assert remaining[-1]["close"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "setup_fn, prune_call, count_query",
    [
        (
            lambda db, portfolio_id: _insert_market_data_rows(
                db,
                portfolio_id,
                datetime.now() - timedelta(minutes=10),
                datetime.now(),
            ),
            lambda db, portfolio_id: db.prune_market_data_for_portfolio(portfolio_id, retention_minutes=5),
            "SELECT COUNT(*) as cnt FROM market_data WHERE portfolio_id = ?",
        ),
        (
            lambda db, portfolio_id: _insert_llm_traces(
                db,
                portfolio_id,
                datetime.now() - timedelta(days=10),
                datetime.now(),
            ),
            lambda db, portfolio_id: db.prune_llm_traces_for_portfolio(portfolio_id, retention_days=7),
            "SELECT COUNT(*) as cnt FROM llm_traces WHERE portfolio_id = ?",
        ),
    ],
)
def test_prune_tables_drop_old_rows(db_session, setup_fn, prune_call, count_query):
    db, portfolio_id = db_session
    setup_fn(db, portfolio_id)

    prune_call(db, portfolio_id)

    cursor = db.conn.cursor()
    cursor.execute(count_query, (portfolio_id,))
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
    db, portfolio_id = db_session
    plan_id = db.create_trade_plan_for_portfolio(
        portfolio_id,
        symbol="BTC/USD",
        side="long",
        entry_price=100.0,
        stop_price=90.0,
        target_price=120.0,
        size=1.0,
        reason="entry",
        entry_client_order_id="cid-1",
    )
    widened_at = "2024-01-02T00:00:00Z"
    db.update_trade_plan_prices(
        plan_id,
        stop_price=95.0,
        target_price=125.0,
        reason="tighten",
        widened_at=widened_at,
        widen_stop_price=95.0,
        widen_target_price=125.0,
        widen_version=2,
    )
    db.update_trade_plan_size(plan_id, size=0.5, reason="partial")

    open_plans = db.get_open_trade_plans_for_portfolio(portfolio_id)
    assert len(open_plans) == 1
    assert open_plans[0]["version"] == 3
    assert open_plans[0]["size"] == 0.5
    assert open_plans[0]["reason"] == "partial"
    assert open_plans[0]["overnight_widened_at"] == widened_at
    assert open_plans[0]["overnight_widen_version"] == 2
    assert open_plans[0]["last_widened_stop_price"] == 95.0
    assert open_plans[0]["last_widened_target_price"] == 125.0

    db.update_trade_plan_status(plan_id, "closed", closed_at="2024-01-01T00:00:00Z", reason="exit")
    assert db.get_open_trade_plans_for_portfolio(portfolio_id) == []
    reason = db.get_trade_plan_reason_by_order_for_portfolio(portfolio_id, client_order_id="cid-1")
    assert reason == "exit"


def test_equity_snapshot_logging_and_pruning(db_session):
    db, portfolio_id = db_session
    db.log_equity_snapshot_for_portfolio(portfolio_id, 100.0)
    db.log_equity_snapshot_for_portfolio(portfolio_id, 150.0)

    assert db.get_latest_equity_for_portfolio(portfolio_id) == 150.0
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM equity_snapshots WHERE equity < 150")
    db.conn.commit()
    assert db.get_latest_equity_for_portfolio(portfolio_id) == 150.0


def test_first_equity_snapshot_returns_earliest(db_session):
    db, portfolio_id = db_session
    db.log_equity_snapshot_for_portfolio(portfolio_id, 200.0, timestamp="2024-01-02T00:00:00Z")
    db.log_equity_snapshot_for_portfolio(portfolio_id, 150.0, timestamp="2024-01-01T00:00:00Z")

    first = db.get_first_equity_snapshot_for_portfolio(portfolio_id)

    assert first["equity"] == pytest.approx(150.0)
    assert first["timestamp"].startswith("2024-01-01")


def test_portfolio_day_updates_from_equity_snapshots(tmp_path):
    db_path = tmp_path / "portfolio-days.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio(name="swing", base_currency="AUD", bot_version="v1")

    tz_name = "Australia/Melbourne"
    first_ts = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)  # midnight local on Jan 2
    db.log_equity_snapshot_for_portfolio(
        portfolio["id"],
        equity=1000.0,
        timestamp=first_ts,
        timezone_name=tz_name,
    )
    second_ts = datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc)  # still Jan 2 locally
    db.log_equity_snapshot_for_portfolio(
        portfolio["id"],
        equity=1100.0,
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
    db.log_equity_snapshot_for_portfolio(
        portfolio["id"],
        equity=1050.0,
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


def test_portfolio_day_defaults_to_configured_timezone(tmp_path):
    db_path = tmp_path / "portfolio-day-tz.db"
    db = TradingDatabase(str(db_path), portfolio_day_timezone="Australia/Sydney")
    portfolio = db.get_or_create_portfolio(name="tz-portfolio", base_currency="AUD", bot_version="v1")

    first_ts = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)  # Jan 2 local (AEDT)
    db.log_equity_snapshot_for_portfolio(portfolio["id"], equity=2000.0, timestamp=first_ts)
    second_ts = datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)  # Jan 3 local (AEDT)
    db.log_equity_snapshot_for_portfolio(
        portfolio["id"],
        equity=2100.0,
        timestamp=second_ts,
        timezone_name="AEST",  # alias should resolve to Australia/Sydney
    )

    rows = db.conn.execute(
        """
        SELECT date, timezone, start_equity, end_equity
        FROM portfolio_days
        WHERE portfolio_id = ?
        ORDER BY date ASC
        """,
        (portfolio["id"],),
    ).fetchall()

    assert len(rows) == 2
    assert rows[0]["date"] == "2024-01-02"
    assert rows[0]["timezone"] == "Australia/Sydney"
    assert rows[0]["start_equity"] == pytest.approx(2000.0)
    assert rows[0]["end_equity"] == pytest.approx(2000.0)
    assert rows[1]["date"] == "2024-01-03"
    assert rows[1]["timezone"] == "Australia/Sydney"
    assert rows[1]["start_equity"] == pytest.approx(2100.0)
    assert rows[1]["end_equity"] == pytest.approx(2100.0)


def test_end_of_day_snapshot_roundtrip(tmp_path):
    db_path = tmp_path / "eod.db"
    db = TradingDatabase(str(db_path), portfolio_day_timezone="Australia/Sydney")
    portfolio = db.get_or_create_portfolio(name="swing", base_currency="AUD", bot_version="v1")
    positions = [{"symbol": "BTC/USD", "quantity": 1.0}]
    plans = [{"id": 1, "symbol": "BTC/USD", "status": "open"}]

    ts = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)  # Jan 2 local
    db.log_end_of_day_snapshot_for_portfolio(
        portfolio["id"],
        equity=1200.0,
        positions=positions,
        plans=plans,
        timestamp=ts,
        run_id="run-1",
    )
    # Second write should upsert the same row
    db.log_end_of_day_snapshot_for_portfolio(
        portfolio["id"],
        equity=1250.0,
        positions=positions,
        plans=plans,
        timestamp=ts,
        timezone_name="AEST",
    )

    latest = db.get_latest_end_of_day_snapshot_for_portfolio(portfolio["id"])
    assert latest["date"] == "2024-01-02"
    assert latest["timezone"] == "Australia/Sydney"
    assert latest["equity"] == pytest.approx(1250.0)
    assert latest["positions"][0]["symbol"] == "BTC/USD"
    assert latest["plans"][0]["status"] == "open"

    by_date = db.get_end_of_day_snapshot_for_date(
        portfolio["id"],
        day="2024-01-02",
        timezone_name="Australia/Sydney",
    )
    assert by_date["equity"] == pytest.approx(1250.0)
    db.close()


def test_get_open_orders_handles_empty_table(db_session):
    db, portfolio_id = db_session
    assert db.get_open_orders_for_portfolio(portfolio_id) == []


def test_replace_positions_writes_portfolio_rows(tmp_path):
    db_path = tmp_path / "positions-clean.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("swing", base_currency="USD", bot_version="v1")

    db.replace_positions_for_portfolio(portfolio["id"], [{"symbol": "BTC/USD", "quantity": 1.0, "avg_price": 10000.0}])

    positions = db.get_positions_for_portfolio(portfolio["id"])
    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTC/USD"
    db.close()


def test_replace_open_orders_writes_portfolio_rows(tmp_path):
    db_path = tmp_path / "orders-clean.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("swing", base_currency="USD", bot_version="v1")

    db.replace_open_orders_for_portfolio(
        portfolio["id"],
        [{"order_id": "fresh", "symbol": "BTC/USD", "side": "buy", "price": 10100, "amount": 1, "remaining": 1}],
    )

    orders = db.get_open_orders_for_portfolio(portfolio["id"])
    assert len(orders) == 1
    assert orders[0]["order_id"] == "fresh"
    db.close()


def test_log_llm_trace_handles_bad_json(db_session):
    db, portfolio_id = db_session
    cyclical = []
    cyclical.append(cyclical)

    trace_id = db.log_llm_trace_for_portfolio(portfolio_id, "prompt", "resp", decision_json="{}", market_context=cyclical)
    cursor = db.conn.cursor()
    cursor.execute("SELECT market_context FROM llm_traces WHERE id = ?", (trace_id,))
    stored = cursor.fetchone()["market_context"]

    assert stored == str(cyclical)


def test_portfolio_version_helpers(tmp_path):
    db_path = tmp_path / "portfolios.db"
    db = TradingDatabase(str(db_path))
    first_id, _ = db.ensure_active_portfolio(name="swing-v1", bot_version="v1")
    second_id, _ = db.ensure_active_portfolio(name="swing-v1-b", bot_version="v1")
    other_id, _ = db.ensure_active_portfolio(name="swing-v2", bot_version="v2")

    assert db.get_portfolio_id_by_version("v1") == second_id
    portfolios_v1 = db.list_portfolios(bot_version="v1")
    assert [p["id"] for p in portfolios_v1] == [second_id, first_id]
    all_portfolios = db.list_portfolios()
    assert {p["id"] for p in all_portfolios} == {first_id, second_id, other_id}
    db.close()


def test_log_estimated_fee_persists_row(db_session):
    db, portfolio_id = db_session
    db.log_estimated_fee_for_portfolio(portfolio_id, order_id="abc", estimated_fee=1.23, symbol="BTC/USD", action="BUY")

    cursor = db.conn.cursor()
    row = cursor.execute(
        "SELECT estimated_fee, action, order_id, portfolio_id FROM estimated_fees WHERE order_id = ?",
        ("abc",),
    ).fetchone()
    assert row["estimated_fee"] == pytest.approx(1.23)
    assert row["action"] == "BUY"
    assert row["order_id"] == "abc"
    assert row["portfolio_id"] == portfolio_id


def test_llm_trace_roundtrip_and_prune(db_session):
    db, portfolio_id = db_session
    trace_id = db.log_llm_trace_for_portfolio(portfolio_id, prompt="p", response="r", decision_json="{}", market_context={"a": 1}, run_id="run-1")
    db.update_llm_trace_execution(trace_id, {"result": "ok"})
    traces = db.get_recent_llm_traces_for_portfolio(portfolio_id, limit=5)
    assert traces and traces[0]["id"] == trace_id
    assert '"result": "ok"' in (traces[0]["execution_result"] or "")
    cursor = db.conn.cursor()
    row = cursor.execute("SELECT run_id FROM llm_traces WHERE id = ?", (trace_id,)).fetchone()
    assert row["run_id"] == "run-1"

    old_ts = (datetime.now() - timedelta(days=10)).isoformat()
    db.conn.execute(
        "INSERT INTO llm_traces (portfolio_id, timestamp, prompt, response, decision_json) VALUES (?, ?, ?, ?, ?)",
        (portfolio_id, old_ts, "old", "resp", "{}"),
    )
    db.conn.commit()

    db.prune_llm_traces_for_portfolio(portfolio_id, retention_days=7)
    remaining = db.get_recent_llm_traces_for_portfolio(portfolio_id, limit=10)
    assert all(t["timestamp"] >= old_ts for t in remaining)


def test_recent_llm_stats_counts_flags(db_session):
    db, portfolio_id = db_session
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=1, output_tokens=1, cost=0.0, decision="schema_error_missing")
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=1, output_tokens=1, cost=0.0, decision="clamped_size")
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=1, output_tokens=1, cost=0.0, decision="ok")

    stats = db.get_recent_llm_stats_for_portfolio(portfolio_id, limit=10)
    assert stats["total"] == 3
    assert stats["schema_errors"] == 1
    assert stats["clamped"] == 1


def test_portfolio_stats_aggregates_without_cache(db_session):
    db, portfolio_id = db_session
    db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "BUY", 0.1, 20000.0, fee=1.0, realized_pnl=2.5)
    db.log_trade_for_portfolio(portfolio_id, "BTC/USD", "SELL", 0.1, 21000.0, fee=1.5, realized_pnl=3.5)
    db.log_llm_call_for_portfolio(portfolio_id, input_tokens=1, output_tokens=1, cost=0.2, decision="ok")

    stats = db.get_portfolio_stats(portfolio_id)

    assert stats["total_trades"] == 2
    assert stats["gross_pnl"] == pytest.approx(6.0)
    assert stats["total_fees"] == pytest.approx(2.5)
    assert stats["total_llm_cost"] == pytest.approx(0.2)


def test_portfolio_stats_prefers_cache(tmp_path):
    db_path = tmp_path / "portfolio-stats.db"
    db = TradingDatabase(str(db_path))
    portfolio = db.get_or_create_portfolio("swing", base_currency="USD", bot_version="v1")
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

    stats = db.get_portfolio_stats(portfolio["id"])

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
    db, portfolio_id = db_session
    plan_id = db.create_trade_plan_for_portfolio(
        portfolio_id,
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
    reason_by_order = db.get_trade_plan_reason_by_order_for_portfolio(portfolio_id, order_id="OID-1")
    reason_by_client = db.get_trade_plan_reason_by_order_for_portfolio(portfolio_id, client_order_id="CID-1")
    count = db.count_open_trade_plans_for_symbol_for_portfolio(portfolio_id, "BTC/USD")

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
