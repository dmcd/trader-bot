import sqlite3
import logging
import os
import json
import uuid
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from trader_bot.config import PORTFOLIO_DAY_TIMEZONE

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Manages SQLite database for persistent trading data."""
    
    def __init__(self, db_path: Optional[str] = None, portfolio_day_timezone: Optional[str] = None):
        # Allow tests or env overrides to point at an isolated database
        self.db_path = db_path or os.getenv("TRADING_DB_PATH", "trading.db")
        self.portfolio_day_timezone = portfolio_day_timezone or PORTFOLIO_DAY_TIMEZONE or "UTC"
        self.conn = None
        self.initialize_database()

    @staticmethod
    def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
        cursor.execute(f"PRAGMA table_info({table})")
        return any(row["name"] == column for row in cursor.fetchall())

    def initialize_database(self):
        """Create database and tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                base_currency TEXT,
                bot_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_days (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                timezone TEXT NOT NULL,
                start_equity REAL,
                end_equity REAL,
                gross_pnl REAL,
                net_pnl REAL,
                fees REAL,
                llm_cost REAL,
                max_drawdown REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(portfolio_id, date, timezone),
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_days_portfolio_date ON portfolio_days (portfolio_id, date DESC)"
            )
        except Exception as exc:
            logger.debug(f"Could not create portfolio_days index: {exc}")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS end_of_day_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                timezone TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                equity REAL,
                positions_json TEXT,
                plans_json TEXT,
                run_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(portfolio_id, date, timezone),
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_eod_snapshots_portfolio_date ON end_of_day_snapshots (portfolio_id, date DESC)"
            )
        except Exception as exc:
            logger.debug(f"Could not create end_of_day_snapshots index: {exc}")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_stats_cache (
                portfolio_id INTEGER PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                total_fees REAL DEFAULT 0.0,
                gross_pnl REAL DEFAULT 0.0,
                total_llm_cost REAL DEFAULT 0.0,
                exposure_notional REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                fee REAL DEFAULT 0.0,
                liquidity TEXT DEFAULT 'unknown',
                realized_pnl REAL DEFAULT 0.0,
                reason TEXT,
                trade_id TEXT,
                UNIQUE(trade_id),
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                client_order_id TEXT,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(portfolio_id, trade_id),
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        
        # LLM calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                run_id TEXT,
                timestamp TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                decision TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)

        # LLM trace table for full prompt/response/decision and optional execution result
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                run_id TEXT,
                timestamp TEXT NOT NULL,
                prompt TEXT,
                response TEXT,
                decision_json TEXT,
                market_context TEXT,
                execution_result TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                bid REAL,
                ask REAL,
                volume REAL,
                spread_pct REAL,
                bid_size REAL,
                ask_size REAL,
                ob_imbalance REAL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)

        # OHLCV bars table (multi-timeframe)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        # Equity snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                equity REAL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        
        # Technical indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bb_upper REAL,
                bb_lower REAL,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)

        # Positions snapshot table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL,
                exchange_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)

        # Open orders snapshot table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS open_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT,
                price REAL,
                amount REAL,
                remaining REAL,
                status TEXT,
                exchange_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)

        # Portfolio-aware indexes for common lookups
        index_specs = [
            ("trades", "idx_trades_portfolio_symbol_ts", "portfolio_id, symbol, timestamp DESC"),
            ("processed_trades", "idx_processed_trades_portfolio_trade", "portfolio_id, trade_id"),
            ("market_data", "idx_market_data_portfolio_symbol_ts", "portfolio_id, symbol, timestamp DESC"),
            ("ohlcv_bars", "idx_ohlcv_portfolio_symbol_tf_ts", "portfolio_id, symbol, timeframe, timestamp DESC"),
            ("equity_snapshots", "idx_equity_portfolio_ts", "portfolio_id, timestamp DESC"),
            ("indicators", "idx_indicators_portfolio_symbol_ts", "portfolio_id, symbol, timestamp DESC"),
            ("positions", "idx_positions_portfolio_symbol", "portfolio_id, symbol"),
            ("open_orders", "idx_open_orders_portfolio_symbol", "portfolio_id, symbol"),
            ("llm_calls", "idx_llm_calls_portfolio_ts", "portfolio_id, timestamp DESC"),
            ("llm_traces", "idx_llm_traces_portfolio_ts", "portfolio_id, timestamp DESC"),
        ]
        for table_name, index_name, columns in index_specs:
            if not self._column_exists(cursor, table_name, "portfolio_id"):
                continue
            try:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})")
            except Exception as e:
                logger.debug(f"Could not create index {index_name}: {e}")

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run_ts ON llm_calls (run_id, timestamp DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_traces_run_ts ON llm_traces (run_id, timestamp DESC)")
        except Exception as exc:
            logger.debug(f"Could not create run_id telemetry indexes: {exc}")
        
        # Commands table for dashboard-to-bot communication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                executed_at TEXT
            )
        """)

        # Health state table (key/value) for surfacing circuit breakers and ops signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                detail TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_portfolio(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_portfolio_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM portfolios WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def ensure_portfolio_day(self, portfolio_id: int, day: date, timezone: str) -> Dict[str, Any]:
        """
        Create a portfolio_days row if missing for the given date/timezone and return the row.
        Called when capturing the first equity snapshot of a day.
        """
        cursor = self.conn.cursor()
        day_str = day.isoformat()
        cursor.execute(
            """
            INSERT INTO portfolio_days (portfolio_id, date, timezone)
            VALUES (?, ?, ?)
            ON CONFLICT(portfolio_id, date, timezone) DO NOTHING
            """,
            (portfolio_id, day_str, timezone),
        )
        self.conn.commit()
        cursor.execute(
            "SELECT * FROM portfolio_days WHERE portfolio_id = ? AND date = ? AND timezone = ?",
            (portfolio_id, day_str, timezone),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}

    @staticmethod
    def _normalize_timestamp(ts: datetime | str | None) -> datetime:
        """Normalize timestamps for snapshot logging."""
        if ts is None:
            return datetime.now(timezone.utc)
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                return datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _resolve_timezone(self, timezone_name: str | None) -> tuple[Any, str]:
        """Resolve timezone for portfolio day reporting with fallbacks and alias support."""
        aliases = {
            "AEST": "Australia/Sydney",
            "AEDT": "Australia/Sydney",
            "UTC": "UTC",
        }
        preferred = [
            timezone_name,
            self.portfolio_day_timezone,
            "UTC",
        ]
        for candidate in preferred:
            if not candidate:
                continue
            normalized = aliases.get(candidate.upper(), candidate)
            try:
                tzinfo = ZoneInfo(normalized)
                label = getattr(tzinfo, "key", normalized)
                return tzinfo, label
            except Exception:
                logger.debug(f"Unknown timezone '{candidate}' when deriving portfolio_days; trying fallback.")
                continue
        return timezone.utc, "UTC"

    @staticmethod
    def _safe_json_loads(payload: str | None) -> list[Any] | dict[str, Any] | None:
        """Parse JSON payloads defensively."""
        if not payload:
            return [] if payload == "" else None
        try:
            return json.loads(payload)
        except Exception:
            return None

    def update_portfolio_day_from_snapshot(
        self,
        portfolio_id: Optional[int],
        as_of: datetime | str | None,
        equity: float,
        timezone_name: str | None = None,
    ):
        """
        Update portfolio_days start/end equity and derived PnL based solely on mark-to-market snapshots.

        This is reporting-only and does not gate risk checks.
        """
        if portfolio_id is None or equity is None:
            return

        ts = self._normalize_timestamp(as_of)
        tzinfo, tz_label = self._resolve_timezone(timezone_name)
        day = ts.astimezone(tzinfo).date()

        try:
            equity_val = float(equity)
        except (TypeError, ValueError):
            return

        row = self.ensure_portfolio_day(portfolio_id, day, tz_label)
        row_id = row.get("id")
        if row_id is None:
            return
        start_equity = row.get("start_equity")
        if start_equity is None:
            start_equity = equity_val
        end_equity = equity_val
        gross_pnl = None
        net_pnl = None
        if start_equity is not None:
            gross_pnl = end_equity - start_equity
            fees = row.get("fees") or 0.0
            llm_cost = row.get("llm_cost") or 0.0
            net_pnl = gross_pnl - fees - llm_cost

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE portfolio_days
            SET start_equity = COALESCE(start_equity, ?),
                end_equity = ?,
                gross_pnl = ?,
                net_pnl = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (start_equity, end_equity, gross_pnl, net_pnl, row_id),
        )
        self.conn.commit()

    def get_or_create_portfolio(self, name: str, base_currency: Optional[str] = None, bot_version: Optional[str] = None) -> Dict[str, Any]:
        """Return an existing portfolio by name or create one with the provided metadata."""
        normalized_base = base_currency.upper() if base_currency else None
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM portfolios WHERE name = ?", (name,))
        row = cursor.fetchone()
        if row:
            needs_update = False
            if normalized_base and row["base_currency"] != normalized_base:
                needs_update = True
            if bot_version and row["bot_version"] != bot_version:
                needs_update = True
            if needs_update:
                cursor.execute(
                    """
                    UPDATE portfolios
                    SET base_currency = COALESCE(?, base_currency),
                        bot_version = COALESCE(?, bot_version)
                    WHERE id = ?
                    """,
                    (normalized_base, bot_version, row["id"]),
                )
                self.conn.commit()
                cursor.execute("SELECT * FROM portfolios WHERE id = ?", (row["id"],))
                row = cursor.fetchone()
            return dict(row)

        cursor.execute(
            """
            INSERT INTO portfolios (name, base_currency, bot_version)
            VALUES (?, ?, ?)
            """,
            (name, normalized_base, bot_version),
        )
        self.conn.commit()
        portfolio_id = cursor.lastrowid
        return self.get_portfolio(portfolio_id)

    def ensure_active_portfolio(
        self,
        name: Optional[str] = None,
        base_currency: Optional[str] = None,
        bot_version: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> tuple[int, str]:
        """
        Ensure a portfolio exists and return (portfolio_id, run_id).
        A new run_id is generated when one is not provided for telemetry/ops scoping.
        """
        portfolio_name = name or os.getenv("PORTFOLIO_NAME") or (bot_version or "portfolio").lower()
        portfolio = self.get_or_create_portfolio(
            portfolio_name,
            base_currency=base_currency,
            bot_version=bot_version,
        )
        resolved_run_id = run_id or f"{(bot_version or portfolio_name)}-{uuid.uuid4().hex[:12]}"
        return portfolio["id"], resolved_run_id

    def get_portfolio_id_by_version(self, bot_version: str) -> Optional[int]:
        """Return the most recent portfolio id for a bot version."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id FROM portfolios
            WHERE bot_version = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (bot_version,),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def list_portfolios(self, bot_version: str | None = None) -> List[Dict[str, Any]]:
        """List portfolios with optional bot_version filter."""
        cursor = self.conn.cursor()
        query = "SELECT id, name, bot_version, base_currency, created_at FROM portfolios"
        params: list[Any] = []
        if bot_version:
            query += " WHERE bot_version = ?"
            params.append(bot_version)
        query += " ORDER BY created_at DESC, id DESC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def list_bot_versions(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT bot_version
            FROM portfolios
            WHERE bot_version IS NOT NULL
            ORDER BY created_at DESC, id DESC
            """
        )
        versions = [row["bot_version"] for row in cursor.fetchall()]
        return versions

    def get_portfolio_stats_cache(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Return persisted portfolio stats aggregates if present."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT portfolio_id, total_trades, total_fees, gross_pnl, total_llm_cost, exposure_notional, updated_at
            FROM portfolio_stats_cache
            WHERE portfolio_id = ?
            """,
            (portfolio_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_portfolio_stats_cache(self, portfolio_id: int, stats: Dict[str, Any]):
        """Upsert portfolio stats aggregates for restart resilience."""
        if portfolio_id is None:
            return
        existing = self.get_portfolio_stats_cache(portfolio_id) or {}
        merged = {**existing, **(stats or {})}
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO portfolio_stats_cache (portfolio_id, total_trades, total_fees, gross_pnl, total_llm_cost, exposure_notional, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(portfolio_id) DO UPDATE SET
                total_trades=excluded.total_trades,
                total_fees=excluded.total_fees,
                gross_pnl=excluded.gross_pnl,
                total_llm_cost=excluded.total_llm_cost,
                exposure_notional=excluded.exposure_notional,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                portfolio_id,
                merged.get("total_trades", 0) or 0,
                merged.get("total_fees", 0.0) or 0.0,
                merged.get("gross_pnl", 0.0) or 0.0,
                merged.get("total_llm_cost", 0.0) or 0.0,
                merged.get("exposure_notional"),
            ),
        )
        self.conn.commit()

    def _insert_processed_trade_ids(self, entries: List[tuple[str, Optional[str]]], portfolio_id: int):
        """Internal helper to persist processed trade ids."""
        if not entries:
            return
        payload = [(portfolio_id, trade_id, client_oid) for trade_id, client_oid in entries if trade_id]
        if not payload:
            return
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT OR IGNORE INTO processed_trades (portfolio_id, trade_id, client_order_id)
            VALUES (?, ?, ?)
            """,
            payload,
        )
        self.conn.commit()

    def get_processed_trade_ids_for_portfolio(self, portfolio_id: int) -> set[str]:
        """Return trade_ids already seen for this portfolio."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT trade_id FROM processed_trades WHERE portfolio_id = ?", (portfolio_id,))
        return {row["trade_id"] for row in cursor.fetchall() if row["trade_id"]}

    def record_processed_trade_ids_for_portfolio(self, portfolio_id: int, entries: List[tuple[str, Optional[str]]]):
        """Persist processed trade ids for a portfolio."""
        self._insert_processed_trade_ids(entries, portfolio_id)
    
    def _log_trade(
        self,
        portfolio_id: int,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        fee: float,
        reason: str = "",
        liquidity: str = "unknown",
        realized_pnl: float = 0.0,
        trade_id: str | None = None,
        timestamp: str | None = None,
    ):
        """Internal helper to log a trade without extra run/session context."""
        cursor = self.conn.cursor()

        if trade_id:
            cursor.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
            if cursor.fetchone():
                logger.debug(f"Skipping duplicate trade {trade_id}")
                return

        ts = timestamp if timestamp else datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO trades (portfolio_id, timestamp, symbol, action, quantity, price, fee, liquidity, realized_pnl, reason, trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (portfolio_id, ts, symbol, action, quantity, price, fee, liquidity, realized_pnl, reason, trade_id),
        )

        self.conn.commit()
        logger.debug(f"Logged trade: {action} {quantity} {symbol} @ ${price}")

    def log_trade_for_portfolio(
        self,
        portfolio_id: int,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        fee: float,
        reason: str = "",
        liquidity: str = "unknown",
        realized_pnl: float = 0.0,
        trade_id: str | None = None,
        timestamp: str | None = None,
    ):
        """Log a trade directly against a portfolio."""
        self._log_trade(
            portfolio_id=portfolio_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            fee=fee,
            reason=reason,
            liquidity=liquidity,
            realized_pnl=realized_pnl,
            trade_id=trade_id,
            timestamp=timestamp,
        )

    def log_estimated_fee_for_portfolio(self, portfolio_id: int, order_id: str, estimated_fee: float, symbol: str, action: str):
        """Optional audit trail for estimated fees to compare with actuals."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS estimated_fees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER NOT NULL,
                    order_id TEXT,
                    symbol TEXT,
                    action TEXT,
                    estimated_fee REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
                )
            """)
            cursor.execute("""
                INSERT INTO estimated_fees (portfolio_id, order_id, symbol, action, estimated_fee)
                VALUES (?, ?, ?, ?, ?)
            """, (portfolio_id, str(order_id), symbol, action, estimated_fee))
            self.conn.commit()
        except Exception as e:
            logger.debug(f"Failed to log estimated fee: {e}")
    
    def _log_llm_call(
        self,
        portfolio_id: int,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        decision: str = "",
        run_id: str | None = None,
    ):
        """Internal helper to persist LLM call metadata."""
        cursor = self.conn.cursor()
        total_tokens = input_tokens + output_tokens
        cursor.execute(
            """
            INSERT INTO llm_calls (portfolio_id, run_id, timestamp, input_tokens, output_tokens, total_tokens, cost, decision)
            VALUES (
                ?,
                ?,
                ?, ?, ?, ?, ?, ?
            )
        """,
            (
                portfolio_id,
                run_id,
                datetime.now().isoformat(),
                input_tokens,
                output_tokens,
                total_tokens,
                cost,
                decision,
            ),
        )

        self.conn.commit()
        logger.debug(f"Logged LLM call: {total_tokens} tokens, ${cost:.6f}")

    def log_llm_call_for_portfolio(
        self,
        portfolio_id: int,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        decision: str = "",
        run_id: str | None = None,
    ):
        """Log an LLM API call for a portfolio."""
        self._log_llm_call(
            portfolio_id=portfolio_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            decision=decision,
            run_id=run_id,
        )

    def _log_llm_trace(
        self,
        portfolio_id: int,
        prompt: str,
        response: str,
        decision_json: str = "",
        market_context: Any = None,
        run_id: str | None = None,
    ) -> int:
        """Internal helper to persist full LLM trace rows."""
        cursor = self.conn.cursor()
        try:
            market_context_str = json.dumps(market_context, default=str) if market_context is not None else None
        except Exception:
            market_context_str = str(market_context)
        cursor.execute(
            """
            INSERT INTO llm_traces (portfolio_id, run_id, timestamp, prompt, response, decision_json, market_context)
            VALUES (
                ?,
                ?,
                ?, ?, ?, ?, ?
            )
        """,
            (portfolio_id, run_id, datetime.now().isoformat(), prompt, response, decision_json, market_context_str),
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_llm_trace_for_portfolio(
        self,
        portfolio_id: int,
        prompt: str,
        response: str,
        decision_json: str = "",
        market_context: Any = None,
        run_id: str | None = None,
    ) -> int:
        """Persist full LLM prompt/response plus parsed decision and context."""
        return self._log_llm_trace(
            portfolio_id=portfolio_id,
            prompt=prompt,
            response=response,
            decision_json=decision_json,
            market_context=market_context,
            run_id=run_id,
        )

    def update_llm_trace_execution(self, trace_id: int, execution_result: Any):
        """Attach execution outcome to an existing LLM trace row."""
        if not trace_id:
            return
        cursor = self.conn.cursor()
        try:
            execution_str = json.dumps(execution_result, default=str) if execution_result is not None else None
        except Exception:
            execution_str = str(execution_result)
        cursor.execute("""
            UPDATE llm_traces
            SET execution_result = ?
            WHERE id = ?
        """, (execution_str, trace_id))
        self.conn.commit()

    def prune_llm_traces_for_portfolio(self, portfolio_id: int, retention_days: int):
        """Delete LLM traces older than the retention window for a portfolio."""
        if retention_days is None or retention_days <= 0:
            return
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM llm_traces WHERE portfolio_id = ? AND timestamp < ?",
            (portfolio_id, cutoff),
        )
        self.conn.commit()

    def get_recent_llm_stats_for_portfolio(self, portfolio_id: int, limit: int = 20) -> Dict[str, Any]:
        """Return summary of recent LLM calls for telemetry scoped to portfolio_id."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT decision FROM llm_calls
            WHERE portfolio_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (portfolio_id, limit))
        rows = [row['decision'] or '' for row in cursor.fetchall()]
        stats = {"total": len(rows), "schema_errors": 0, "clamped": 0}
        for d in rows:
            if d.startswith("schema_error"):
                stats["schema_errors"] += 1
            if "clamped" in d.lower():
                stats["clamped"] += 1
        return stats

    def get_latest_run_metadata_for_portfolio(self, portfolio_id: int) -> Dict[str, Any]:
        """Return latest run_id and timestamp for the portfolio (from LLM telemetry)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT run_id, timestamp
            FROM llm_calls
            WHERE portfolio_id = ? AND run_id IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (portfolio_id,),
        )
        row = cursor.fetchone()
        if row and row["run_id"]:
            return {"run_id": row["run_id"], "last_seen": row["timestamp"], "source": "llm_calls"}

        cursor.execute(
            """
            SELECT run_id, timestamp
            FROM llm_traces
            WHERE portfolio_id = ? AND run_id IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (portfolio_id,),
        )
        row = cursor.fetchone()
        if row and row["run_id"]:
            return {"run_id": row["run_id"], "last_seen": row["timestamp"], "source": "llm_traces"}
        return {}

    def get_recent_llm_traces_for_portfolio(self, portfolio_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Return recent LLM traces (decision + execution) scoped to a portfolio."""
        cursor = self.conn.cursor()
        query = """
            SELECT id, timestamp, decision_json, execution_result
            FROM llm_traces
            WHERE portfolio_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        cursor.execute(query, (portfolio_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def _preserve_integer_if_whole(value: Any) -> Any:
        """Return ints for whole numbers (to preserve share counts) and pass through other values."""
        if value is None or isinstance(value, bool):
            return value
        try:
            if float(value).is_integer():
                return int(value)
        except Exception:
            return value
        return value

    def _log_market_data(
        self,
        portfolio_id: int,
        symbol: str,
        price: float | int | None,
        bid: float | int | None,
        ask: float | int | None,
        volume: float | int | None = None,
        spread_pct: float | int | None = None,
        bid_size: float | int | None = None,
        ask_size: float | int | None = None,
        ob_imbalance: float | int | None = None,
    ):
        """Internal helper to log market data snapshots."""
        volume = self._preserve_integer_if_whole(volume)
        bid_size = self._preserve_integer_if_whole(bid_size)
        ask_size = self._preserve_integer_if_whole(ask_size)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO market_data (portfolio_id, timestamp, symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (portfolio_id, datetime.now().isoformat(), symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance),
        )
        self.conn.commit()

    def log_market_data_for_portfolio(
        self,
        portfolio_id: int,
        symbol: str,
        price: float | int | None,
        bid: float | int | None,
        ask: float | int | None,
        volume: float | int | None = None,
        spread_pct: float | int | None = None,
        bid_size: float | int | None = None,
        ask_size: float | int | None = None,
        ob_imbalance: float | int | None = None,
    ):
        """Log market data snapshot for a portfolio."""
        self._log_market_data(
            portfolio_id=portfolio_id,
            symbol=symbol,
            price=price,
            bid=bid,
            ask=ask,
            volume=volume,
            spread_pct=spread_pct,
            bid_size=bid_size,
            ask_size=ask_size,
            ob_imbalance=ob_imbalance,
        )

    def prune_market_data_for_portfolio(self, portfolio_id: int, retention_minutes: int):
        """Trim market data older than the retention window for a portfolio."""
        if retention_minutes is None or retention_minutes <= 0:
            return
        cutoff = (datetime.now() - timedelta(minutes=retention_minutes)).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM market_data WHERE portfolio_id = ? AND timestamp < ?",
            (portfolio_id, cutoff),
        )
        self.conn.commit()

    def get_recent_trades_for_portfolio(self, portfolio_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for context by portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
                SELECT * FROM trades 
                WHERE portfolio_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
            (portfolio_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trades_for_portfolio(self, portfolio_id: int) -> List[Dict[str, Any]]:
        """Get all trades for a portfolio ordered chronologically."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
                SELECT * FROM trades 
                WHERE portfolio_id = ?
                ORDER BY timestamp ASC
            """,
            (portfolio_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _get_recent_market_data(
        self,
        portfolio_id: int,
        symbol: str,
        limit: int = 100,
        before_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Internal helper to fetch market data by portfolio."""
        cursor = self.conn.cursor()
        
        if before_timestamp:
            query = """
                SELECT * FROM market_data 
                WHERE portfolio_id = ? AND symbol = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (portfolio_id, symbol, before_timestamp, limit))
        else:
            query = """
                SELECT * FROM market_data 
                WHERE portfolio_id = ? AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            cursor.execute(query, (portfolio_id, symbol, limit))
        
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["volume"] = self._preserve_integer_if_whole(row.get("volume"))
            row["bid_size"] = self._preserve_integer_if_whole(row.get("bid_size"))
            row["ask_size"] = self._preserve_integer_if_whole(row.get("ask_size"))
        return rows

    def get_recent_market_data_for_portfolio(
        self,
        portfolio_id: int,
        symbol: str,
        limit: int = 100,
        before_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent market data for technical analysis scoped to a portfolio."""
        return self._get_recent_market_data(portfolio_id, symbol, limit=limit, before_timestamp=before_timestamp)

    def _log_equity_snapshot(
        self,
        portfolio_id: int,
        equity: float,
        timestamp: datetime | str | None = None,
        timezone_name: str | None = None,
    ):
        """Internal helper to persist equity snapshots."""
        cursor = self.conn.cursor()
        ts = self._normalize_timestamp(timestamp)
        ts_str = ts.isoformat()
        cursor.execute(
            """
            INSERT INTO equity_snapshots (portfolio_id, timestamp, equity)
            VALUES (?, ?, ?)
            """,
            (portfolio_id, ts_str, equity),
        )
        self.conn.commit()
        try:
            self.update_portfolio_day_from_snapshot(
                portfolio_id,
                ts,
                equity,
                timezone_name=timezone_name,
            )
        except Exception as exc:
            logger.debug(f"Could not update portfolio_days from equity snapshot: {exc}")

    def log_equity_snapshot_for_portfolio(
        self,
        portfolio_id: int,
        equity: float,
        timestamp: datetime | str | None = None,
        timezone_name: str | None = None,
    ):
        """Log equity snapshot for a portfolio."""
        self._log_equity_snapshot(
            portfolio_id=portfolio_id,
            equity=equity,
            timestamp=timestamp,
            timezone_name=timezone_name,
        )

    def log_end_of_day_snapshot_for_portfolio(
        self,
        portfolio_id: int,
        equity: float,
        positions: list[dict[str, Any]] | None,
        plans: list[dict[str, Any]] | None,
        timestamp: datetime | str | None = None,
        timezone_name: str | None = None,
        run_id: str | None = None,
    ):
        """Persist end-of-day snapshot for portfolio reporting without flattening positions/plans."""
        if portfolio_id is None:
            return
        ts = self._normalize_timestamp(timestamp)
        tzinfo, tz_label = self._resolve_timezone(timezone_name)
        day = ts.astimezone(tzinfo).date().isoformat()
        captured_at = ts.isoformat()
        try:
            positions_json = json.dumps(positions or [], default=str)
        except Exception:
            positions_json = "[]"
        try:
            plans_json = json.dumps(plans or [], default=str)
        except Exception:
            plans_json = "[]"

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO end_of_day_snapshots (portfolio_id, date, timezone, captured_at, equity, positions_json, plans_json, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(portfolio_id, date, timezone) DO UPDATE SET
                captured_at=excluded.captured_at,
                equity=excluded.equity,
                positions_json=excluded.positions_json,
                plans_json=excluded.plans_json,
                run_id=COALESCE(excluded.run_id, end_of_day_snapshots.run_id),
                updated_at=CURRENT_TIMESTAMP
            """,
            (portfolio_id, day, tz_label, captured_at, equity, positions_json, plans_json, run_id),
        )
        self.conn.commit()

    def get_latest_end_of_day_snapshot_for_portfolio(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Return the most recent end-of-day snapshot for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT portfolio_id, date, timezone, captured_at, equity, positions_json, plans_json, run_id
            FROM end_of_day_snapshots
            WHERE portfolio_id = ?
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            (portfolio_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        result["positions"] = self._safe_json_loads(result.pop("positions_json", None))
        result["plans"] = self._safe_json_loads(result.pop("plans_json", None))
        return result

    def get_end_of_day_snapshot_for_date(
        self,
        portfolio_id: int,
        day: date | str,
        timezone_name: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Return end-of-day snapshot for a specific local day."""
        if isinstance(day, date):
            day_str = day.isoformat()
        else:
            day_str = str(day)
        _, tz_label = self._resolve_timezone(timezone_name)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT portfolio_id, date, timezone, captured_at, equity, positions_json, plans_json, run_id
            FROM end_of_day_snapshots
            WHERE portfolio_id = ? AND date = ? AND timezone = ?
            LIMIT 1
            """,
            (portfolio_id, day_str, tz_label),
        )
        row = cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        result["positions"] = self._safe_json_loads(result.pop("positions_json", None))
        result["plans"] = self._safe_json_loads(result.pop("plans_json", None))
        return result

    def log_ohlcv_batch_for_portfolio(self, portfolio_id: int, symbol: str, timeframe: str, bars: List[Dict[str, Any]]):
        """Persist a batch of OHLCV bars for a symbol/timeframe."""
        if not bars:
            return
        cursor = self.conn.cursor()
        records = []
        for bar in bars:
            try:
                ts = bar.get('timestamp')
                if ts is None:
                    continue
                # Normalize timestamp to ISO if milliseconds provided
                if isinstance(ts, (int, float)):
                    ts_iso = datetime.fromtimestamp(ts / 1000).isoformat()
                else:
                    ts_iso = str(ts)
                records.append((
                    portfolio_id,
                    ts_iso,
                    symbol,
                    timeframe,
                    bar.get('open'),
                    bar.get('high'),
                    bar.get('low'),
                    bar.get('close') if bar.get('close') is not None else bar.get('price'),
                    bar.get('volume'),
                ))
            except Exception:
                continue
        if not records:
            return
        cursor.executemany("""
            INSERT INTO ohlcv_bars (portfolio_id, timestamp, symbol, timeframe, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        self.conn.commit()

    def prune_ohlcv_for_portfolio(self, portfolio_id: int, symbol: str, timeframe: str, retain: int):
        """Keep only the most recent `retain` rows for a symbol/timeframe."""
        if retain is None or retain <= 0:
            return
        cursor = self.conn.cursor()
        query = """
            DELETE FROM ohlcv_bars
            WHERE portfolio_id = ?
              AND symbol = ?
              AND timeframe = ?
              AND id NOT IN (
                SELECT id FROM ohlcv_bars
                WHERE portfolio_id = ?
                  AND symbol = ?
                  AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
              )
            """
        cursor.execute(
            query,
            (
                portfolio_id,
                symbol,
                timeframe,
                portfolio_id,
                symbol,
                timeframe,
                retain,
            ),
        )
        self.conn.commit()

    def get_recent_ohlcv_for_portfolio(self, portfolio_id: int, symbol: str, timeframe: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch recent OHLCV bars for a symbol/timeframe scoped to a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_bars
            WHERE portfolio_id = ? AND symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (portfolio_id, symbol, timeframe, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_equity_for_portfolio(self, portfolio_id: int) -> Optional[float]:
        """Get most recent equity snapshot for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT equity FROM equity_snapshots
            WHERE portfolio_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """,
            (portfolio_id,),
        )
        row = cursor.fetchone()
        return row["equity"] if row else None

    def get_first_equity_snapshot_for_portfolio(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        """Return the earliest equity snapshot for a portfolio (baseline reference)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, equity
            FROM equity_snapshots
            WHERE portfolio_id = ?
            ORDER BY timestamp ASC
            LIMIT 1
            """,
            (portfolio_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_portfolio_stats(self, portfolio_id: int) -> Dict[str, Any]:
        """Return portfolio-level statistics, preferring the stats cache."""
        stats: Dict[str, Any] = {}
        try:
            stats = self.get_portfolio_stats_cache(portfolio_id) or {}
        except Exception as exc:
            logger.debug(f"Could not load portfolio stats cache for portfolio {portfolio_id}: {exc}")

        cursor = self.conn.cursor()
        if stats.get("gross_pnl") is None or stats.get("total_trades") is None or stats.get("total_fees") is None:
            cursor.execute(
                """
                SELECT COUNT(*) as cnt,
                       COALESCE(SUM(fee), 0) as total_fees,
                       COALESCE(SUM(realized_pnl), 0) as realized
                FROM trades
                WHERE portfolio_id = ?
                """,
                (portfolio_id,),
            )
            trade_row = dict(cursor.fetchone() or {})
            stats.setdefault("total_trades", trade_row.get("cnt") or 0)
            stats.setdefault("total_fees", trade_row.get("total_fees") or 0.0)
            stats.setdefault("gross_pnl", trade_row.get("realized") or 0.0)

        if stats.get("total_llm_cost") is None:
            cursor.execute(
                "SELECT COALESCE(SUM(cost), 0) as total_cost FROM llm_calls WHERE portfolio_id = ?",
                (portfolio_id,),
            )
            cost_row = dict(cursor.fetchone() or {})
            stats["total_llm_cost"] = cost_row.get("total_cost") or 0.0

        stats.setdefault("total_trades", 0)
        stats.setdefault("total_fees", 0.0)
        stats.setdefault("gross_pnl", 0.0)
        stats.setdefault("total_llm_cost", 0.0)
        stats.setdefault("exposure_notional", 0.0)
        stats["portfolio_id"] = portfolio_id
        if stats.get("net_pnl") is None:
            stats["net_pnl"] = (
                (stats.get("gross_pnl", 0.0) or 0.0)
                - (stats.get("total_fees", 0.0) or 0.0)
                - (stats.get("total_llm_cost", 0.0) or 0.0)
            )
        return stats


    def replace_positions_for_portfolio(self, portfolio_id: int, positions: List[Dict[str, Any]]):
        """Replace stored positions snapshot for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM positions WHERE portfolio_id = ?",
            (portfolio_id,),
        )
        for pos in positions:
            cursor.execute("""
                INSERT INTO positions (portfolio_id, symbol, quantity, avg_price, exchange_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                portfolio_id,
                pos.get('symbol'),
                pos.get('quantity', 0),
                pos.get('avg_price'),
                pos.get('timestamp')
            ))
        self.conn.commit()

    def get_positions_for_portfolio(self, portfolio_id: int) -> List[Dict[str, Any]]:
        """Return last stored positions for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
                SELECT * FROM positions 
                WHERE portfolio_id = ?
            """,
            (portfolio_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


    def replace_open_orders_for_portfolio(self, portfolio_id: int, orders: List[Dict[str, Any]]):
        """Replace stored open orders snapshot for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM open_orders WHERE portfolio_id = ?",
            (portfolio_id,),
        )
        for order in orders:
            cursor.execute("""
                INSERT INTO open_orders (portfolio_id, order_id, symbol, side, price, amount, remaining, status, exchange_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_id,
                order.get('order_id'),
                order.get('symbol'),
                order.get('side'),
                order.get('price'),
                order.get('amount'),
                order.get('remaining'),
                order.get('status'),
                order.get('timestamp')
            ))
        self.conn.commit()

    def get_open_orders_for_portfolio(self, portfolio_id: int) -> List[Dict[str, Any]]:
        """Return last stored open orders for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
                SELECT * FROM open_orders 
                WHERE portfolio_id = ?
            """,
            (portfolio_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


    def get_net_positions_from_trades_for_portfolio(self, portfolio_id: int) -> Dict[str, float]:
        """Compute net position per symbol from recorded trades for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT symbol,
                   SUM(CASE WHEN action = 'BUY' THEN quantity ELSE -quantity END) as net_qty
            FROM trades
            WHERE portfolio_id = ?
            GROUP BY symbol
        """,
            (portfolio_id,),
        )
        rows = cursor.fetchall()
        return {row['symbol']: row['net_qty'] for row in rows}

    def get_trade_count_for_portfolio(self, portfolio_id: int) -> int:
        """Return count of trades for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM trades WHERE portfolio_id = ?", (portfolio_id,))
        row = cursor.fetchone()
        return row['cnt'] if row else 0

    def get_latest_trade_timestamp_for_portfolio(self, portfolio_id: int) -> Optional[str]:
        """Return latest trade timestamp for a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT timestamp FROM trades WHERE portfolio_id = ? ORDER BY timestamp DESC LIMIT 1",
            (portfolio_id,),
        )
        row = cursor.fetchone()
        return row["timestamp"] if row else None

    def get_distinct_trade_symbols_for_portfolio(self, portfolio_id: int) -> List[str]:
        """Return symbols traded within a portfolio."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM trades WHERE portfolio_id = ?", (portfolio_id,))
        return [row['symbol'] for row in cursor.fetchall()]
    
    def create_command(self, command: str):
        """Create a new command for the bot to execute."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO commands (command, status)
            VALUES (?, 'pending')
        """, (command,))
        self.conn.commit()
        logger.info(f"Created command: {command}")
    
    def get_pending_commands(self) -> List[Dict[str, Any]]:
        """Get all pending commands."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM commands 
            WHERE status = 'pending'
            ORDER BY created_at ASC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def mark_command_executed(self, command_id: int):
        """Mark a command as executed."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE commands 
            SET status = 'executed',
                executed_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), command_id))
        self.conn.commit()
    
    def clear_old_commands(self):
        """Clear all pending commands (called on bot startup)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE commands 
            SET status = 'cancelled',
                executed_at = ?
            WHERE status = 'pending'
        """, (datetime.now().isoformat(),))
        cancelled_count = cursor.rowcount
        self.conn.commit()
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} old pending command(s) from previous run")

    def prune_commands(self, retention_days: int):
        """Remove executed/cancelled commands older than the retention window."""
        if retention_days is None or retention_days <= 0:
            return
        cutoff = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM commands
            WHERE created_at < ?
              AND status IN ('executed', 'cancelled')
            """,
            (cutoff_str,),
        )
        self.conn.commit()

    # Health state helpers
    def set_health_state(self, key: str, value: str, detail: str = None):
        """Upsert health state key/value with optional detail JSON/text."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO health_state (key, value, detail, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                detail=excluded.detail,
                updated_at=CURRENT_TIMESTAMP
        """, (key, value, detail))
        self.conn.commit()

    def get_health_state(self) -> List[Dict[str, Any]]:
        """Return all health state entries."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT key, value, detail, updated_at
            FROM health_state
            ORDER BY key
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    # Trade plans for stops/targets
    def ensure_trade_plans_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL,
                target_price REAL,
                size REAL NOT NULL,
                version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'open',
                opened_at TEXT DEFAULT CURRENT_TIMESTAMP,
                closed_at TEXT,
                reason TEXT,
                entry_order_id TEXT,
                entry_client_order_id TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
            )
        """)
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_plans_portfolio_symbol_status ON trade_plans (portfolio_id, symbol, status)"
            )
        except Exception as exc:
            logger.debug(f"Could not create trade_plans index: {exc}")
        optional_columns = {
            "overnight_widened_at": "TEXT",
            "overnight_widen_version": "INTEGER",
            "last_widened_stop_price": "REAL",
            "last_widened_target_price": "REAL",
        }
        for column, col_type in optional_columns.items():
            if not self._column_exists(cursor, "trade_plans", column):
                try:
                    cursor.execute(f"ALTER TABLE trade_plans ADD COLUMN {column} {col_type}")
                except Exception as exc:
                    logger.debug(f"Could not add column {column} to trade_plans: {exc}")
        self.conn.commit()

    def create_trade_plan_for_portfolio(self, portfolio_id: int, symbol: str, side: str, entry_price: float, stop_price: float, target_price: float, size: float, reason: str = "", entry_order_id: str = None, entry_client_order_id: str = None) -> int:
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trade_plans (portfolio_id, symbol, side, entry_price, stop_price, target_price, size, reason, entry_order_id, entry_client_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (portfolio_id, symbol, side, entry_price, stop_price, target_price, size, reason, entry_order_id, entry_client_order_id))
        self.conn.commit()
        return cursor.lastrowid

    def update_trade_plan_prices(
        self,
        plan_id: int,
        stop_price: float = None,
        target_price: float = None,
        reason: str = None,
        widened_at: str | None = None,
        widen_stop_price: float | None = None,
        widen_target_price: float | None = None,
        widen_version: int | None = None,
    ):
        """Update stop/target and bump version."""
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        set_clauses = [
            "stop_price = COALESCE(?, stop_price)",
            "target_price = COALESCE(?, target_price)",
            "version = version + 1",
            "reason = COALESCE(?, reason)",
        ]
        params: list[Any] = [stop_price, target_price, reason]
        if widened_at is not None:
            set_clauses.append("overnight_widened_at = ?")
            params.append(widened_at)
        if widen_stop_price is not None:
            set_clauses.append("last_widened_stop_price = ?")
            params.append(widen_stop_price)
        if widen_target_price is not None:
            set_clauses.append("last_widened_target_price = ?")
            params.append(widen_target_price)
        if widen_version is not None:
            set_clauses.append("overnight_widen_version = ?")
            params.append(widen_version)
        set_expr = ",\n                ".join(set_clauses)
        cursor.execute(
            f"""
            UPDATE trade_plans
            SET {set_expr}
            WHERE id = ?
        """,
            (*params, plan_id),
        )
        self.conn.commit()

    def update_trade_plan_size(self, plan_id: int, size: float, reason: str = None):
        """Update plan size after partial closes and bump version."""
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE trade_plans
            SET size = ?,
                version = version + 1,
                reason = COALESCE(?, reason)
            WHERE id = ?
        """, (size, reason, plan_id))
        self.conn.commit()

    def update_trade_plan_status(self, plan_id: int, status: str, closed_at: str = None, reason: str = None):
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE trade_plans
            SET status = ?,
                closed_at = COALESCE(?, closed_at),
                reason = COALESCE(?, reason)
            WHERE id = ?
        """, (status, closed_at, reason, plan_id))
        self.conn.commit()

    def get_open_trade_plans_for_portfolio(self, portfolio_id: int) -> List[Dict[str, Any]]:
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM trade_plans
            WHERE portfolio_id = ? AND status = 'open'
            ORDER BY opened_at DESC
        """,
            (portfolio_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trade_plan_reason_by_order_for_portfolio(self, portfolio_id: int, order_id: str = None, client_order_id: str = None) -> Optional[str]:
        """
        Lookup a trade plan reason using either exchange order_id or client_order_id.
        Returns None when not found.
        """
        self.ensure_trade_plans_table()
        if not order_id and not client_order_id:
            return None

        cursor = self.conn.cursor()
        if order_id and client_order_id:
            cursor.execute(
                """
                SELECT reason FROM trade_plans
                WHERE portfolio_id = ? AND (entry_order_id = ? OR entry_client_order_id = ?)
                ORDER BY opened_at DESC
                LIMIT 1
                """,
                (portfolio_id, str(order_id), str(client_order_id)),
            )
        elif order_id:
            cursor.execute(
                """
                SELECT reason FROM trade_plans
                WHERE portfolio_id = ? AND entry_order_id = ?
                ORDER BY opened_at DESC
                LIMIT 1
                """,
                (portfolio_id, str(order_id)),
            )
        else:
            cursor.execute(
                """
                SELECT reason FROM trade_plans
                WHERE portfolio_id = ? AND entry_client_order_id = ?
                ORDER BY opened_at DESC
                LIMIT 1
                """,
                (portfolio_id, str(client_order_id)),
            )
        row = cursor.fetchone()
        return row["reason"] if row and row["reason"] else None

    def count_open_trade_plans_for_symbol_for_portfolio(self, portfolio_id: int, symbol: str) -> int:
        """Return number of open plans for a symbol scoped to a portfolio."""
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) as cnt FROM trade_plans
            WHERE portfolio_id = ? AND symbol = ? AND status = 'open'
        """,
            (portfolio_id, symbol),
        )
        row = cursor.fetchone()
        return row['cnt'] if row else 0
