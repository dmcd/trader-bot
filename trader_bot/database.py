import sqlite3
import logging
import os
import json
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Manages SQLite database for persistent trading data."""
    
    def __init__(self, db_path: Optional[str] = None):
        # Allow tests or env overrides to point at an isolated database
        self.db_path = db_path or os.getenv("TRADING_DB_PATH", "trading.db")
        self.conn = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database and tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                bot_version TEXT,
                starting_balance REAL,
                ending_balance REAL,
                base_currency TEXT,
                total_trades INTEGER DEFAULT 0,
                total_fees REAL DEFAULT 0.0,
                total_llm_cost REAL DEFAULT 0.0,
                net_pnl REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
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
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                client_order_id TEXT,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, trade_id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # LLM calls table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                decision TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # LLM trace table for full prompt/response/decision and optional execution result
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                prompt TEXT,
                response TEXT,
                decision_json TEXT,
                market_context TEXT,
                execution_result TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
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
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # OHLCV bars table (multi-timeframe)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_session_symbol_tf_ts ON ohlcv_bars (session_id, symbol, timeframe, timestamp DESC)")
        except Exception as e:
            logger.debug(f"Could not create ohlcv index: {e}")

        # Equity snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                equity REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Technical indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bb_upper REAL,
                bb_lower REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Positions snapshot table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL,
                exchange_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Open orders snapshot table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS open_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                order_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT,
                price REAL,
                amount REAL,
                remaining REAL,
                status TEXT,
                exchange_timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
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

        # Session stats cache to persist in-memory aggregates across restarts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_stats_cache (
                session_id INTEGER PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                total_fees REAL DEFAULT 0.0,
                gross_pnl REAL DEFAULT 0.0,
                total_llm_cost REAL DEFAULT 0.0,
                start_of_day_equity REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Risk state table to persist daily loss baselines across restarts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_state (
                session_id INTEGER PRIMARY KEY,
                start_of_day_equity REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Allow multiple sessions per version; keep a non-unique index for lookups
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_bot_version ON sessions(bot_version)")
        except Exception as e:
            logger.warning(f"Could not create non-unique index on bot_version: {e}")

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_or_create_session(self, starting_balance: float, bot_version: str, base_currency: Optional[str] = None) -> int:
        """Create a new session for this run, tagged by bot version."""
        cursor = self.conn.cursor()
        base_ccy = base_currency.upper() if base_currency else None
        cursor.execute("""
            INSERT INTO sessions (date, bot_version, starting_balance, base_currency)
            VALUES (?, ?, ?, ?)
        """, (date.today().isoformat(), bot_version, starting_balance, base_ccy))
        self.conn.commit()
        
        session_id = cursor.lastrowid
        logger.info(f"Created new session {session_id} for version {bot_version} (base_currency={base_ccy or 'unset'})")
        return session_id

    def get_session_id_by_version(self, bot_version: str) -> Optional[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM sessions WHERE bot_version = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (bot_version,)
        )
        row = cursor.fetchone()
        return row['id'] if row else None

    def list_bot_versions(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT bot_version FROM sessions WHERE bot_version IS NOT NULL ORDER BY created_at DESC")
        return [row['bot_version'] for row in cursor.fetchall()]

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a session row by id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_processed_trade_ids(self, session_id: int) -> set[str]:
        """Return trade_ids already seen for this session."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT trade_id FROM processed_trades WHERE session_id = ?", (session_id,))
        return {row["trade_id"] for row in cursor.fetchall() if row["trade_id"]}

    def record_processed_trade_ids(self, session_id: int, entries: List[tuple[str, Optional[str]]]):
        """Persist processed trade ids to avoid reprocessing across restarts."""
        if not entries:
            return
        payload = [(session_id, trade_id, client_oid) for trade_id, client_oid in entries if trade_id]
        if not payload:
            return
        cursor = self.conn.cursor()
        cursor.executemany(
            """
            INSERT OR IGNORE INTO processed_trades (session_id, trade_id, client_order_id)
            VALUES (?, ?, ?)
            """,
            payload,
        )
        self.conn.commit()
    
    def log_trade(self, session_id: int, symbol: str, action: str, 
                  quantity: float, price: float, fee: float, reason: str = "", liquidity: str = "unknown", realized_pnl: float = 0.0, trade_id: str = None, timestamp: str = None):
        """Log a trade to the database."""
        cursor = self.conn.cursor()
        
        # Check for duplicates if trade_id is provided
        if trade_id:
            cursor.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
            if cursor.fetchone():
                logger.debug(f"Skipping duplicate trade {trade_id}")
                return

        ts = timestamp if timestamp else datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO trades (session_id, timestamp, symbol, action, quantity, price, fee, liquidity, realized_pnl, reason, trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, ts, symbol, action, quantity, price, fee, liquidity, realized_pnl, reason, trade_id))
        
        # Update session trade count and fees
        cursor.execute("""
            UPDATE sessions 
            SET total_trades = total_trades + 1,
                total_fees = total_fees + ?
            WHERE id = ?
        """, (fee, session_id))
        
        self.conn.commit()
        logger.debug(f"Logged trade: {action} {quantity} {symbol} @ ${price}")

    def log_estimated_fee(self, session_id: int, order_id: str, estimated_fee: float, symbol: str, action: str):
        """Optional audit trail for estimated fees to compare with actuals."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS estimated_fees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    order_id TEXT,
                    symbol TEXT,
                    action TEXT,
                    estimated_fee REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            cursor.execute("""
                INSERT INTO estimated_fees (session_id, order_id, symbol, action, estimated_fee)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, str(order_id), symbol, action, estimated_fee))
            self.conn.commit()
        except Exception as e:
            logger.debug(f"Failed to log estimated fee: {e}")
    
    def log_llm_call(self, session_id: int, input_tokens: int, output_tokens: int, 
                     cost: float, decision: str = ""):
        """Log an LLM API call."""
        cursor = self.conn.cursor()
        total_tokens = input_tokens + output_tokens
        
        cursor.execute("""
            INSERT INTO llm_calls (session_id, timestamp, input_tokens, output_tokens, total_tokens, cost, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), input_tokens, output_tokens, total_tokens, cost, decision))
        
        # Update session LLM cost
        cursor.execute("""
            UPDATE sessions 
            SET total_llm_cost = total_llm_cost + ?
            WHERE id = ?
        """, (cost, session_id))
        
        self.conn.commit()
        logger.debug(f"Logged LLM call: {total_tokens} tokens, ${cost:.6f}")

    def log_llm_trace(self, session_id: int, prompt: str, response: str, decision_json: str = "", market_context: Any = None) -> int:
        """Persist full LLM prompt/response plus parsed decision and context; returns trace id."""
        cursor = self.conn.cursor()
        try:
            market_context_str = json.dumps(market_context, default=str) if market_context is not None else None
        except Exception:
            market_context_str = str(market_context)
        cursor.execute("""
            INSERT INTO llm_traces (session_id, timestamp, prompt, response, decision_json, market_context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), prompt, response, decision_json, market_context_str))
        self.conn.commit()
        return cursor.lastrowid

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

    def prune_llm_traces(self, session_id: int, retention_days: int):
        """Delete LLM traces older than the retention window for a session."""
        if retention_days is None or retention_days <= 0:
            return
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM llm_traces WHERE session_id = ? AND timestamp < ?",
            (session_id, cutoff),
        )
        self.conn.commit()

    def get_recent_llm_stats(self, session_id: int, limit: int = 20) -> Dict[str, Any]:
        """Return summary of recent LLM calls for telemetry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT decision FROM llm_calls
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        rows = [row['decision'] or '' for row in cursor.fetchall()]
        stats = {"total": len(rows), "schema_errors": 0, "clamped": 0}
        for d in rows:
            if d.startswith("schema_error"):
                stats["schema_errors"] += 1
            if "clamped" in d.lower():
                stats["clamped"] += 1
        return stats

    def get_recent_llm_traces(self, session_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Return recent LLM traces (decision + execution) for memory/context."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, decision_json, execution_result
            FROM llm_traces
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
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

    def log_market_data(self, session_id: int, symbol: str, price: float | int | None,
                       bid: float | int | None, ask: float | int | None, volume: float | int | None = None,
                       spread_pct: float | int | None = None, bid_size: float | int | None = None,
                       ask_size: float | int | None = None, ob_imbalance: float | int | None = None):
        """Log market data snapshot."""
        volume = self._preserve_integer_if_whole(volume)
        bid_size = self._preserve_integer_if_whole(bid_size)
        ask_size = self._preserve_integer_if_whole(ask_size)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_data (session_id, timestamp, symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), symbol, price, bid, ask, volume, spread_pct, bid_size, ask_size, ob_imbalance))
        self.conn.commit()

    def prune_market_data(self, session_id: int, retention_minutes: int):
        """Trim market data older than the retention window."""
        if retention_minutes is None or retention_minutes <= 0:
            return
        cutoff = (datetime.now() - timedelta(minutes=retention_minutes)).isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM market_data WHERE session_id = ? AND timestamp < ?",
            (session_id, cutoff),
        )
        self.conn.commit()
    
    def get_recent_trades(self, session_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades for context."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]

    def get_trades_for_session(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all trades for a session ordered chronologically."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_market_data(self, session_id: int, symbol: str, limit: int = 100, before_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent market data for technical analysis."""
        cursor = self.conn.cursor()
        
        if before_timestamp:
            cursor.execute("""
                SELECT * FROM market_data 
                WHERE session_id = ? AND symbol = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, symbol, before_timestamp, limit))
        else:
            cursor.execute("""
                SELECT * FROM market_data 
                WHERE session_id = ? AND symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, symbol, limit))
        
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["volume"] = self._preserve_integer_if_whole(row.get("volume"))
            row["bid_size"] = self._preserve_integer_if_whole(row.get("bid_size"))
            row["ask_size"] = self._preserve_integer_if_whole(row.get("ask_size"))
        return rows

    def log_equity_snapshot(self, session_id: int, equity: float):
        """Log mark-to-market equity snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO equity_snapshots (session_id, timestamp, equity)
            VALUES (?, ?, ?)
        """, (session_id, datetime.now().isoformat(), equity))
        self.conn.commit()

    def log_ohlcv_batch(self, session_id: int, symbol: str, timeframe: str, bars: List[Dict[str, Any]]):
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
                    session_id,
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
            INSERT INTO ohlcv_bars (session_id, timestamp, symbol, timeframe, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        self.conn.commit()

    def prune_ohlcv(self, session_id: int, symbol: str, timeframe: str, retain: int):
        """Keep only the most recent `retain` rows for a symbol/timeframe."""
        if retain is None or retain <= 0:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM ohlcv_bars
            WHERE session_id = ?
              AND symbol = ?
              AND timeframe = ?
              AND id NOT IN (
                SELECT id FROM ohlcv_bars
                WHERE session_id = ?
                  AND symbol = ?
                  AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
              )
            """,
            (session_id, symbol, timeframe, session_id, symbol, timeframe, retain),
        )
        self.conn.commit()

    def get_recent_ohlcv(self, session_id: int, symbol: str, timeframe: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch recent OHLCV bars for a symbol/timeframe."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_bars
            WHERE session_id = ? AND symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, symbol, timeframe, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_equity(self, session_id: int) -> Optional[float]:
        """Get most recent equity snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT equity FROM equity_snapshots
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (session_id,))
        row = cursor.fetchone()
        return row['equity'] if row else None

    def get_session_stats(self, session_id: int) -> Dict[str, Any]:
        """Get session statistics."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        session = dict(cursor.fetchone())

        # Prefer cached aggregates when present
        cache = self.get_session_stats_cache(session_id) or {}
        for key in ["total_trades", "total_fees", "gross_pnl", "total_llm_cost"]:
            if key in cache and cache[key] is not None:
                session[key] = cache[key]

        # If gross_pnl is missing, rebuild fee-exclusive realized PnL from trades
        if session.get('gross_pnl') is None:
            cursor.execute("""
                SELECT action, quantity, price, fee
                FROM trades
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            trades = cursor.fetchall()
            holdings = {}  # symbol-agnostic average cost; one-symbol assumption
            realized = 0.0
            total_fees = 0.0
            total_trades = 0
            avg_cost = 0.0
            qty_held = 0.0
            for t in trades:
                total_trades += 1
                fee = t['fee'] or 0.0
                total_fees += fee
                if t['action'].upper() == 'BUY':
                    new_qty = qty_held + t['quantity']
                    avg_cost = ((qty_held * avg_cost) + (t['quantity'] * t['price'])) / new_qty if new_qty > 0 else 0.0
                    qty_held = new_qty
                else:
                    realized += (t['price'] - avg_cost) * t['quantity']
                    qty_held = max(0.0, qty_held - t['quantity'])
            session['gross_pnl'] = realized
            session['total_trades'] = max(session.get('total_trades', 0) or 0, total_trades)
            session['total_fees'] = session.get('total_fees', 0.0) or total_fees

        # Derive net_pnl if missing
        if session.get('net_pnl') is None:
            session['net_pnl'] = (session.get('gross_pnl', 0.0) or 0.0) - (session.get('total_fees', 0.0) or 0.0) - (session.get('total_llm_cost', 0.0) or 0.0)

        return session
    
    def update_session_balance(self, session_id: int, ending_balance: float, net_pnl: float):
        """Update session ending balance and net PnL."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions 
            SET ending_balance = ?,
                net_pnl = ?
            WHERE id = ?
        """, (ending_balance, net_pnl, session_id))
        self.conn.commit()

    def update_session_totals(self, session_id: int, total_trades: int = None, total_fees: float = None, total_llm_cost: float = None, net_pnl: float = None):
        """Update aggregate totals on sessions row."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET total_trades = COALESCE(?, total_trades),
                total_fees = COALESCE(?, total_fees),
                total_llm_cost = COALESCE(?, total_llm_cost),
                net_pnl = COALESCE(?, net_pnl)
            WHERE id = ?
        """, (total_trades, total_fees, total_llm_cost, net_pnl, session_id))
        self.conn.commit()

    def update_session_starting_balance(self, session_id: int, starting_balance: float):
        """Reset starting balance for a session (useful in PAPER to avoid stale baselines)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET starting_balance = ? WHERE id = ?",
            (starting_balance, session_id),
        )
        self.conn.commit()

    def replace_positions(self, session_id: int, positions: List[Dict[str, Any]]):
        """Replace stored positions snapshot for the session."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM positions WHERE session_id = ?", (session_id,))
        for pos in positions:
            cursor.execute("""
                INSERT INTO positions (session_id, symbol, quantity, avg_price, exchange_timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                pos.get('symbol'),
                pos.get('quantity', 0),
                pos.get('avg_price'),
                pos.get('timestamp')
            ))
        self.conn.commit()

    def get_positions(self, session_id: int) -> List[Dict[str, Any]]:
        """Return last stored positions for the session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM positions 
            WHERE session_id = ?
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def replace_open_orders(self, session_id: int, orders: List[Dict[str, Any]]):
        """Replace stored open orders snapshot for the session."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM open_orders WHERE session_id = ?", (session_id,))
        for order in orders:
            cursor.execute("""
                INSERT INTO open_orders (session_id, order_id, symbol, side, price, amount, remaining, status, exchange_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
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

    def get_open_orders(self, session_id: int) -> List[Dict[str, Any]]:
        """Return last stored open orders for the session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM open_orders 
            WHERE session_id = ?
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_net_positions_from_trades(self, session_id: int) -> Dict[str, float]:
        """Compute net position per symbol from recorded trades."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT symbol,
                   SUM(CASE WHEN action = 'BUY' THEN quantity ELSE -quantity END) as net_qty
            FROM trades
            WHERE session_id = ?
            GROUP BY symbol
        """, (session_id,))
        rows = cursor.fetchall()
        return {row['symbol']: row['net_qty'] for row in rows}

    def get_trade_count(self, session_id: int) -> int:
        """Return count of trades for a session."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM trades WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        return row['cnt'] if row else 0

    def get_latest_trade_timestamp(self, session_id: int) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp FROM trades WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1", (session_id,))
        row = cursor.fetchone()
        return row['timestamp'] if row else None

    def get_distinct_trade_symbols(self, session_id: int) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM trades WHERE session_id = ?", (session_id,))
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
            logger.info(f"Cancelled {cancelled_count} old pending command(s) from previous session")

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
                session_id INTEGER NOT NULL,
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
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self.conn.commit()

    def create_trade_plan(self, session_id: int, symbol: str, side: str, entry_price: float, stop_price: float, target_price: float, size: float, reason: str = "", entry_order_id: str = None, entry_client_order_id: str = None) -> int:
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trade_plans (session_id, symbol, side, entry_price, stop_price, target_price, size, reason, entry_order_id, entry_client_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, symbol, side, entry_price, stop_price, target_price, size, reason, entry_order_id, entry_client_order_id))
        self.conn.commit()
        return cursor.lastrowid

    def update_trade_plan_prices(self, plan_id: int, stop_price: float = None, target_price: float = None, reason: str = None):
        """Update stop/target and bump version."""
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE trade_plans
            SET stop_price = COALESCE(?, stop_price),
                target_price = COALESCE(?, target_price),
                version = version + 1,
                reason = COALESCE(?, reason)
            WHERE id = ?
        """, (stop_price, target_price, reason, plan_id))
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

    def get_open_trade_plans(self, session_id: int) -> List[Dict[str, Any]]:
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trade_plans
            WHERE session_id = ? AND status = 'open'
            ORDER BY opened_at DESC
        """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_trade_plan_reason_by_order(self, session_id: int, order_id: str = None, client_order_id: str = None) -> Optional[str]:
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
                WHERE session_id = ? AND (entry_order_id = ? OR entry_client_order_id = ?)
                ORDER BY opened_at DESC
                LIMIT 1
                """,
                (session_id, str(order_id), str(client_order_id)),
            )
        elif order_id:
            cursor.execute(
                """
                SELECT reason FROM trade_plans
                WHERE session_id = ? AND entry_order_id = ?
                ORDER BY opened_at DESC
                LIMIT 1
                """,
                (session_id, str(order_id)),
            )
        else:
            cursor.execute(
                """
                SELECT reason FROM trade_plans
                WHERE session_id = ? AND entry_client_order_id = ?
                ORDER BY opened_at DESC
                LIMIT 1
                """,
                (session_id, str(client_order_id)),
            )
        row = cursor.fetchone()
        return row["reason"] if row and row["reason"] else None

    def count_open_trade_plans_for_symbol(self, session_id: int, symbol: str) -> int:
        """Return number of open plans for a symbol."""
        self.ensure_trade_plans_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM trade_plans
            WHERE session_id = ? AND symbol = ? AND status = 'open'
        """, (session_id, symbol))
        row = cursor.fetchone()
        return row['cnt'] if row else 0

    # Session stats cache helpers
    def get_session_stats_cache(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Return persisted session stats aggregates if present."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT session_id, total_trades, total_fees, gross_pnl, total_llm_cost, start_of_day_equity
            FROM session_stats_cache
            WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_session_stats_cache(self, session_id: int, stats: Dict[str, Any]):
        """Upsert session stats aggregates for restart resilience."""
        existing = self.get_session_stats_cache(session_id) or {}
        merged = {**existing, **(stats or {})}
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO session_stats_cache (session_id, total_trades, total_fees, gross_pnl, total_llm_cost, start_of_day_equity, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                total_trades=excluded.total_trades,
                total_fees=excluded.total_fees,
                gross_pnl=excluded.gross_pnl,
                total_llm_cost=excluded.total_llm_cost,
                start_of_day_equity=excluded.start_of_day_equity,
                updated_at=CURRENT_TIMESTAMP
        """, (
            session_id,
            merged.get('total_trades', 0) or 0,
            merged.get('total_fees', 0.0) or 0.0,
            merged.get('gross_pnl', 0.0) or 0.0,
            merged.get('total_llm_cost', 0.0) or 0.0,
            merged.get('start_of_day_equity', None),
        ))
        self.conn.commit()

    def get_start_of_day_equity(self, session_id: int) -> Optional[float]:
        """Return persisted start-of-day equity for loss checks."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT start_of_day_equity FROM risk_state
            WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        if row and row['start_of_day_equity'] is not None:
            return row['start_of_day_equity']

        # Fallback to cache column if present
        stats = self.get_session_stats_cache(session_id)
        if stats and stats.get('start_of_day_equity') is not None:
            return stats['start_of_day_equity']
        return None

    def set_start_of_day_equity(self, session_id: int, equity: float):
        """Persist start-of-day equity, keeping cache in sync."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO risk_state (session_id, start_of_day_equity)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                start_of_day_equity=excluded.start_of_day_equity
        """, (session_id, equity))
        self.conn.commit()
        try:
            self.set_session_stats_cache(session_id, {'start_of_day_equity': equity})
        except Exception as e:
            logger.debug(f"Could not sync start_of_day_equity to cache: {e}")
