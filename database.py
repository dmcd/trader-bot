import sqlite3
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class TradingDatabase:
    """Manages SQLite database for persistent trading data."""
    
    def __init__(self, db_path: str = "trading.db"):
        self.db_path = db_path
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
                date TEXT UNIQUE NOT NULL,
                starting_balance REAL,
                ending_balance REAL,
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

        # Backfill liquidity column for existing installs (SQLite allows additive ALTER)
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN liquidity TEXT DEFAULT 'unknown'")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN realized_pnl REAL DEFAULT 0.0")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN trade_id TEXT")
        except sqlite3.OperationalError:
            pass
        
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
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

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

        # Session stats cache to persist in-memory aggregates across restarts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_stats_cache (
                session_id INTEGER PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                total_fees REAL DEFAULT 0.0,
                gross_pnl REAL DEFAULT 0.0,
                total_llm_cost REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_or_create_session(self, starting_balance: float) -> int:
        """Get today's session or create a new one."""
        today = date.today().isoformat()
        cursor = self.conn.cursor()
        
        # Try to get existing session
        cursor.execute("SELECT id FROM sessions WHERE date = ?", (today,))
        row = cursor.fetchone()
        
        if row:
            session_id = row['id']
            logger.info(f"Loaded existing session {session_id} for {today}")
            return session_id
        
        # Create new session
        cursor.execute("""
            INSERT INTO sessions (date, starting_balance)
            VALUES (?, ?)
        """, (today, starting_balance))
        self.conn.commit()
        
        session_id = cursor.lastrowid
        logger.info(f"Created new session {session_id} for {today}")
        return session_id

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a session row by id."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def log_trade(self, session_id: int, symbol: str, action: str, 
                  quantity: float, price: float, fee: float, reason: str = "", liquidity: str = "unknown", realized_pnl: float = 0.0, trade_id: str = None):
        """Log a trade to the database."""
        cursor = self.conn.cursor()
        
        # Check for duplicates if trade_id is provided
        if trade_id:
            cursor.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
            if cursor.fetchone():
                logger.debug(f"Skipping duplicate trade {trade_id}")
                return

        cursor.execute("""
            INSERT INTO trades (session_id, timestamp, symbol, action, quantity, price, fee, liquidity, realized_pnl, reason, trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), symbol, action, quantity, price, fee, liquidity, realized_pnl, reason, trade_id))
        
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
    
    def log_market_data(self, session_id: int, symbol: str, price: float, 
                       bid: float, ask: float, volume: float = 0.0):
        """Log market data snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_data (session_id, timestamp, symbol, price, bid, ask, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, datetime.now().isoformat(), symbol, price, bid, ask, volume))
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
        
        return [dict(row) for row in cursor.fetchall()]

    def log_equity_snapshot(self, session_id: int, equity: float):
        """Log mark-to-market equity snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO equity_snapshots (session_id, timestamp, equity)
            VALUES (?, ?, ?)
        """, (session_id, datetime.now().isoformat(), equity))
        self.conn.commit()

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
        
        # Calculate win rate
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN action = 'BUY' THEN -1 * quantity * price ELSE quantity * price END) as gross_pnl
            FROM trades
            WHERE session_id = ?
        """, (session_id,))
        
        stats = dict(cursor.fetchone())
        session.update(stats)
        
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
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    # Session stats cache helpers
    def get_session_stats_cache(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Return persisted session stats aggregates if present."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT session_id, total_trades, total_fees, gross_pnl, total_llm_cost
            FROM session_stats_cache
            WHERE session_id = ?
        """, (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def set_session_stats_cache(self, session_id: int, stats: Dict[str, Any]):
        """Upsert session stats aggregates for restart resilience."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO session_stats_cache (session_id, total_trades, total_fees, gross_pnl, total_llm_cost, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                total_trades=excluded.total_trades,
                total_fees=excluded.total_fees,
                gross_pnl=excluded.gross_pnl,
                total_llm_cost=excluded.total_llm_cost,
                updated_at=CURRENT_TIMESTAMP
        """, (
            session_id,
            stats.get('total_trades', 0) or 0,
            stats.get('total_fees', 0.0) or 0.0,
            stats.get('gross_pnl', 0.0) or 0.0,
            stats.get('total_llm_cost', 0.0) or 0.0,
        ))
        self.conn.commit()
