import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass
from strategy import BaseStrategy, StrategySignal
from database import TradingDatabase
from cost_tracker import CostTracker

logger = logging.getLogger(__name__)

@dataclass
class BacktestContext:
    current_time: float
    current_iso_time: str
    
    def get_context_summary(self, symbol):
        return "" # Mock or implement if needed

class BacktestEngine:
    def __init__(self, db: TradingDatabase, strategy: BaseStrategy, cost_tracker: CostTracker):
        self.db = db
        self.strategy = strategy
        self.cost_tracker = cost_tracker
        self.trades = []
        self.holdings = {} # symbol -> {'qty': float, 'avg_cost': float}
        self.cash = 0.0
        self.initial_balance = 0.0
        self.equity_curve = []

    def _update_holdings(self, symbol: str, action: str, quantity: float, price: float, fee: float):
        pos = self.holdings.get(symbol, {'qty': 0.0, 'avg_cost': 0.0})
        qty = pos['qty']
        avg_cost = pos['avg_cost']
        
        if action == 'BUY':
            cost = quantity * price
            self.cash -= (cost + fee)
            
            new_qty = qty + quantity
            if new_qty > 0:
                new_avg = ((qty * avg_cost) + cost) / new_qty
            else:
                new_avg = 0.0
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': new_avg}
            
        elif action == 'SELL':
            proceeds = quantity * price
            self.cash += (proceeds - fee)
            
            new_qty = max(0.0, qty - quantity)
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': avg_cost if new_qty > 0 else 0.0}

    def get_current_equity(self, current_prices: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, pos in self.holdings.items():
            price = current_prices.get(symbol, 0)
            equity += pos['qty'] * price
        return equity

    async def run(self, session_id: int, initial_cash: float = 10000.0):
        """Run backtest on a specific session."""
        self.cash = initial_cash
        self.initial_balance = initial_cash
        self.trades = []
        self.holdings = {}
        self.equity_curve = []
        
        # Fetch market data
        # We need ALL data for the session, ordered chronologically
        # get_recent_market_data gets DESC limit N. We need ASC all.
        # We'll need a new DB method or just fetch manually here.
        # For now, let's add a method to DB or use raw query.
        # Adding method to DB is cleaner but I can just do it here for speed if DB is accessible.
        # DB is accessible.
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM market_data 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        market_data_rows = [dict(row) for row in cursor.fetchall()]
        
        if not market_data_rows:
            logger.warning(f"No market data found for session {session_id}")
            return
            
        logger.info(f"Starting backtest on {len(market_data_rows)} data points...")
        
        # Group by timestamp (approximate if multiple symbols)
        # Assuming single symbol stream for now or interleaved.
        # Strategy expects a dict of {symbol: data}
        
        for row in market_data_rows:
            symbol = row['symbol']
            price = row['price']
            timestamp_str = row['timestamp']
            try:
                dt = datetime.fromisoformat(timestamp_str)
                timestamp = dt.timestamp()
            except:
                continue
                
            # Create context
            context = BacktestContext(current_time=timestamp, current_iso_time=timestamp_str)
            
            # Prepare market data input
            market_data_input = {symbol: row}
            
            # Calculate PnL for strategy input
            current_equity = self.get_current_equity({symbol: price})
            current_pnl = current_equity - self.initial_balance # This is net PnL roughly
            # Strategy expects portfolio value usually? 
            # LLM prompt says "Portfolio Value: $X".
            # So we pass current_equity.
            
            # Generate signal
            # We mock the LLM call to avoid costs? 
            # Or do we want to actually call the LLM?
            # The user wants to "find the most profitable instructions", so we probably want to call the LLM.
            # But that's expensive and slow.
            # For "Phase 1", we just want the capability.
            # We should probably add a "mock_llm" flag to the strategy or engine.
            
            signal = await self.strategy.generate_signal(
                session_id, 
                market_data_input, 
                current_equity, 
                context
            )
            
            if signal:
                # Execute
                if signal.action in ['BUY', 'SELL']:
                    # Calculate fee
                    fee = self.cost_tracker.calculate_trade_fee(symbol, signal.quantity, price, signal.action)
                    
                    self._update_holdings(symbol, signal.action, signal.quantity, price, fee)
                    
                    trade_record = {
                        'timestamp': timestamp_str,
                        'symbol': symbol,
                        'action': signal.action,
                        'quantity': signal.quantity,
                        'price': price,
                        'fee': fee,
                        'reason': signal.reason
                    }
                    self.trades.append(trade_record)
                    
                    # Notify strategy
                    self.strategy.on_trade_executed(timestamp)
                    
            # Record equity curve
            self.equity_curve.append({'timestamp': timestamp_str, 'equity': current_equity})

        # Final report
        final_equity = self.get_current_equity({symbol: price}) # Use last price
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_equity,
            'total_return_pct': total_return,
            'total_trades': len(self.trades),
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
