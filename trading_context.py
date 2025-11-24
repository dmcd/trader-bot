import logging
from datetime import datetime
from typing import Dict, List, Any
from database import TradingDatabase

logger = logging.getLogger(__name__)

class TradingContext:
    """Provides rich trading context for LLM decision-making."""

    def __init__(self, db: TradingDatabase, session_id: int):
        self.db = db
        self.session_id = session_id
    
    def get_context_summary(self, symbol: str, open_orders: List[Dict[str, Any]] = None) -> str:
        """Generate comprehensive context summary for LLM."""
        
        # Get session stats
        session = self.db.get_session_stats(self.session_id)
        
        # Get recent trades
        recent_trades = list(reversed(self.db.get_recent_trades(self.session_id, limit=50)))  # chronological
        
        # Calculate session duration
        session_start = datetime.fromisoformat(session['created_at'])
        duration = datetime.now() - session_start
        hours = duration.total_seconds() / 3600
        
        # Calculate win rate using chronological buy->sell pairing
        wins = losses = 0
        open_position = None
        for trade in recent_trades:
            action = trade['action'].upper()
            qty = trade['quantity']
            price = trade['price']
            fee = trade.get('fee', 0.0) or 0.0
            if action == 'BUY':
                # Start/aggregate position
                if open_position is None:
                    open_position = {'qty': qty, 'cost': qty * price, 'fees': fee}
                else:
                    open_position['qty'] += qty
                    open_position['cost'] += qty * price
                    open_position['fees'] += fee
            elif action == 'SELL' and open_position:
                sell_proceeds = qty * price
                # Assume FIFO: close against existing open qty
                closed_qty = min(open_position['qty'], qty)
                avg_cost = open_position['cost'] / open_position['qty'] if open_position['qty'] else 0.0
                cost_closed = closed_qty * avg_cost
                pnl = sell_proceeds - cost_closed - fee - open_position.get('fees', 0.0)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                open_position['qty'] -= closed_qty
                open_position['cost'] -= cost_closed
                if open_position['qty'] <= 1e-9:
                    open_position = None

        total_closed = wins + losses
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        
        # Get recent market data for trend analysis
        market_data = self.db.get_recent_market_data(self.session_id, symbol, limit=20)
        
        price_trend = "Unknown"
        if len(market_data) >= 2:
            recent_price = market_data[0]['price']
            older_price = market_data[-1]['price']
            price_change = ((recent_price - older_price) / older_price) * 100
            
            if price_change > 0.5:
                price_trend = f"Upward +{price_change:.2f}%"
            elif price_change < -0.5:
                price_trend = f"Downward {price_change:.2f}%"
            else:
                price_trend = f"Sideways ({price_change:+.2f}%)"
        
        # Get current positions and open orders (prefer provided snapshot)
        positions = self.db.get_positions(self.session_id)
        if open_orders is None:
            open_orders = self.db.get_open_orders(self.session_id)
        
        # Build context summary
        context = f"""
=== TRADING SESSION CONTEXT ===
Session Date: {session['date']}
Duration: {hours:.1f} hours
Starting Balance: ${session['starting_balance']:,.2f}

Performance:
- Total Trades: {session['total_trades']}
- Win Rate: {win_rate:.1f}% ({wins} wins, {losses} losses)
- Total Fees Paid: ${session['total_fees']:.2f}
- Total LLM Costs: ${session['total_llm_cost']:.4f}
- Net PnL: ${session['net_pnl']:.2f}

Current Positions:
"""
        
        # Add position details with unrealized PnL
        if positions:
            total_exposure = 0.0
            for pos in positions:
                sym = pos['symbol']
                qty = pos['quantity']
                avg_price = pos.get('avg_price') or 0
                
                # Get current price for this symbol
                current_price = avg_price  # Default fallback
                recent_data = self.db.get_recent_market_data(self.session_id, sym, limit=1)
                if recent_data and recent_data[0].get('price'):
                    current_price = recent_data[0]['price']
                
                # Skip if we don't have valid prices
                if not avg_price or not current_price:
                    continue
                
                # Calculate unrealized PnL
                cost_basis = qty * avg_price
                current_value = qty * current_price
                unrealized_pnl = current_value - cost_basis

                total_exposure += qty * current_price
                
                context += f"  - {sym}: {qty:.6f} units @ ${avg_price:,.2f} avg (Current: ${current_price:,.2f}, Unrealized PnL: ${unrealized_pnl:+,.2f})\n"
            
            context += f"  Total Exposure: ${total_exposure:,.2f}\n"
        else:
            context += "  No open positions\n"

        context += "Open Orders:\n"

        if open_orders:
            for order in open_orders:
                side = (order.get('side') or '').upper() or 'N/A'
                sym = order.get('symbol') or 'Unknown'
                qty = order.get('amount') or 0
                remaining = order.get('remaining') if order.get('remaining') is not None else qty
                price = order.get('price')
                status = order.get('status') or 'open'
                price_str = f"@ ${price:,.2f}" if price else "@ mkt"
                order_id = order.get('order_id')
                id_str = f"#{order_id}" if order_id is not None else "#?"
                context += f"  - {id_str} {side} {qty:.6f} {sym} {price_str} (rem {remaining:.6f}, {status})\n"
        else:
            context += "  No open orders\n"

        context += """
Recent Activity:
"""
        
        # Add recent trades
        if recent_trades:
            context += "Last 5 Trades:\n"
            for trade in list(reversed(recent_trades))[:5]:  # show most recent
                timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
                context += f"  {timestamp} - {trade['action']} {trade['quantity']:.6f} {trade['symbol']} @ ${trade['price']:,.2f}\n"
        else:
            context += "  No trades yet today\n"
        
        # Add market trend
        context += f"\nMarket Trend (last 20 observations):\n"
        context += f"  {symbol}: {price_trend}\n"
        
        # Add trading advice based on performance
        context += "\nContext Notes:\n"
        if session['total_trades'] > 5 and win_rate < 40:
            context += "  ⚠️ Low win rate - consider being more selective\n"
        if session['total_fees'] > abs(session['net_pnl']) * 0.5:
            context += "  ⚠️ High fees relative to PnL - reduce trade frequency\n"
        if len(recent_trades) >= 3:
            last_3 = recent_trades[:3]
            if all(t['action'] == last_3[0]['action'] for t in last_3):
                context += f"  ⚠️ Last 3 trades were all {last_3[0]['action']} - consider diversifying\n"
        
        return context.strip()
    
    def get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics."""
        recent_trades = list(reversed(self.db.get_recent_trades(self.session_id, limit=50)))  # chronological
        
        if not recent_trades:
            return {
                'total_trades': 0,
                'avg_profit': 0,
                'win_rate': 0,
                'last_trade_profitable': None
            }
        
        profits = []
        open_position = None
        for trade in recent_trades:
            action = trade['action'].upper()
            qty = trade['quantity']
            price = trade['price']
            fee = trade.get('fee', 0.0) or 0.0
            if action == 'BUY':
                if open_position is None:
                    open_position = {'qty': qty, 'cost': qty * price, 'fees': fee}
                else:
                    open_position['qty'] += qty
                    open_position['cost'] += qty * price
                    open_position['fees'] += fee
            elif action == 'SELL' and open_position:
                closed_qty = min(open_position['qty'], qty)
                avg_cost = open_position['cost'] / open_position['qty'] if open_position['qty'] else 0.0
                pnl = (closed_qty * price) - (closed_qty * avg_cost) - fee - open_position.get('fees', 0.0)
                profits.append(pnl)
                open_position['qty'] -= closed_qty
                open_position['cost'] -= closed_qty * avg_cost
                if open_position['qty'] <= 1e-9:
                    open_position = None

        wins = sum(1 for p in profits if p > 0)
        return {
            'total_trades': len(recent_trades),
            'avg_profit': sum(profits) / len(profits) if profits else 0,
            'win_rate': (wins / len(profits) * 100) if profits else 0,
            'last_trade_profitable': profits[-1] > 0 if profits else None
        }
