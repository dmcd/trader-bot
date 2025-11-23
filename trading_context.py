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
    
    def get_context_summary(self, symbol: str) -> str:
        """Generate comprehensive context summary for LLM."""
        
        # Get session stats
        session = self.db.get_session_stats(self.session_id)
        
        # Get recent trades
        recent_trades = self.db.get_recent_trades(self.session_id, limit=10)
        
        # Calculate session duration
        session_start = datetime.fromisoformat(session['created_at'])
        duration = datetime.now() - session_start
        hours = duration.total_seconds() / 3600
        
        # Calculate win rate
        if recent_trades:
            # Simple win calculation: compare consecutive buy/sell pairs
            wins = 0
            losses = 0
            for i in range(len(recent_trades) - 1):
                if recent_trades[i]['action'] == 'SELL' and i + 1 < len(recent_trades):
                    sell_price = recent_trades[i]['price']
                    # Find corresponding buy
                    for j in range(i + 1, len(recent_trades)):
                        if recent_trades[j]['action'] == 'BUY':
                            buy_price = recent_trades[j]['price']
                            if sell_price > buy_price:
                                wins += 1
                            else:
                                losses += 1
                            break
            
            total_closed = wins + losses
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        else:
            win_rate = 0
            wins = 0
            losses = 0
        
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
        
        # Get current positions from database
        positions = self.db.get_positions(self.session_id)
        
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
                
                total_exposure += current_value
                
                context += f"  - {sym}: {qty:.6f} units @ ${avg_price:,.2f} avg (Current: ${current_price:,.2f}, Unrealized PnL: ${unrealized_pnl:+,.2f})\n"
            
            context += f"  Total Exposure: ${total_exposure:,.2f}\n"
        else:
            context += "  No open positions\n"
        
        context += """
Recent Activity:
"""
        
        # Add recent trades
        if recent_trades:
            context += "Last 5 Trades:\n"
            for trade in recent_trades[:5]:
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
        recent_trades = self.db.get_recent_trades(self.session_id, limit=10)
        
        if not recent_trades:
            return {
                'total_trades': 0,
                'avg_profit': 0,
                'win_rate': 0,
                'last_trade_profitable': None
            }
        
        # Calculate metrics
        profits = []
        for i in range(len(recent_trades) - 1):
            if recent_trades[i]['action'] == 'SELL':
                sell_value = recent_trades[i]['quantity'] * recent_trades[i]['price']
                # Find corresponding buy
                for j in range(i + 1, len(recent_trades)):
                    if recent_trades[j]['action'] == 'BUY':
                        buy_value = recent_trades[j]['quantity'] * recent_trades[j]['price']
                        profit = sell_value - buy_value - recent_trades[i]['fee'] - recent_trades[j]['fee']
                        profits.append(profit)
                        break
        
        wins = sum(1 for p in profits if p > 0)
        
        return {
            'total_trades': len(recent_trades),
            'avg_profit': sum(profits) / len(profits) if profits else 0,
            'win_rate': (wins / len(profits) * 100) if profits else 0,
            'last_trade_profitable': profits[0] > 0 if profits else None
        }
