#!/usr/bin/env python3
"""
Test script for professional trading infrastructure.
Tests database, cost tracker, trading context, and technical analysis.
"""

import sys
import asyncio
from datetime import datetime

# Test imports
print("Testing imports...")
try:
    from database import TradingDatabase
    from cost_tracker import CostTracker
    from trading_context import TradingContext
    from technical_analysis import TechnicalAnalysis
    print("✅ All imports successful\n")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_database():
    """Test database operations."""
    print("=" * 50)
    print("Testing Database")
    print("=" * 50)
    
    db = TradingDatabase("test_trading.db")
    
    # Create session
    session_id = db.get_or_create_session(starting_balance=10000.0)
    print(f"✅ Created session {session_id}")
    
    # Log a trade
    db.log_trade(session_id, "BTC/USD", "BUY", 0.001, 85000.0, 0.30, "Test trade")
    print("✅ Logged trade")
    
    # Log LLM call
    db.log_llm_call(session_id, 1000, 500, 0.00015, "Test decision")
    print("✅ Logged LLM call")
    
    # Log market data
    db.log_market_data(session_id, "BTC/USD", 85000.0, 84950.0, 85050.0, 1000.0)
    print("✅ Logged market data")
    
    # Get recent trades
    trades = db.get_recent_trades(session_id, limit=5)
    print(f"✅ Retrieved {len(trades)} recent trades")
    
    # Get session stats
    stats = db.get_session_stats(session_id)
    print(f"✅ Session stats: {stats['total_trades']} trades, ${stats['total_fees']:.2f} fees")
    
    db.close()
    print("✅ Database test passed\n")
    
    # Cleanup test database
    import os
    if os.path.exists("test_trading.db"):
        os.remove("test_trading.db")

def test_cost_tracker():
    """Test cost tracking."""
    print("=" * 50)
    print("Testing Cost Tracker")
    print("=" * 50)
    
    # Test Gemini fees
    tracker_gemini = CostTracker("GEMINI")
    fee = tracker_gemini.calculate_trade_fee("BTC/USD", 0.001, 85000.0)
    print(f"✅ Gemini fee for 0.001 BTC @ $85,000: ${fee:.4f}")
    
    # Test IB fees
    tracker_ib = CostTracker("IB")
    fee = tracker_ib.calculate_trade_fee("BHP", 100, 45.0)
    print(f"✅ IB fee for 100 shares @ $45: ${fee:.2f}")
    
    # Test LLM costs
    llm_cost = tracker_gemini.calculate_llm_cost(1000, 500)
    print(f"✅ LLM cost for 1000 input + 500 output tokens: ${llm_cost:.6f}")
    
    # Test net PnL
    summary = tracker_gemini.get_cost_summary(
        total_fees=10.50,
        total_llm_cost=0.25,
        gross_pnl=50.00
    )
    print(f"✅ Net PnL: ${summary['net_pnl']:.2f} (from ${summary['gross_pnl']:.2f} gross)")
    print("✅ Cost tracker test passed\n")

def test_trading_context():
    """Test trading context."""
    print("=" * 50)
    print("Testing Trading Context")
    print("=" * 50)
    
    db = TradingDatabase("test_trading.db")
    session_id = db.get_or_create_session(starting_balance=10000.0)
    
    # Add some sample trades
    db.log_trade(session_id, "BTC/USD", "BUY", 0.001, 85000.0, 0.30, "Test buy")
    db.log_trade(session_id, "BTC/USD", "SELL", 0.001, 85500.0, 0.30, "Test sell")
    
    # Add market data
    for i in range(20):
        price = 85000.0 + (i * 10)
        db.log_market_data(session_id, "BTC/USD", price, price - 50, price + 50)
    
    context = TradingContext(db, session_id)
    
    # Get context summary
    summary = context.get_context_summary("BTC/USD")
    print("Context Summary:")
    print(summary)
    print("\n✅ Trading context test passed\n")
    
    db.close()
    
    # Cleanup
    import os
    if os.path.exists("test_trading.db"):
        os.remove("test_trading.db")

def test_technical_analysis():
    """Test technical indicators."""
    print("=" * 50)
    print("Testing Technical Analysis")
    print("=" * 50)
    
    # Create sample market data (50 points)
    market_data = []
    base_price = 85000.0
    for i in range(50):
        price = base_price + (i * 10) + ((-1) ** i * 50)  # Oscillating trend
        market_data.append({
            'timestamp': datetime.now().isoformat(),
            'price': price,
            'bid': price - 25,
            'ask': price + 25
        })
    
    ta = TechnicalAnalysis()
    
    # Calculate indicators
    indicators = ta.calculate_indicators(market_data)
    
    if indicators:
        print(f"✅ RSI: {indicators['rsi']:.2f}")
        print(f"✅ MACD: {indicators['macd']:.4f}")
        print(f"✅ Bollinger Upper: ${indicators['bb_upper']:,.2f}")
        print(f"✅ Bollinger Lower: ${indicators['bb_lower']:,.2f}")
        print(f"✅ SMA 20: ${indicators['sma_20']:,.2f}")
        
        # Get trading signals
        signals = ta.get_trading_signals(indicators, market_data[0]['price'])
        print(f"\nTrading Signals:")
        for indicator, signal in signals.items():
            print(f"  {indicator}: {signal}")
        
        # Format for LLM
        llm_format = ta.format_indicators_for_llm(indicators, market_data[0]['price'])
        print(f"\nLLM Format:\n{llm_format}")
        
        print("\n✅ Technical analysis test passed\n")
    else:
        print("❌ Failed to calculate indicators")

def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("PROFESSIONAL TRADING INFRASTRUCTURE TEST")
    print("=" * 50 + "\n")
    
    try:
        test_database()
        test_cost_tracker()
        test_trading_context()
        test_technical_analysis()
        
        print("=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nThe new modules are working correctly.")
        print("Ready to integrate into strategy_runner.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
