import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from database import TradingDatabase
from cost_tracker import CostTracker
from strategy import LLMStrategy
from backtester import BacktestEngine
from logger_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

async def main():
    db = TradingDatabase()
    cost_tracker = CostTracker('GEMINI')
    ta = MagicMock() # Mock TA for now or use real one
    
    # Use real TA
    from technical_analysis import TechnicalAnalysis
    ta = TechnicalAnalysis()
    
    strategy = LLMStrategy(db, ta, cost_tracker)
    
    # Mock LLM to avoid costs and ensure signals
    strategy.model = MagicMock()
    mock_response = MagicMock()
    # Alternate between BUY and SELL to generate trades
    # We need a side effect to return different responses
    
    async def mock_generate(*args, **kwargs):
        # Return a valid JSON response
        return MagicMock(text='{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.001, "reason": "Backtest"}', usage_metadata=MagicMock(prompt_token_count=10, candidates_token_count=10))

    # Patch the _get_llm_decision method to avoid the async wait_for issues with mocks
    # Actually, let's just patch strategy.model.generate_content if we can make it async-compatible or just patch _get_llm_decision
    
    # Patching _get_llm_decision is safer
    strategy._get_llm_decision = AsyncMock(return_value='{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.001, "reason": "Backtest"}')
    
    engine = BacktestEngine(db, strategy, cost_tracker)
    
    logger.info("Running backtest on Session 1...")
    report = await engine.run(session_id=1)
    
    if report:
        print("\nBacktest Report:")
        print(f"Initial Balance: ${report['initial_balance']:,.2f}")
        print(f"Final Balance: ${report['final_balance']:,.2f}")
        print(f"Total Return: {report['total_return_pct']:.2f}%")
        print(f"Total Trades: {report['total_trades']}")
        print(f"Trades: {report['trades']}")
    else:
        print("Backtest failed or no data.")

if __name__ == "__main__":
    asyncio.run(main())
