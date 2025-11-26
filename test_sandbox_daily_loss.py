import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from strategy_runner import StrategyRunner
from risk_manager import RiskManager

@pytest.mark.asyncio
async def test_sandbox_daily_loss_check():
    # Mock dependencies
    mock_bot = AsyncMock()
    mock_bot.connect_async = AsyncMock()
    mock_bot.get_equity_async = AsyncMock(return_value=100000.0)
    mock_bot.close = AsyncMock()
    mock_bot.get_market_data_async = AsyncMock(return_value={'price': 50000.0})
    mock_bot.get_positions_async = AsyncMock(return_value=[])
    mock_bot.get_open_orders_async = AsyncMock(return_value=[])
    
    # Mock config values
    with patch('strategy_runner.TRADING_MODE', 'PAPER'), \
         patch('strategy_runner.MAX_DAILY_LOSS', 500.0), \
         patch('strategy_runner.MAX_DAILY_LOSS_PERCENT', 3.0), \
         patch('strategy_runner.GeminiTrader', return_value=mock_bot):
        
        runner = StrategyRunner(execute_orders=False)
        runner.bot = mock_bot
        runner.risk_manager = MagicMock(spec=RiskManager)
        runner.risk_manager.start_of_day_equity = 100000.0
        runner.daily_loss_pct = 3.0 # Set this manually as initialize is mocked
        
        # Scenario 1: Daily loss > MAX_DAILY_LOSS but < MAX_DAILY_LOSS_PERCENT
        # 500 < 600 < 3000 (3% of 100k)
        runner.risk_manager.daily_loss = 600.0
        
        runner.running = True
        runner.initialize = AsyncMock()
        runner.db = MagicMock()
        runner.db.get_pending_commands.return_value = []
        runner.risk_manager.update_equity = MagicMock()
        
        # We need to break the loop after one iteration
        # We'll use a side effect on asyncio.sleep to raise an exception
        async def break_loop(*args, **kwargs):
            runner.running = False
            return
            
        # We'll patch asyncio.sleep inside strategy_runner
        with patch('strategy_runner.asyncio.sleep', side_effect=break_loop):
             try:
                await runner.run_loop()
             except Exception:
                pass
            
        assert runner._kill_switch is False, "Bot should NOT stop for absolute daily loss in PAPER mode"
        
        # Scenario 2: Daily loss > MAX_DAILY_LOSS_PERCENT
        # 3100 > 3000
        runner.risk_manager.daily_loss = 3100.0
        runner.running = True
        runner._kill_switch = False
        
        with patch('strategy_runner.asyncio.sleep', side_effect=break_loop):
             try:
                await runner.run_loop()
             except Exception:
                pass
            
        assert runner._kill_switch is True, "Bot SHOULD stop for percentage daily loss in PAPER mode"

if __name__ == "__main__":
    asyncio.run(test_sandbox_daily_loss_check())
