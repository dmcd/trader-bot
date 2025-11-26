import asyncio
import logging
import json
import os
from unittest.mock import MagicMock, AsyncMock
from strategy import LLMStrategy
from logger_config import setup_logging

# Setup logging to file
setup_logging()

async def main():
    # Mock dependencies
    mock_db = MagicMock()
    mock_ta = MagicMock()
    mock_cost_tracker = MagicMock()
    
    # Initialize strategy
    strategy = LLMStrategy(mock_db, mock_ta, mock_cost_tracker)
    
    # Mock LLM generation to avoid actual API call
    strategy.model.generate_content = MagicMock(return_value=MagicMock(text='{"action": "HOLD", "symbol": "BTC/USD", "quantity": 0, "reason": "Test"}'))
    
    # Mock market data
    market_data = {
        'BTC/USD': {
            'price': 50000.0,
            'bid': 49990.0,
            'ask': 50010.0,
            'spread_pct': 0.04,
            'volume': 100.0
        }
    }
    
    print("Generating signal...")
    await strategy.generate_signal(
        session_id=123,
        market_data=market_data,
        current_equity=10000.0,
        current_exposure=0.0
    )
    
    print("Checking telemetry.log...")
    if os.path.exists('telemetry.log'):
        with open('telemetry.log', 'r') as f:
            lines = f.readlines()
            found_prompt = False
            for line in lines:
                try:
                    data = json.loads(line)
                    if data.get('type') == 'llm_prompt':
                        print("Found prompt log!")
                        print(f"Timestamp: {data.get('timestamp')}")
                        print(f"Prompt length: {len(data.get('prompt'))}")
                        found_prompt = True
                        break
                except json.JSONDecodeError:
                    pass
            
            if not found_prompt:
                print("ERROR: LLM prompt log not found in telemetry.log")
    else:
        print("ERROR: telemetry.log not found")

if __name__ == "__main__":
    asyncio.run(main())
