import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from gemini_trader import GeminiTrader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    trader = GeminiTrader()
    await trader.connect_async()

    if not trader.sandbox:
        logger.warning("Not in sandbox mode! Be careful.")
        # In this specific task, we only care about sandbox, but the script should be safe.
    
    logger.info("Fetching positions...")
    positions = await trader.get_positions_async()
    
    logger.info(f"Found {len(positions)} positions.")
    
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    logger.info(f"Cutoff time: {cutoff}")

    for pos in positions:
        symbol = pos['symbol']
        quantity = pos['quantity']
        logger.info(f"Checking {symbol}: {quantity}")
        
        if symbol == 'USD':
            continue

        # Fetch last trade
        try:
            trades = await trader.get_my_trades_async(symbol, limit=1)
            if trades:
                last_trade = trades[0]
                timestamp = last_trade['timestamp']
                trade_time = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                
                logger.info(f"  Last trade time: {trade_time}")
                
                if trade_time < cutoff:
                    logger.info(f"  -> OLD POSITION (Ignore)")
                else:
                    logger.info(f"  -> ACTIVE POSITION (Keep)")
            else:
                logger.info(f"  No trades found. (Ignore?)")
        except Exception as e:
            logger.error(f"  Error fetching trades for {symbol}: {e}")

    await trader.close()

if __name__ == "__main__":
    asyncio.run(main())
