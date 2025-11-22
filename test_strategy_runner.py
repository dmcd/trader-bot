import asyncio
import logging
from trader import TraderBot
from risk_manager import RiskManager
from config import TRADING_MODE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockStrategyRunner:
    def __init__(self):
        self.bot = TraderBot()
        self.risk_manager = RiskManager(self.bot)
        self.running = False

    async def initialize(self):
        """Connects and initializes the bot."""
        logger.info(f"Initializing Mock Strategy Runner in {TRADING_MODE} mode...")
        await self.bot.connect_async()
        
        # Initial PnL sync
        pnl = await self.bot.get_pnl_async()
        self.risk_manager.update_pnl(pnl)
        logger.info(f"Initial PnL (Net Liquidation): {pnl} AUD")

    async def get_llm_decision(self, market_data, current_pnl):
        """Mocks Gemini response."""
        logger.info("Mocking LLM decision...")
        # Simulate a decision to BUY if price is valid, else HOLD
        if str(market_data['price']) != 'nan':
             return '{"action": "BUY", "quantity": 1, "reason": "Mock decision: Price is valid"}'
        return '{"action": "HOLD", "quantity": 0, "reason": "Mock decision: Price is NaN"}'

    async def execute_decision(self, decision_json, price):
        """Parses and executes the decision."""
        import json
        try:
            decision = json.loads(decision_json)
            action = decision.get('action')
            quantity = decision.get('quantity', 0)
            reason = decision.get('reason')
            
            logger.info(f"LLM Decision: {action} {quantity} - {reason}")
            
            if action in ['BUY', 'SELL'] and quantity > 0:
                # Risk Check
                # Mock price if NaN for risk check
                check_price = price if str(price) != 'nan' else 10.0
                
                if self.risk_manager.check_trade_allowed('BHP', action, quantity, check_price):
                    logger.info(f"Executing {action} {quantity} BHP...")
                    # In mock mode, we might not want to actually place order if market data is NaN
                    # But let's try to place it to see what happens (it will fail or be rejected by IB if invalid)
                    if str(price) != 'nan':
                        result = await self.bot.place_order_async('BHP', action, quantity)
                        logger.info(f"Order Result: {result}")
                    else:
                        logger.info("Skipping actual order placement because price is NaN")
                else:
                    logger.warning("Trade blocked by Risk Manager.")
            elif action == 'HOLD':
                logger.info("Holding position.")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {decision_json}")
        except Exception as e:
            logger.error(f"Execution Error: {e}")

    async def run_loop(self):
        """Main autonomous loop."""
        await self.initialize()
        self.running = True
        
        # Run for 2 iterations only
        for _ in range(2):
            try:
                # 1. Update PnL
                pnl = await self.bot.get_pnl_async()
                self.risk_manager.update_pnl(pnl)
                
                # 2. Fetch Data
                market_data = await self.bot.get_market_data_async('BHP')
                price = market_data['price']

                # 3. Get Decision
                decision_json = await self.get_llm_decision(market_data, pnl)
                
                # 4. Execute
                if decision_json:
                    await self.execute_decision(decision_json, price)
                
                # 5. Sleep
                logger.info("Sleeping for 2 seconds...")
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Loop Error: {e}")
        
        self.bot.disconnect()

async def main():
    print("Starting main...")
    runner = MockStrategyRunner()
    print("Runner created. Running loop...")
    await runner.run_loop()
    print("Loop finished.")

if __name__ == "__main__":
    print("Script started.")
    asyncio.run(main())
