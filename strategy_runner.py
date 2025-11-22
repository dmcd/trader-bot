import asyncio
import logging
import os
import google.generativeai as genai
from trader import TraderBot
from risk_manager import RiskManager
from config import GEMINI_API_KEY, TRADING_MODE

import json
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found. The bot will not be able to make decisions.")

class StrategyRunner:
    def __init__(self):
        self.bot = TraderBot()
        self.risk_manager = RiskManager(self.bot)
        self.model = genai.GenerativeModel('gemini-pro')
        self.running = False
        self.history_file = "trade_history.jsonl"

    def log_decision(self, pnl, market_data, decision_json, action, reason):
        """Logs the decision to a JSONL file."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pnl": pnl,
            "market_data": market_data,
            "decision_raw": decision_json,
            "action": action,
            "reason": reason
        }
        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    async def initialize(self):
        """Connects and initializes the bot."""
        logger.info(f"Initializing Strategy Runner in {TRADING_MODE} mode...")
        await self.bot.connect_async()
        
        # Initial PnL sync
        pnl = await self.bot.get_pnl_async()
        self.risk_manager.update_pnl(pnl)
        logger.info(f"Initial PnL (Net Liquidation): {pnl} AUD")

    async def get_llm_decision(self, market_data, current_pnl):
        """Asks Gemini for a trading decision."""
        if not GEMINI_API_KEY:
            return None

        prompt = f"""
        You are an autonomous trading bot. Your goal is to cover token costs by making small, profitable trades.
        
        Current Status:
        - PnL (Net Liquidation): {current_pnl} AUD
        - Market Data (BHP): {market_data}
        
        Risk Constraints:
        - Max Order Value: $100 AUD
        - Max Daily Loss: $50 AUD
        
        Decide on an action for BHP.
        Return ONLY a JSON object with the following format:
        {{
            "action": "BUY" | "SELL" | "HOLD",
            "quantity": <integer>,
            "reason": "<short explanation>"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Simple cleanup to ensure JSON
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3]
            return text
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None

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
                if self.risk_manager.check_trade_allowed('BHP', action, quantity, price):
                    logger.info(f"Executing {action} {quantity} BHP...")
                    result = await self.bot.place_order_async('BHP', action, quantity)
                    logger.info(f"Order Result: {result}")
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
        
        while self.running:
            try:
                # 1. Update PnL
                pnl = await self.bot.get_pnl_async()
                self.risk_manager.update_pnl(pnl)
                
                if self.risk_manager.daily_loss > 50: # Hardcoded check from config
                    logger.error("Max daily loss exceeded. Stopping loop.")
                    break

                # 2. Fetch Data
                market_data = await self.bot.get_market_data_async('BHP')
                if not market_data or str(market_data['price']) == 'nan':
                    logger.warning("Market data unavailable (NaN). Waiting...")
                    await asyncio.sleep(5)
                    continue

                price = market_data['price']

                # 3. Get Decision
                decision_json = await self.get_llm_decision(market_data, pnl)
                
                # 4. Execute
                if decision_json:
                    # Parse for logging
                    import json as json_lib # Avoid conflict with local var if any
                    try:
                        d = json_lib.loads(decision_json)
                        action = d.get('action')
                        reason = d.get('reason')
                    except:
                        action = "ERROR"
                        reason = "Failed to parse"

                    self.log_decision(pnl, market_data, decision_json, action, reason)
                    await self.execute_decision(decision_json, price)
                
                # 5. Sleep
                logger.info("Sleeping for 10 seconds...")
                await asyncio.sleep(10)

            except KeyboardInterrupt:
                logger.info("Stopping loop...")
                self.running = False
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                await asyncio.sleep(5)

async def main():
    runner = StrategyRunner()
    await runner.run_loop()

if __name__ == "__main__":
    asyncio.run(main())
