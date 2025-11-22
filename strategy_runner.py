import asyncio
import logging
import os
import google.generativeai as genai
from ib_trader import IBTrader
from gemini_trader import GeminiTrader
from risk_manager import RiskManager
from config import GEMINI_API_KEY, TRADING_MODE, ACTIVE_EXCHANGE

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
        self.ib_bot = IBTrader()
        self.gemini_bot = GeminiTrader()
        # Risk manager needs to handle multiple bots or we wrap them?
        # For now, let's pass the ib_bot as primary, but we might need to refactor RiskManager
        self.risk_manager = RiskManager(self.ib_bot) 
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
        """Connects and initializes the bots."""
        logger.info(f"Initializing Strategy Runner in {TRADING_MODE} mode (Active: {ACTIVE_EXCHANGE})...")
        
        # Connect based on config
        if ACTIVE_EXCHANGE in ['IB', 'ALL']:
            await self.ib_bot.connect_async()
        
        if ACTIVE_EXCHANGE in ['GEMINI', 'ALL']:
            await self.gemini_bot.connect_async()
        
        # Initial PnL sync
        total_pnl = 0.0
        if ACTIVE_EXCHANGE in ['IB', 'ALL']:
            ib_pnl = await self.ib_bot.get_pnl_async()
            total_pnl += ib_pnl
            logger.info(f"IB PnL: {ib_pnl}")

        if ACTIVE_EXCHANGE in ['GEMINI', 'ALL']:
            gemini_pnl = await self.gemini_bot.get_pnl_async()
            total_pnl += gemini_pnl
            logger.info(f"Gemini PnL: {gemini_pnl}")
        
        self.risk_manager.update_pnl(total_pnl)
        logger.info(f"Total Initial PnL: {total_pnl}")

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
                total_pnl = 0.0
                if ACTIVE_EXCHANGE in ['IB', 'ALL']:
                    total_pnl += await self.ib_bot.get_pnl_async()
                if ACTIVE_EXCHANGE in ['GEMINI', 'ALL']:
                    total_pnl += await self.gemini_bot.get_pnl_async()
                
                self.risk_manager.update_pnl(total_pnl)
                
                if self.risk_manager.daily_loss > 50: # Hardcoded check from config
                    logger.error("Max daily loss exceeded. Stopping loop.")
                    break

                # 2. Fetch Data
                market_data = {}
                
                if ACTIVE_EXCHANGE in ['IB', 'ALL']:
                    ib_data = await self.ib_bot.get_market_data_async('BHP')
                    market_data['BHP'] = ib_data

                if ACTIVE_EXCHANGE in ['GEMINI', 'ALL']:
                    gemini_data = await self.gemini_bot.get_market_data_async('BTC/USD')
                    market_data['BTC/USD'] = gemini_data

                # 3. Get Decision
                decision_json = await self.get_llm_decision(market_data, total_pnl)
                
                # 4. Execute
                if decision_json:
                    # Parse for logging
                    import json as json_lib
                    try:
                        d = json_lib.loads(decision_json)
                        action = d.get('action')
                        reason = d.get('reason')
                        symbol = d.get('symbol', 'BHP') # Default to BHP if not specified
                        quantity = d.get('quantity', 0)
                    except:
                        action = "ERROR"
                        reason = "Failed to parse"
                        symbol = "UNKNOWN"
                        quantity = 0

                    self.log_decision(total_pnl, market_data, decision_json, action, reason)
                    
                    if action in ['BUY', 'SELL'] and quantity > 0:
                        # Route to correct bot
                        if symbol == 'BHP':
                             # Risk Check needs price
                             price = ib_data['price'] if ib_data else 0
                             if self.risk_manager.check_trade_allowed(symbol, action, quantity, price):
                                 await self.ib_bot.place_order_async(symbol, action, quantity)
                        elif symbol == 'BTC/USD':
                             price = gemini_data['price'] if gemini_data else 0
                             if self.risk_manager.check_trade_allowed(symbol, action, quantity, price):
                                 await self.gemini_bot.place_order_async(symbol, action, quantity)
                        else:
                            logger.warning(f"Unknown symbol: {symbol}")
                
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
