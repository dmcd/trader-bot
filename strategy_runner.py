import asyncio
import logging
import os
import signal
import sys
import google.generativeai as genai
from ib_trader import IBTrader
from gemini_trader import GeminiTrader
from risk_manager import RiskManager
from config import GEMINI_API_KEY, TRADING_MODE, ACTIVE_EXCHANGE, MAX_DAILY_LOSS_PERCENT, MAX_ORDER_VALUE, MAX_DAILY_LOSS

import json
import datetime

from logger_config import setup_logging

# Configure logging
bot_actions_logger = setup_logging()
logger = logging.getLogger(__name__)

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found. The bot will not be able to make decisions.")

class StrategyRunner:
    def __init__(self):
        # Only instantiate the bot for the active exchange
        if ACTIVE_EXCHANGE == 'IB':
            self.bot = IBTrader()
            self.exchange_name = 'IB'
        elif ACTIVE_EXCHANGE == 'GEMINI':
            self.bot = GeminiTrader()
            self.exchange_name = 'GEMINI'
        else:
            raise ValueError(f"Invalid ACTIVE_EXCHANGE: {ACTIVE_EXCHANGE}. Must be 'IB' or 'GEMINI'")
        
        self.risk_manager = RiskManager(self.bot)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
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
        logger.info(f"Initializing Strategy Runner in {TRADING_MODE} mode (Exchange: {self.exchange_name})...")
        bot_actions_logger.info(f"🤖 Trading Bot Started - Mode: {TRADING_MODE}, Exchange: {self.exchange_name}")
        
        # Connect to the active exchange
        await self.bot.connect_async()
        
        # Get initial PnL
        initial_pnl = await self.bot.get_pnl_async()
        logger.info(f"{self.exchange_name} PnL: {initial_pnl}")
        
        self.risk_manager.update_pnl(initial_pnl)
        bot_actions_logger.info(f"💰 Starting Portfolio Value: ${initial_pnl:,.2f}")

    async def get_llm_decision(self, market_data, current_pnl):
        """Asks Gemini for a trading decision."""
        if not GEMINI_API_KEY:
            return None

        # Determine what symbols are available and build appropriate prompt
        available_symbols = list(market_data.keys())
        
        if not available_symbols:
            logger.warning("No market data available for decision making")
            return None
        
        # Build market data summary
        market_summary = ""
        for symbol, data in market_data.items():
            if data:
                market_summary += f"\n  - {symbol}: Price ${data.get('price', 'N/A')}, Bid ${data.get('bid', 'N/A')}, Ask ${data.get('ask', 'N/A')}"
        
        # Determine if we're trading crypto or stocks
        is_crypto = ACTIVE_EXCHANGE == 'GEMINI' and any('/' in symbol for symbol in available_symbols)
        
        if is_crypto:
            # Crypto trading prompt (supports fractional quantities)
            prompt = f"""
You are an autonomous crypto trading bot. Your goal is to make small, profitable trades.

Current Status:
- Portfolio Value: ${current_pnl:,.2f} USD
- Market Data:{market_summary}

Risk Constraints:
- Max Order Value: ${MAX_ORDER_VALUE:.2f} USD
- Max Daily Loss: {MAX_DAILY_LOSS_PERCENT}% of portfolio

Available Symbols: {', '.join(available_symbols)}

For crypto trading, you can use FRACTIONAL quantities (e.g., 0.001 BTC).
Calculate the appropriate fractional amount to stay within the ${MAX_ORDER_VALUE:.2f} max order value.

Decide on an action for one of the available symbols.
Return ONLY a JSON object with the following format:
{{
    "action": "BUY" | "SELL" | "HOLD",
    "symbol": "<symbol from available symbols>",
    "quantity": <float (fractional amounts allowed)>,
    "reason": "<short explanation>"
}}

Example: {{"action": "BUY", "symbol": "BTC/USD", "quantity": 0.0012, "reason": "Price dip, good entry"}}
"""
        else:
            # Stock trading prompt (integer quantities only)
            prompt = f"""
You are an autonomous stock trading bot. Your goal is to make small, profitable trades.

Current Status:
- Portfolio Value: ${current_pnl:,.2f} AUD
- Market Data:{market_summary}

Risk Constraints:
- Max Order Value: ${MAX_ORDER_VALUE:.2f} AUD
- Max Daily Loss: ${MAX_DAILY_LOSS:.2f} AUD

Available Symbols: {', '.join(available_symbols)}

For stock trading, quantities must be WHOLE NUMBERS (integers).

Decide on an action for one of the available symbols.
Return ONLY a JSON object with the following format:
{{
    "action": "BUY" | "SELL" | "HOLD",
    "symbol": "<symbol from available symbols>",
    "quantity": <integer>,
    "reason": "<short explanation>"
}}

Example: {{"action": "BUY", "symbol": "BHP", "quantity": 2, "reason": "Upward trend detected"}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            # Simple cleanup to ensure JSON
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3]
            elif text.startswith('```'):
                text = text[3:-3]
            return text
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None
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
            
            # Log to user-friendly bot.log
            if action == 'HOLD':
                bot_actions_logger.info(f"📊 Decision: HOLD - {reason}")
            elif action in ['BUY', 'SELL']:
                bot_actions_logger.info(f"📊 Decision: {action} {quantity} units - {reason}")
            
            if action in ['BUY', 'SELL'] and quantity > 0:
                # Risk Check
                if self.risk_manager.check_trade_allowed('BHP', action, quantity, price):
                    logger.info(f"Executing {action} {quantity} BHP...")
                    bot_actions_logger.info(f"✅ Executing: {action} {quantity} units at ${price:.2f}")
                    result = await self.bot.place_order_async('BHP', action, quantity)
                    logger.info(f"Order Result: {result}")
                    if result:
                        bot_actions_logger.info(f"✅ Order Completed: {result}")
                else:
                    logger.warning("Trade blocked by Risk Manager.")
                    bot_actions_logger.info(f"⛔ Trade Blocked: Risk limits exceeded")
            elif action == 'HOLD':
                logger.info("Holding position.")
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {decision_json}")
            bot_actions_logger.info(f"⚠️ Error: Could not parse trading decision")
        except Exception as e:
            logger.error(f"Execution Error: {e}")
            bot_actions_logger.info(f"⚠️ Error during execution: {str(e)}")

    async def cleanup(self):
        """Cleanup and close connection."""
        logger.info("Cleaning up connections...")
        try:
            await self.bot.close()
            logger.info(f"{self.exchange_name} connection closed")
        except Exception as e:
            logger.error(f"Error closing {self.exchange_name} connection: {e}")
        
        logger.info("Cleanup complete.")

    async def run_loop(self):
        """Main autonomous loop."""
        try:
            await self.initialize()
            self.running = True
            
            while self.running:
                try:
                    # 1. Update PnL
                    current_pnl = await self.bot.get_pnl_async()
                    self.risk_manager.update_pnl(current_pnl)
                    
                    # Check percentage-based daily loss
                    if self.risk_manager.start_of_day_equity and self.risk_manager.start_of_day_equity > 0:
                        loss_percent = (self.risk_manager.daily_loss / self.risk_manager.start_of_day_equity) * 100
                        if loss_percent > MAX_DAILY_LOSS_PERCENT:
                            logger.error(f"Max daily loss exceeded: {loss_percent:.2f}% > {MAX_DAILY_LOSS_PERCENT}%. Stopping loop.")
                            bot_actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded ({loss_percent:.2f}%)")
                            break

                    # 2. Fetch Market Data
                    market_data = {}
                    
                    # Determine which symbol to fetch based on exchange
                    if ACTIVE_EXCHANGE == 'IB':
                        symbol = 'BHP'
                    else:  # GEMINI
                        symbol = 'BTC/USD'
                    
                    data = await self.bot.get_market_data_async(symbol)
                    market_data[symbol] = data

                    # 3. Get Decision
                    decision_json = await self.get_llm_decision(market_data, current_pnl)
                    
                    # 4. Execute
                    if decision_json:
                        # Parse for logging
                        import json as json_lib
                        try:
                            d = json_lib.loads(decision_json)
                            action = d.get('action')
                            reason = d.get('reason')
                            symbol = d.get('symbol', symbol)  # Use symbol from decision or default
                            quantity = d.get('quantity', 0)
                        except:
                            action = "ERROR"
                            reason = "Failed to parse"
                            symbol = "UNKNOWN"
                            quantity = 0

                        self.log_decision(current_pnl, market_data, decision_json, action, reason)
                    
                        # Log decision to user-friendly log
                        if action == 'HOLD':
                            bot_actions_logger.info(f"📊 Decision: HOLD - {reason}")
                        elif action in ['BUY', 'SELL']:
                            # Format quantity appropriately (show more decimals for small amounts)
                            if quantity < 1:
                                qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                            else:
                                qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                            bot_actions_logger.info(f"📊 Decision: {action} {qty_str} {symbol} - {reason}")
                        
                        if action in ['BUY', 'SELL'] and quantity > 0:
                            # Get price for risk check
                            price = data['price'] if data else 0
                            
                            if self.risk_manager.check_trade_allowed(symbol, action, quantity, price):
                                # Format fractional crypto amounts nicely
                                if quantity < 1:
                                    qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                                else:
                                    qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                                bot_actions_logger.info(f"✅ Executing: {action} {qty_str} {symbol} at ${price:,.2f}")
                                await self.bot.place_order_async(symbol, action, quantity)
                            else:
                                bot_actions_logger.info(f"⛔ Trade Blocked: Risk limits exceeded")
                    
                    # 5. Sleep
                    logger.info("Sleeping for 10 seconds...")
                    await asyncio.sleep(10)

                except KeyboardInterrupt:
                    logger.info("Stopping loop...")
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Loop Error: {e}")
                    await asyncio.sleep(5)
        finally:
            # Always cleanup, even if there's an exception or break
            await self.cleanup()

async def main():
    runner = StrategyRunner()
    
    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("Received shutdown signal, stopping bot...")
        bot_actions_logger.info("🛑 Bot shutting down...")
        runner.running = False
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await runner.run_loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Bot stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        sys.exit(0)
