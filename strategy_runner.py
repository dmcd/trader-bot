import asyncio
import logging
import os
import signal
import sys
import google.generativeai as genai
from ib_trader import IBTrader
from gemini_trader import GeminiTrader
from risk_manager import RiskManager
from database import TradingDatabase
from cost_tracker import CostTracker
from trading_context import TradingContext
from technical_analysis import TechnicalAnalysis
from config import GEMINI_API_KEY, TRADING_MODE, ACTIVE_EXCHANGE, MAX_DAILY_LOSS_PERCENT, MAX_ORDER_VALUE, MAX_DAILY_LOSS



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
        # self.history_file = "trade_history.jsonl"  # Deprecated: using DB instead
        
        # Professional trading infrastructure
        self.db = TradingDatabase()
        self.cost_tracker = CostTracker(self.exchange_name)
        self.technical_analysis = TechnicalAnalysis()
        self.session_id = None
        self.context = None
        self.session = None

    async def initialize(self):
        """Connects and initializes the bot."""
        logger.info(f"Initializing Strategy Runner in {TRADING_MODE} mode (Exchange: {self.exchange_name})...")
        bot_actions_logger.info(f"🤖 Trading Bot Started - Mode: {TRADING_MODE}, Exchange: {self.exchange_name}")
        
        # Connect to the active exchange
        await self.bot.connect_async()
        
        # Get initial PnL
        initial_pnl = await self.bot.get_pnl_async()
        logger.info(f"{self.exchange_name} PnL: {initial_pnl}")
        
        # Create or load today's trading session
        self.session_id = self.db.get_or_create_session(starting_balance=initial_pnl)
        self.session = self.db.get_session(self.session_id)
        
        # Initialize trading context
        self.context = TradingContext(self.db, self.session_id)
        
        # Seed risk manager with persisted start-of-day equity to survive restarts
        start_equity = None
        if self.session and self.session.get('starting_balance') is not None:
            start_equity = self.session.get('starting_balance')
        else:
            start_equity = initial_pnl
        self.risk_manager.seed_start_of_day(start_equity)
        
        # Reconcile with exchange state
        await self.reconcile_exchange_state()

        self.risk_manager.update_pnl(initial_pnl)
        bot_actions_logger.info(f"💰 Starting Portfolio Value: ${initial_pnl:,.2f}")

    async def reconcile_exchange_state(self):
        """Pull positions/open orders from exchange and compare to local session."""
        try:
            exchange_positions = await self.bot.get_positions_async()
            exchange_orders = await self.bot.get_open_orders_async()
        except Exception as e:
            logger.error(f"Reconciliation failed: unable to fetch exchange state: {e}")
            return

        stored_positions = self.db.get_positions(self.session_id)
        stored_orders = self.db.get_open_orders(self.session_id)

        # If no stored positions yet, treat exchange as source of truth and seed DB without warnings.
        if not stored_positions:
            logger.info("No stored position snapshot for today; seeding from exchange.")
            self.db.replace_positions(self.session_id, exchange_positions)
            self.db.replace_open_orders(self.session_id, exchange_orders)
            if exchange_positions:
                logger.info(f"Seeded positions: {len(exchange_positions)} symbols")
            if exchange_orders:
                logger.info(f"Seeded open orders: {len(exchange_orders)}")
            return

        # Compare positions by symbol and quantity
        def to_map(items):
            return {i['symbol']: i for i in items if i.get('symbol')}

        exchange_pos_map = to_map(exchange_positions)
        stored_pos_map = to_map(stored_positions)

        mismatches = []
        if stored_positions:
            for sym, pos in exchange_pos_map.items():
                qty = pos.get('quantity', 0)
                stored_qty = stored_pos_map.get(sym, {}).get('quantity', 0)
                if abs(qty - stored_qty) > 1e-8:
                    mismatches.append((sym, stored_qty, qty))
            for sym, pos in stored_pos_map.items():
                if sym not in exchange_pos_map:
                    mismatches.append((sym, pos.get('quantity', 0), 0))

        if mismatches:
            for sym, local_qty, exch_qty in mismatches:
                logger.warning(f"Position mismatch for {sym}: DB={local_qty} Exchange={exch_qty}")
        elif stored_positions:
            logger.info("Positions match exchange snapshot.")
        else:
            logger.info("No prior position snapshot; seeding from exchange.")

        # Persist latest snapshots
        self.db.replace_positions(self.session_id, exchange_positions)
        self.db.replace_open_orders(self.session_id, exchange_orders)

        if exchange_orders:
            logger.info(f"Open orders on exchange: {len(exchange_orders)}")
        else:
            logger.info("No open orders on exchange.")

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
        
        # Get trading context summary and technical indicators
        symbol = available_symbols[0] if available_symbols else None
        context_summary = ""
        indicator_summary = ""
        
        if symbol and self.context:
            try:
                # Get session context
                context_summary = self.context.get_context_summary(symbol)
                
                # Calculate technical indicators from recent market data
                recent_data = self.db.get_recent_market_data(self.session_id, symbol, limit=50)
                if recent_data and len(recent_data) >= 20:
                    indicators = self.technical_analysis.calculate_indicators(recent_data)
                    if indicators:
                        current_price = market_data[symbol]['price']
                        indicator_summary = self.technical_analysis.format_indicators_for_llm(indicators, current_price)
            except Exception as e:
                logger.warning(f"Error getting context/indicators: {e}")
        
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
"""
            # Add context and indicators if available
            if context_summary:
                prompt += f"\n{context_summary}\n"
            if indicator_summary:
                prompt += f"\n{indicator_summary}\n"
            
            prompt += """
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
"""
            # Add context and indicators if available
            if context_summary:
                prompt += f"\n{context_summary}\n"
            if indicator_summary:
                prompt += f"\n{indicator_summary}\n"
            
            prompt += """
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
            
            # Track token usage and cost
            if hasattr(response, 'usage_metadata') and self.session_id:
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    cost = self.cost_tracker.calculate_llm_cost(input_tokens, output_tokens)
                    
                    # Log to database
                    self.db.log_llm_call(self.session_id, input_tokens, output_tokens, cost, response.text[:500])
                    logger.debug(f"LLM call: {input_tokens + output_tokens} tokens, ${cost:.6f}")
                except Exception as e:
                    logger.warning(f"Error tracking LLM usage: {e}")
            
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
        
        # Save final session statistics
        if self.session_id:
            try:
                # Get final PnL
                final_pnl = await self.bot.get_pnl_async()
                
                # Get session stats
                session_stats = self.db.get_session_stats(self.session_id)
                
                # Calculate net PnL
                gross_pnl = final_pnl - session_stats['starting_balance']
                net_pnl = self.cost_tracker.calculate_net_pnl(
                    gross_pnl,
                    session_stats['total_fees'],
                    session_stats['total_llm_cost']
                )
                
                # Update database
                self.db.update_session_balance(self.session_id, final_pnl, net_pnl)
                
                # Log summary to bot.log
                bot_actions_logger.info("=" * 50)
                bot_actions_logger.info("📊 SESSION SUMMARY")
                bot_actions_logger.info("=" * 50)
                bot_actions_logger.info(f"Total Trades: {session_stats['total_trades']}")
                bot_actions_logger.info(f"Gross PnL: ${gross_pnl:,.2f}")
                bot_actions_logger.info(f"Trading Fees: ${session_stats['total_fees']:.2f}")
                bot_actions_logger.info(f"LLM Costs: ${session_stats['total_llm_cost']:.4f}")
                bot_actions_logger.info(f"Net PnL: ${net_pnl:,.2f}")
                
                if net_pnl > 0:
                    bot_actions_logger.info(f"✅ Profitable session!")
                else:
                    bot_actions_logger.info(f"❌ Unprofitable session")
                bot_actions_logger.info("=" * 50)
                
            except Exception as e:
                logger.error(f"Error saving session stats: {e}")
        
        # Close bot connection
        try:
            await self.bot.close()
            logger.info(f"{self.exchange_name} connection closed")
        except Exception as e:
            logger.error(f"Error closing {self.exchange_name} connection: {e}")
        
        # Close database
        try:
            if self.db:
                self.db.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
        
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
                    
                    # Log market data to database
                    if data and self.session_id:
                        try:
                            self.db.log_market_data(
                                self.session_id,
                                symbol,
                                data.get('price', 0),
                                data.get('bid', 0),
                                data.get('ask', 0),
                                data.get('volume', 0)
                            )
                        except Exception as e:
                            logger.warning(f"Error logging market data: {e}")

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
                                # Calculate fee before execution
                                fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, action)
                                
                                # Format fractional crypto amounts nicely
                                if quantity < 1:
                                    qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                                else:
                                    qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                                
                                bot_actions_logger.info(f"✅ Executing: {action} {qty_str} {symbol} at ${price:,.2f} (fee: ${fee:.4f})")
                                
                                # Execute trade
                                order_result = await self.bot.place_order_async(symbol, action, quantity)
                                
                                # Log trade to database
                                if self.session_id and order_result:
                                    try:
                                        self.db.log_trade(
                                            self.session_id,
                                            symbol,
                                            action,
                                            quantity,
                                            price,
                                            fee,
                                            reason
                                        )
                                    except Exception as e:
                                        logger.warning(f"Error logging trade: {e}")
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
