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
from config import (
    GEMINI_API_KEY,
    TRADING_MODE,
    ACTIVE_EXCHANGE,
    MAX_DAILY_LOSS_PERCENT,
    MAX_ORDER_VALUE,
    MAX_DAILY_LOSS,
    LOOP_INTERVAL_SECONDS,
    MIN_TRADE_INTERVAL_SECONDS,
    FEE_RATIO_COOLDOWN,
    SIZE_TIER,
    ORDER_SIZE_BY_TIER,
    DAILY_LOSS_PCT_BY_TIER,
    PRIORITY_MOVE_PCT,
    PRIORITY_LOOKBACK_MIN,
    BREAK_GLASS_COOLDOWN_MIN,
    BREAK_GLASS_SIZE_FACTOR,
)



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
        
        # Professional trading infrastructure
        self.db = TradingDatabase()
        self.cost_tracker = CostTracker(self.exchange_name)
        self.technical_analysis = TechnicalAnalysis()
        self.session_id = None
        self.context = None
        self.session = None
        self.last_rejection_reason = None
        self.last_trade_ts = None
        self.sandbox_seed_symbols = {'USD', 'BTC/USD', 'BTC', 'ETH/USD', 'LTC/USD', 'ZEC/USD', 'BCH/USD'}
        # Simple in-memory holdings tracker for realized PnL
        self.holdings = {}  # symbol -> {'qty': float, 'avg_cost': float}
        self.size_tier = SIZE_TIER if SIZE_TIER in ORDER_SIZE_BY_TIER else 'MODERATE'
        self._last_reconnect = 0.0
        self._kill_switch = False
        self._last_break_glass = 0.0
        self.size_tier = SIZE_TIER if SIZE_TIER in ORDER_SIZE_BY_TIER else 'MODERATE'

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
        
        # Clear any old pending commands from previous sessions
        self.db.clear_old_commands()
        
        # Initialize trading context
        self.context = TradingContext(self.db, self.session_id)
        
        # Seed risk manager with persisted start-of-day equity to survive restarts
        start_equity = None
        if self.session and self.session.get('starting_balance') is not None:
            start_equity = self.session.get('starting_balance')
        else:
            # Prefer latest broker equity snapshot if available
            latest_equity = self.db.get_latest_equity(self.session_id)
            start_equity = latest_equity if latest_equity is not None else initial_pnl
        self.risk_manager.seed_start_of_day(start_equity)
        
        # Reconcile with exchange state
        await self.reconcile_exchange_state()

        # Rebuild holdings from historical trades to keep realized PnL continuity after restarts
        try:
            self._load_holdings_from_db()
            logger.info(f"Holdings rebuilt from trade history: {self.holdings}")
        except Exception as e:
            logger.warning(f"Could not rebuild holdings: {e}")

        # Apply tier-specific daily loss %
        tier_loss_pct = DAILY_LOSS_PCT_BY_TIER.get(self.size_tier, MAX_DAILY_LOSS_PERCENT)
        if tier_loss_pct != MAX_DAILY_LOSS_PERCENT:
            logger.info(f"Overriding daily loss percent to {tier_loss_pct}% for tier {self.size_tier}")
        self.daily_loss_pct = tier_loss_pct

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
        exchange_positions = self._sanitize_positions(exchange_positions)

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

    def _sanitize_positions(self, positions):
        """Remove sandbox seed balances so exposure/risk isn't polluted."""
        if not positions or TRADING_MODE != 'PAPER':
            return positions

        cleaned = []
        for pos in positions:
            sym = pos.get('symbol')
            qty = pos.get('quantity', 0) or 0
            avg_price = pos.get('avg_price')

            # Heuristic: ignore known sandbox seeds that come with null avg_price
            if sym in self.sandbox_seed_symbols and avg_price in (None, 0) and qty >= 100:
                logger.debug(f"Dropping sandbox seed position {sym} qty {qty}")
                continue
            cleaned.append(pos)
        return cleaned

    def _update_holdings_and_realized(self, symbol: str, action: str, quantity: float, price: float, fee: float) -> float:
        """
        Maintain in-memory holdings to compute realized PnL per trade.
        Realized PnL subtracts fee; buys record negative fee only.
        """
        pos = self.holdings.get(symbol, {'qty': 0.0, 'avg_cost': 0.0})
        qty = pos['qty']
        avg_cost = pos['avg_cost']
        realized = 0.0

        if action == 'BUY':
            new_qty = qty + quantity
            if new_qty > 0:
                new_avg = ((qty * avg_cost) + (quantity * price)) / new_qty
            else:
                new_avg = 0.0
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': new_avg}
            realized = -fee
        else:  # SELL
            realized = (price - avg_cost) * quantity - fee
            new_qty = max(0.0, qty - quantity)
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': avg_cost if new_qty > 0 else 0.0}

        return realized

    def _apply_trade_to_holdings(self, symbol: str, action: str, quantity: float, price: float):
        """Update holdings without computing realized PnL (used for replay)."""
        pos = self.holdings.get(symbol, {'qty': 0.0, 'avg_cost': 0.0})
        qty = pos['qty']
        avg_cost = pos['avg_cost']

        if action == 'BUY':
            new_qty = qty + quantity
            new_avg = ((qty * avg_cost) + (quantity * price)) / new_qty if new_qty > 0 else 0.0
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': new_avg}
        else:
            new_qty = max(0.0, qty - quantity)
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': avg_cost if new_qty > 0 else 0.0}

    def _load_holdings_from_db(self):
        """Rebuild holdings from historical trades for this session."""
        trades = self.db.get_trades_for_session(self.session_id)
        self.holdings = {}
        for t in trades:
            self._apply_trade_to_holdings(
                t['symbol'],
                t['action'],
                t['quantity'],
                t['price']
            )

    async def _close_all_positions_safely(self):
        """Attempt to flatten all positions using market-ish orders."""
        try:
            positions = self.db.get_positions(self.session_id)
            if not positions:
                return
            for pos in positions:
                symbol = pos['symbol']
                quantity = pos['quantity']
                if quantity <= 0:
                    continue
                try:
                    data = await self.bot.get_market_data_async(symbol)
                    price = data.get('price', 0) if data else 0
                    result = await self.bot.place_order_async(symbol, 'SELL', quantity, prefer_maker=False)
                    if result:
                        fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, 'SELL')
                        realized_pnl = self._update_holdings_and_realized(symbol, 'SELL', quantity, price, fee)
                        self.db.log_trade(
                            self.session_id,
                            symbol,
                            'SELL',
                            quantity,
                            price,
                            fee,
                            "Auto-flatten on stop",
                            liquidity=result.get('liquidity', 'taker'),
                            realized_pnl=realized_pnl
                        )
                        bot_actions_logger.info(f"✅ Flattened {quantity} {symbol} @ ${price:,.2f}")
                except Exception as e:
                    logger.error(f"Error flattening {symbol}: {e}")
        except Exception as e:
            logger.error(f"Flatten-all failed: {e}")
            self._kill_switch = True

    async def _reconnect_bot(self):
        """Reconnect the broker client with a cooldown to avoid thrash."""
        now = asyncio.get_event_loop().time()
        if now - self._last_reconnect < 30:
            return
        try:
            await self.bot.connect_async()
            self._last_reconnect = now
            logger.info("Reconnected to broker")
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")

    def _is_choppy(self, symbol: str, market_data_point, recent_data):
        """
        Simple regime filter: skip when Bollinger width is tight and RSI is neutral.
        """
        try:
            if not recent_data or len(recent_data) < 20:
                return False
            indicators = self.technical_analysis.calculate_indicators(recent_data)
            if not indicators:
                return False
            bb_width = indicators.get('bb_width', 0)
            rsi = indicators.get('rsi', 50)
            # Define chop as very tight bands and RSI near 50
            if bb_width is not None and bb_width < 1.0 and abs(rsi - 50) < 5:
                return True
        except Exception as e:
            logger.warning(f"Chop filter error: {e}")
        return False
    def _priority_signal(self, symbol: str) -> bool:
        """
        Detect breakout move to allow a break-glass trade inside cooldown.
        """
        try:
            lookback_points = PRIORITY_LOOKBACK_MIN * 60 // LOOP_INTERVAL_SECONDS + 2
            recent = self.db.get_recent_market_data(self.session_id, symbol, limit=max(lookback_points, 10))
            if not recent or len(recent) < 2:
                return False
            latest = recent[0]['price']
            past = recent[min(len(recent)-1, lookback_points-1)]['price']
            if past and latest:
                move_pct = abs((latest - past) / past) * 100
                if move_pct >= PRIORITY_MOVE_PCT:
                    return True
        except Exception as e:
            logger.warning(f"Priority signal error: {e}")
        return False

    def _fees_too_high(self):
        """
        Check fee ratio vs gross PnL; returns True to pause when fees dominate.
        """
        try:
            stats = self.db.get_session_stats(self.session_id)
            gross = stats.get('gross_pnl', 0) or 0
            fees = stats.get('total_fees', 0) or 0
            if gross == 0:
                return False
            ratio = (fees / abs(gross)) * 100
            return ratio >= FEE_RATIO_COOLDOWN
        except Exception as e:
            logger.warning(f"Fee ratio check failed: {e}")
            return False

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
        
        # Add last rejection reason to prompt if it exists
        if hasattr(self, 'last_rejection_reason') and self.last_rejection_reason:
            prompt += f"\nIMPORTANT: Your previous order was REJECTED by the Risk Manager for the following reason:\n"
            prompt += f"'{self.last_rejection_reason}'\n"
            prompt += "You MUST adjust your strategy to avoid this rejection (e.g., lower quantity, check limits).\n"
            # Clear it after using it once so we don't nag forever if they fix it
            self.last_rejection_reason = None
            
        try:
            # Run blocking LLM call off the event loop with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(self.model.generate_content, prompt),
                timeout=30
            )
            
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
                risk_result = self.risk_manager.check_trade_allowed('BHP', action, quantity, price)
                if risk_result.allowed:
                    logger.info(f"Executing {action} {quantity} BHP...")
                    bot_actions_logger.info(f"✅ Executing: {action} {quantity} units at ${price:.2f}")
                    result = await self.bot.place_order_async('BHP', action, quantity)
                    logger.info(f"Order Result: {result}")
                    if result:
                        bot_actions_logger.info(f"✅ Order Completed: {result}")
                else:
                    logger.warning(f"Trade blocked by Risk Manager: {risk_result.reason}")
                    bot_actions_logger.info(f"⛔ Trade Blocked: {risk_result.reason}")
                    # Store reason for next turn
                    self.last_rejection_reason = risk_result.reason
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
                    # 0. Check for pending commands from dashboard
                    pending_commands = self.db.get_pending_commands()
                    for cmd in pending_commands:
                        command = cmd['command']
                        command_id = cmd['id']
                        
                        if command == 'CLOSE_ALL_POSITIONS':
                            logger.info("Executing command: CLOSE_ALL_POSITIONS")
                            bot_actions_logger.info("🛑 Manual Command: Closing all positions...")
                            
                            # Get current positions
                            positions = self.db.get_positions(self.session_id)
                            for pos in positions:
                                symbol = pos['symbol']
                                quantity = pos['quantity']
                                if quantity > 0:
                                    try:
                                        # Get current price
                                        data = await self.bot.get_market_data_async(symbol)
                                        price = data.get('price', 0) if data else 0
                                        
                                        # Execute sell order
                                        logger.info(f"Closing position: SELL {quantity} {symbol}")
                                        result = await self.bot.place_order_async(symbol, 'SELL', quantity)
                                        
                                        if result:
                                            # Log the trade
                                            fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, 'SELL')
                                            realized_pnl = self._update_holdings_and_realized(symbol, 'SELL', quantity, price, fee)
                                            self.db.log_trade(
                                                self.session_id,
                                                symbol,
                                                'SELL',
                                                quantity,
                                                price,
                                                fee,
                                                "Manual close all positions",
                                                liquidity="taker",
                                                realized_pnl=realized_pnl
                                            )
                                            bot_actions_logger.info(f"✅ Closed: {quantity} {symbol} @ ${price:,.2f}")
                                    except Exception as e:
                                        logger.error(f"Error closing position {symbol}: {e}")
                                        bot_actions_logger.info(f"⚠️ Error closing {symbol}: {str(e)}")
                            
                            bot_actions_logger.info("✅ All positions closed")
                            self.db.mark_command_executed(command_id)
                            
                        elif command == 'STOP_BOT':
                            logger.info("Executing command: STOP_BOT")
                            bot_actions_logger.info("🛑 Manual Command: Stopping bot...")
                            self.db.mark_command_executed(command_id)
                            self.running = False
                            break
                    
                    # 1. Update PnL
                    current_pnl = await self.bot.get_pnl_async()
                    # Log equity snapshot for MTM tracking
                    if self.session_id is not None:
                        try:
                            self.db.log_equity_snapshot(self.session_id, current_pnl)
                        except Exception as e:
                            logger.warning(f"Could not log equity snapshot: {e}")
                    self.risk_manager.update_pnl(current_pnl)
                    
                    # Check percentage-based daily loss (tier-aware)
                    if self.risk_manager.start_of_day_equity and self.risk_manager.start_of_day_equity > 0:
                        loss_percent = (self.risk_manager.daily_loss / self.risk_manager.start_of_day_equity) * 100
                        limit_pct = self.daily_loss_pct
                        if loss_percent > limit_pct:
                            logger.error(f"Max daily loss exceeded: {loss_percent:.2f}% > {limit_pct}%. Stopping loop.")
                            bot_actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded ({loss_percent:.2f}%)")
                            # Attempt to flatten positions before stopping
                            await self._close_all_positions_safely()
                            self._kill_switch = True
                            break
                    # Check absolute daily loss
                    if self.risk_manager.daily_loss > MAX_DAILY_LOSS:
                        logger.error(f"Max daily loss exceeded: ${self.risk_manager.daily_loss:.2f} > ${MAX_DAILY_LOSS:.2f}. Stopping loop.")
                        bot_actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded (${self.risk_manager.daily_loss:.2f})")
                        await self._close_all_positions_safely()
                        self._kill_switch = True
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

                    # Refresh live positions each loop for accurate exposure (sanitized for sandbox seeds)
                    try:
                        live_positions = await self.bot.get_positions_async()
                        live_positions = self._sanitize_positions(live_positions)
                        self.db.replace_positions(self.session_id, live_positions)
                    except Exception as e:
                        logger.warning(f"Could not refresh positions: {e}")

                    # Skip trading if fees are dominating
                    if self._fees_too_high():
                        bot_actions_logger.info(f"⏸ Pausing: fee ratio above {FEE_RATIO_COOLDOWN}%")
                        logger.info("Skipping trading due to high fee ratio")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    # Kill switch check
                    if self._kill_switch:
                        bot_actions_logger.info("🛑 Kill switch active; not trading.")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue
                    
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

                    # 3. Simple chop filter
                    recent_data = self.db.get_recent_market_data(self.session_id, symbol, limit=50)
                    if self._is_choppy(symbol, data, recent_data):
                        bot_actions_logger.info("⏸ Skipping trade: market in chop (tight bands, neutral RSI)")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    # 4. Decide whether to call LLM (respect spacing unless priority signal)
                    now_ts = asyncio.get_event_loop().time()
                    can_trade = not self.last_trade_ts or (now_ts - self.last_trade_ts) >= MIN_TRADE_INTERVAL_SECONDS
                    priority = self._priority_signal(symbol)
                    allow_break_glass = priority and (now_ts - self._last_break_glass) >= (BREAK_GLASS_COOLDOWN_MIN * 60)

                    if not can_trade and not allow_break_glass:
                        bot_actions_logger.info("⏸ Skipping LLM: trade spacing active and no priority signal")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    decision_json = await self.get_llm_decision(market_data, current_pnl)

                    # 5. Execute
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
                            # Enforce minimum spacing between trades to cut churn/fees
                            now_ts = asyncio.get_event_loop().time()
                            if self.last_trade_ts and (now_ts - self.last_trade_ts) < MIN_TRADE_INTERVAL_SECONDS:
                                # Allow break-glass if priority was set and cooldown allows
                                if priority and allow_break_glass:
                                    logger.info("Break-glass trade allowed despite spacing guard")
                                else:
                                    wait_remaining = MIN_TRADE_INTERVAL_SECONDS - (now_ts - self.last_trade_ts)
                                    msg = f"Skipped trade: spacing guard ({wait_remaining:.0f}s remaining)"
                                    bot_actions_logger.info(f"⏸ {msg}")
                                    logger.info(msg)
                                    continue

                            # If price missing, skip for safety
                            if not price:
                                msg = "Skipped trade: missing price data"
                                logger.warning(msg)
                                bot_actions_logger.info(f"⏸ {msg}")
                                continue

                            # Get price for risk check
                            # Prefer the symbol's own market data if available
                            md = market_data.get(symbol)
                            price = md.get('price') if md else (data['price'] if data else 0)

                            # Update RiskManager with current positions for exposure calculation
                            positions_data = self.db.get_positions(self.session_id)
                            positions_dict = {}
                            for pos in positions_data:
                                sym = pos['symbol']
                                # Get current price for this position
                                current_price = pos.get('avg_price') or 0
                                recent_data = self.db.get_recent_market_data(self.session_id, sym, limit=1)
                                if recent_data and recent_data[0].get('price'):
                                    current_price = recent_data[0]['price']
                                
                                # Only add if we have a valid price
                                if current_price:
                                    positions_dict[sym] = {
                                        'quantity': pos['quantity'],
                                        'current_price': current_price
                                    }
                            self.risk_manager.update_positions(positions_dict)
                            
                            risk_result = self.risk_manager.check_trade_allowed(symbol, action, quantity, price)

                            # Auto-reduce if order value is too high or exceed tier cap (reduce further on break-glass)
                            tier_cap = ORDER_SIZE_BY_TIER.get(self.size_tier, MAX_ORDER_VALUE)
                            max_value = min(MAX_ORDER_VALUE, tier_cap)
                            if not can_trade and allow_break_glass:
                                max_value *= BREAK_GLASS_SIZE_FACTOR
                            if (quantity * price) > max_value:
                                old_qty = quantity
                                target_value = max_value * 0.99
                                quantity = target_value / price
                                logger.info(f"⚠️ Auto-reducing trade size from {old_qty:.4f} to {quantity:.4f} to fit cap ${max_value:.2f}")
                                bot_actions_logger.info(f"📉 Auto-Reducing: Trade size too large, adjusting to {quantity:.4f} {symbol}")
                                risk_result = self.risk_manager.check_trade_allowed(symbol, action, quantity, price)

                            if risk_result.allowed:
                                # Calculate fee before execution
                                fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, action)
                                liquidity = "maker_intent"
                                
                                # Format fractional crypto amounts nicely
                                if quantity < 1:
                                    qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                                else:
                                    qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                                
                                bot_actions_logger.info(f"✅ Executing: {action} {qty_str} {symbol} at ${price:,.2f} (fee: ${fee:.4f})")
                                
                                # Execute trade
                                retries = 0
                                order_result = None
                                while retries < 2 and order_result is None:
                                    try:
                                        order_result = await asyncio.wait_for(
                                            self.bot.place_order_async(symbol, action, quantity, prefer_maker=True),
                                            timeout=15
                                        )
                                    except asyncio.TimeoutError:
                                        logger.error("Order placement timed out")
                                        await self._reconnect_bot()
                                        retries += 1
                                    except Exception as e:
                                        logger.error(f"Order placement error: {e}")
                                        await self._reconnect_bot()
                                        retries += 1

                                # Capture reported liquidity if present
                                if order_result and isinstance(order_result, dict) and order_result.get('liquidity'):
                                    liquidity = order_result.get('liquidity')
                                self.last_trade_ts = asyncio.get_event_loop().time()
                                
                                # Log trade to database
                                if self.session_id and order_result:
                                    try:
                                        realized_pnl = self._update_holdings_and_realized(symbol, action, quantity, price, fee)
                                        self.db.log_trade(
                                            self.session_id,
                                            symbol,
                                            action,
                                            quantity,
                                            price,
                                            fee,
                                            reason,
                                            liquidity=liquidity,
                                            realized_pnl=realized_pnl
                                        )
                                        # Snapshot open orders if any remain
                                        try:
                                            open_orders = await self.bot.get_open_orders_async()
                                            self.db.replace_open_orders(self.session_id, open_orders)
                                        except Exception as e:
                                            logger.warning(f"Could not snapshot open orders: {e}")
                                    except Exception as e:
                                        logger.warning(f"Error logging trade: {e}")
                                elif self.session_id and not order_result:
                                    bot_actions_logger.info("⚠️ Order failed or timed out; not logged.")
                            else:
                                bot_actions_logger.info(f"⛔ Trade Blocked: {risk_result.reason}")
                                self.last_rejection_reason = risk_result.reason
                    
                    # 6. Sleep
                    logger.info(f"Sleeping for {LOOP_INTERVAL_SECONDS} seconds...")
                    await asyncio.sleep(LOOP_INTERVAL_SECONDS)

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
