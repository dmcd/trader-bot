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
from strategy import LLMStrategy
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
    MAX_TOTAL_EXPOSURE,
    ORDER_VALUE_BUFFER,
    PRIORITY_MOVE_PCT,
    PRIORITY_LOOKBACK_MIN,
    BREAK_GLASS_COOLDOWN_MIN,
    BREAK_GLASS_SIZE_FACTOR,
)

from logger_config import setup_logging

# Configure logging
bot_actions_logger = setup_logging()
logger = logging.getLogger(__name__)

# Configure Gemini (still needed for direct usage if any, but mostly in Strategy now)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
        self.running = False
        
        # Professional trading infrastructure
        self.db = TradingDatabase()
        self.cost_tracker = CostTracker(self.exchange_name)
        self.technical_analysis = TechnicalAnalysis()
        self.session_id = None
        self.context = None
        self.session = None
        self.sandbox_seed_symbols = {'USD', 'BTC/USD', 'BTC', 'ETH/USD', 'LTC/USD', 'ZEC/USD', 'BCH/USD'}
        # Simple in-memory holdings tracker for realized PnL
        self.holdings = {}  # symbol -> {'qty': float, 'avg_cost': float}
        self.size_tier = SIZE_TIER if SIZE_TIER in ORDER_SIZE_BY_TIER else 'MODERATE'
        self._last_reconnect = 0.0
        self._kill_switch = False
        
        # Initialize Strategy
        self.strategy = LLMStrategy(
            self.db, 
            self.technical_analysis, 
            self.cost_tracker, 
            self.size_tier
        )
        
        # Trade syncing state
        self.order_reasons = {}  # order_id -> reason
        self.processed_trade_ids = set()

    def _apply_order_value_buffer(self, quantity: float, price: float):
        """Trim quantity so the notional sits under the order cap minus buffer."""
        adjusted_qty, overage = self.risk_manager.apply_order_value_buffer(quantity, price)
        if adjusted_qty < quantity:
            original_value = quantity * price
            adjusted_value = adjusted_qty * price
            bot_actions_logger.info(
                f"✂️ Trimmed order from ${original_value:.2f} to ${adjusted_value:.2f} "
                f"to stay under ${MAX_ORDER_VALUE - ORDER_VALUE_BUFFER:.2f} cap"
            )
        return adjusted_qty

    async def initialize(self):
        """Connects and initializes the bot."""
        logger.info(f"Initializing Strategy Runner in {TRADING_MODE} mode (Exchange: {self.exchange_name})...")
        bot_actions_logger.info(f"🤖 Trading Bot Started - Mode: {TRADING_MODE}, Exchange: {self.exchange_name}")
        
        # Connect to the active exchange
        await self.bot.connect_async()
        
        # Get initial equity (full account value)
        initial_equity = await self.bot.get_equity_async()
        logger.info(f"{self.exchange_name} Equity: {initial_equity}")
        
        # Create or load today's trading session
        self.session_id = self.db.get_or_create_session(starting_balance=initial_equity)
        self.session = self.db.get_session(self.session_id)
        
        # Clear any old pending commands from previous sessions
        self.db.clear_old_commands()
        
        # Initialize trading context
        self.context = TradingContext(self.db, self.session_id)
        
        # Sync trades from exchange to ensure local state is up to date before risk/holdings init
        logger.info("Syncing trades from exchange before initialization...")
        await self.sync_trades_from_exchange()
        
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

        self.risk_manager.update_equity(initial_equity)
        bot_actions_logger.info(f"💰 Starting Equity: ${initial_equity:,.2f}")

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

    async def sync_trades_from_exchange(self):
        """Sync recent trades from exchange to DB."""
        if not self.session_id:
            return

        try:
            # Fetch recent trades
            trades = await self.bot.get_my_trades_async('BTC/USD', limit=20)
            
            for t in trades:
                trade_id = str(t['id'])
                if trade_id in self.processed_trade_ids:
                    continue
                
                # Check DB for existence
                existing = self.db.conn.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
                if existing:
                    self.processed_trade_ids.add(trade_id)
                    continue
                
                order_id = t.get('order')
                symbol = t['symbol']
                side = t['side'].upper()
                price = t['price']
                quantity = t['amount']
                fee = t.get('fee', {}).get('cost', 0.0)
                
                # Extract liquidity if available
                liquidity = 'unknown'
                if 'liquidity' in t.get('info', {}):
                    liquidity = t['info']['liquidity']
                
                # Get reason from local memory
                reason = self.order_reasons.get(str(order_id), "Synced from exchange")
                
                # Calculate realized PnL
                realized_pnl = self._update_holdings_and_realized(symbol, side, quantity, price, fee)
                
                self.db.log_trade(
                    self.session_id,
                    symbol,
                    side,
                    quantity,
                    price,
                    fee,
                    reason,
                    liquidity=liquidity,
                    realized_pnl=realized_pnl,
                    trade_id=trade_id
                )
                self.processed_trade_ids.add(trade_id)
                bot_actions_logger.info(f"✅ Synced trade: {side} {quantity} {symbol} @ ${price:,.2f} (Fee: ${fee:.4f})")
                
        except Exception as e:
            logger.error(f"Error syncing trades: {e}")

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
                            await self._close_all_positions_safely()
                            bot_actions_logger.info("✅ All positions closed")
                            self.db.mark_command_executed(command_id)
                            
                        elif command == 'STOP_BOT':
                            logger.info("Executing command: STOP_BOT")
                            bot_actions_logger.info("🛑 Manual Command: Stopping bot...")
                            self.db.mark_command_executed(command_id)
                            self.running = False
                            break
                    
                    # 1. Update Equity / PnL
                    current_equity = await self.bot.get_equity_async()
                    if self.session_id is not None:
                        try:
                            self.db.log_equity_snapshot(self.session_id, current_equity)
                        except Exception as e:
                            logger.warning(f"Could not log equity snapshot: {e}")
                    self.risk_manager.update_equity(current_equity)
                    
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

                    # Build latest positions with marks for exposure checks
                    positions_dict = {}
                    current_exposure = 0.0
                    try:
                        positions_data = self.db.get_positions(self.session_id)
                        for pos in positions_data:
                            sym = pos['symbol']
                            current_price = pos.get('avg_price') or 0

                            # Prefer most recent market tick
                            recent_data = self.db.get_recent_market_data(self.session_id, sym, limit=1)
                            if recent_data and recent_data[0].get('price'):
                                current_price = recent_data[0]['price']

                            # If this is the actively traded symbol, use live price
                            if sym == symbol and data and data.get('price'):
                                current_price = data['price']

                            if current_price:
                                positions_dict[sym] = {
                                    'quantity': pos['quantity'],
                                    'current_price': current_price
                                }
                        self.risk_manager.update_positions(positions_dict)
                        price_overrides = {symbol: data['price']} if data and data.get('price') else None
                        current_exposure = self.risk_manager.get_total_exposure(price_overrides=price_overrides)
                    except Exception as e:
                        logger.warning(f"Could not build positions for exposure: {e}")

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

                    # 2.5 Sync Trades from Exchange
                    await self.sync_trades_from_exchange()

                    # 3. Generate Signal via Strategy
                    signal = await self.strategy.generate_signal(
                        self.session_id,
                        market_data,
                        current_equity,
                        current_exposure,
                        self.context
                    )

                    # 4. Execute Signal
                    if signal:
                        action = signal.action
                        quantity = signal.quantity
                        reason = signal.reason
                        symbol = signal.symbol
                        
                        # Log decision to user-friendly log
                        if action == 'HOLD':
                            bot_actions_logger.info(f"📊 Decision: HOLD - {reason}")
                        
                        elif action in ['BUY', 'SELL'] and quantity > 0:
                            # Format quantity appropriately (show more decimals for small amounts)
                            if quantity < 1:
                                qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                            else:
                                qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                            bot_actions_logger.info(f"📊 Decision: {action} {qty_str} {symbol} - {reason}")

                            # Get price for risk checks and execution
                            md = market_data.get(symbol)
                            price = md.get('price') if md else (data['price'] if data else 0)
                            
                            if not price:
                                logger.warning("Skipped trade: missing price data")
                                continue

                            # Guardrails: clip size to sit under the max order cap minus buffer
                            quantity = self._apply_order_value_buffer(quantity, price)

                            if quantity <= 0:
                                logger.warning("Skipped trade: buffered quantity became non-positive")
                                continue

                            risk_result = self.risk_manager.check_trade_allowed(symbol, action, quantity, price)

                            if risk_result.allowed:
                                # Calculate fee before execution (estimate)
                                estimated_fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, action)
                                liquidity = "maker_intent"
                                
                                bot_actions_logger.info(f"✅ Executing: {action} {qty_str} {symbol} at ${price:,.2f} (est. fee: ${estimated_fee:.4f})")
                                
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
                                
                                # Notify strategy of execution
                                now_ts = asyncio.get_event_loop().time()
                                self.strategy.on_trade_executed(now_ts)
                                
                                # Store reason for syncing
                                if order_result and order_result.get('order_id'):
                                    self.order_reasons[str(order_result['order_id'])] = reason
                                    
                                # Snapshot open orders if any remain
                                try:
                                    open_orders = await self.bot.get_open_orders_async()
                                    self.db.replace_open_orders(self.session_id, open_orders)
                                except Exception as e:
                                    logger.warning(f"Could not snapshot open orders: {e}")

                            else:
                                bot_actions_logger.info(f"⛔ Trade Blocked: {risk_result.reason}")
                                self.strategy.on_trade_rejected(risk_result.reason)
                    
                    # 5. Sleep
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
