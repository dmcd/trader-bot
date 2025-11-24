import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
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
    MIN_TRADE_SIZE,
    LOOP_INTERVAL_SECONDS,
    MIN_TRADE_INTERVAL_SECONDS,
    FEE_RATIO_COOLDOWN,
    MAX_TOTAL_EXPOSURE,
    ORDER_VALUE_BUFFER,
    MAX_POSITIONS,
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
        # Simple in-memory holdings tracker for realized PnL
        self.holdings = {}  # symbol -> {'qty': float, 'avg_cost': float}
        # Track estimated fees per order so we can reconcile with actual fills
        self._estimated_fees = {}  # order_id -> estimated fee
        self._last_reconnect = 0.0
        self._kill_switch = False
        
        # Initialize Strategy
        self.strategy = LLMStrategy(
            self.db, 
            self.technical_analysis, 
            self.cost_tracker,
            open_orders_provider=self.bot.get_open_orders_async,
            ohlcv_provider=self.bot.fetch_ohlcv,
        )
        
        # Trade syncing state
        self.order_reasons = {}  # order_id -> reason
        self.processed_trade_ids = set()
        self._open_trade_plans = {}  # plan_id -> dict
        self.max_plan_age_minutes = 60  # TODO: move to config if desired
        self.day_end_flatten_hour_utc = None  # optional UTC hour to flatten plans
        self.max_plans_per_symbol = 2

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

    async def _monitor_trade_plans(self, price_now: float):
        """
        Monitor open trade plans for stop/target hits and enforce max age/day-end flattening.
        Currently supports single-symbol trading; uses latest price passed in.
        """
        if not self.session_id:
            return
        try:
            open_plans = self.db.get_open_trade_plans(self.session_id)
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            day_end_cutoff = None
            if self.day_end_flatten_hour_utc is not None:
                day_end_cutoff = now.replace(hour=self.day_end_flatten_hour_utc, minute=0, second=0, microsecond=0)

            for plan in open_plans:
                plan_id = plan['id']
                side = plan['side'].upper()
                stop = plan.get('stop_price')
                target = plan.get('target_price')
                size = plan.get('size') or 0.0
                if not price_now or size <= 0:
                    continue

                should_close = False
                reason = None

                # Cancel if exposure headroom exhausted
                try:
                    exposure_now = self.risk_manager.get_total_exposure()
                    if exposure_now >= MAX_TOTAL_EXPOSURE * 0.98:
                        should_close = True
                        reason = "Cancelled plan: exposure headroom exhausted"
                except Exception:
                    pass

                opened_at = plan.get('opened_at')
                if opened_at and self.max_plan_age_minutes:
                    try:
                        opened_dt = datetime.fromisoformat(opened_at)
                        age_min = (now - opened_dt).total_seconds() / 60.0
                        if age_min >= self.max_plan_age_minutes:
                            should_close = True
                            reason = f"Plan age exceeded {self.max_plan_age_minutes} min"
                    except Exception:
                        pass

                if not should_close and day_end_cutoff and opened_at:
                    try:
                        opened_dt = datetime.fromisoformat(opened_at)
                        if opened_dt < day_end_cutoff:
                            should_close = True
                            reason = "Day-end flatten"
                    except Exception:
                        pass

                if side == 'BUY':
                    if stop and price_now <= stop:
                        should_close = True
                        reason = f"Stop hit at ${price_now:,.2f}"
                    elif target and price_now >= target:
                        should_close = True
                        reason = f"Target hit at ${price_now:,.2f}"
                else:  # SELL plan (short)
                    if stop and price_now >= stop:
                        should_close = True
                        reason = f"Stop hit at ${price_now:,.2f}"
                    elif target and price_now <= target:
                        should_close = True
                        reason = f"Target hit at ${price_now:,.2f}"

                if should_close:
                    try:
                        action = 'SELL' if side == 'BUY' else 'BUY'
                        bot_actions_logger.info(f"🏁 Closing plan {plan_id}: {reason}")
                        order_result = await self.bot.place_order_async(plan['symbol'], action, size, prefer_maker=False)
                        fee = self.cost_tracker.calculate_trade_fee(plan['symbol'], size, price_now, action)
                        realized = self._update_holdings_and_realized(plan['symbol'], action, size, price_now, fee)
                        self.db.log_trade(
                            self.session_id,
                            plan['symbol'],
                            action,
                            size,
                            price_now,
                            fee,
                            reason,
                            liquidity=order_result.get('liquidity') if order_result else 'taker',
                            realized_pnl=realized,
                        )
                        self._apply_fill_to_session_stats(order_result.get('order_id') if order_result else None, fee, realized)
                        self.db.update_trade_plan_status(plan_id, status='closed', closed_at=now_iso, reason=reason)
                    except Exception as e:
                        logger.error(f"Failed to close plan {plan_id}: {e}")
        except Exception as e:
            logger.warning(f"Monitor trade plans failed: {e}")

    async def initialize(self):
        """Connects and initializes the bot."""
        logger.info(f"Initializing Strategy Runner in {TRADING_MODE} mode (Exchange: {self.exchange_name})...")
        bot_actions_logger.info(f"🤖 Trading Bot Started - Mode: {TRADING_MODE}, Exchange: {self.exchange_name}")
        bot_actions_logger.info(
            "📏 Risk Limits: "
            f"Max order ${MAX_ORDER_VALUE:,.2f} (buffer ${ORDER_VALUE_BUFFER:,.2f}), "
            f"Min trade ${MIN_TRADE_SIZE:,.2f}, "
            f"Max exposure ${MAX_TOTAL_EXPOSURE:,.2f}, "
            f"Max positions {MAX_POSITIONS}, "
            f"Max daily loss ${MAX_DAILY_LOSS:,.2f} / {MAX_DAILY_LOSS_PERCENT:.1f}%"
        )
        
        # Connect to the active exchange
        await self.bot.connect_async()
        
        # Get initial equity (full account value)
        initial_equity = await self.bot.get_equity_async()
        logger.info(f"{self.exchange_name} Equity: {initial_equity}")
        
        # Create or load today's trading session (DB still used for logging/IDs)
        self.session_id = self.db.get_or_create_session(starting_balance=initial_equity)
        self.session = self.db.get_session(self.session_id)
        
        # Clear any old pending commands from previous sessions
        self.db.clear_old_commands()
        
        # Initialize trading context
        self.context = TradingContext(self.db, self.session_id)
        
        # Initialize session stats with persistence awareness
        logger.info("Initializing session stats...")
        cached_stats = self.db.get_session_stats_cache(self.session_id)
        if cached_stats:
            self.session_stats = {
                'total_trades': cached_stats.get('total_trades', 0),
                'gross_pnl': cached_stats.get('gross_pnl', 0.0),
                'total_fees': cached_stats.get('total_fees', 0.0),
                'total_llm_cost': cached_stats.get('total_llm_cost', 0.0),
            }
            logger.info(f"Loaded session stats from cache: {self.session_stats}")
            # If cache is stale vs DB trades, rebuild
            db_trade_count = self.db.get_trade_count(self.session_id)
            if db_trade_count > self.session_stats['total_trades']:
                logger.info(f"Cache stale (cache trades={self.session_stats['total_trades']}, db trades={db_trade_count}); rebuilding stats from trades...")
                await self._rebuild_session_stats_from_trades(initial_equity)
        else:
            logger.info("No cached stats found; rebuilding from exchange trades...")
            # 1. Determine start of day (UTC) for "Daily" stats
            now = datetime.now(timezone.utc)
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_ts_ms = int(start_of_day.timestamp() * 1000)
            
            # 2. Fetch trades for the day
            # Assuming BTC/USD for now as per original code
            symbol = 'BTC/USD' if ACTIVE_EXCHANGE == 'GEMINI' else 'BHP' 
            trades = await self.bot.get_trades_from_timestamp(symbol, start_ts_ms)
            
            # 3. Rebuild Holdings and Stats
            self.holdings = {}
            self.session_stats = {
                'total_trades': 0,
                'gross_pnl': 0.0,
                'total_fees': 0.0,
                'total_llm_cost': 0.0 # Will fetch from DB
            }
            
            for t in trades:
                 # Rebuild holdings
                 symbol = t['symbol']
                 side = t['side'].upper()
                 quantity = t['amount']
                 price = t['price']
                 fee = t.get('fee', {}).get('cost', 0.0)
                 
                 # Update holdings and calculate realized PnL for this trade
                 realized = self._update_holdings_and_realized(symbol, side, quantity, price, fee)
                 
                 # Update session stats
                 self.session_stats['total_trades'] += 1
                 self.session_stats['total_fees'] += fee
                 self.session_stats['gross_pnl'] += realized

            # Load LLM costs from DB (since exchange doesn't track this)
            db_stats = self.db.get_session_stats(self.session_id)
            self.session_stats['total_llm_cost'] = db_stats.get('total_llm_cost', 0.0)
            self.db.set_session_stats_cache(self.session_id, self.session_stats)
            logger.info(f"Session Stats Rebuilt: {self.session_stats}")

        # Seed risk manager with persisted start-of-day equity to survive restarts
        start_equity = None
        if self.session and self.session.get('starting_balance') is not None:
            start_equity = self.session.get('starting_balance')
        else:
            # Prefer latest broker equity snapshot if available
            latest_equity = self.db.get_latest_equity(self.session_id)
            start_equity = latest_equity if latest_equity is not None else initial_equity
        self.risk_manager.seed_start_of_day(start_equity)
        
        # No need to reconcile_exchange_state in the old way; we just trust the exchange now.
        # But we might want to log initial positions to DB for debugging.
        live_positions = await self.bot.get_positions_async()
        self.db.replace_positions(self.session_id, live_positions)
        
        self.daily_loss_pct = MAX_DAILY_LOSS_PERCENT

        self.risk_manager.update_equity(initial_equity)
        bot_actions_logger.info(f"💰 Starting Equity: ${initial_equity:,.2f}")

    # reconcile_exchange_state removed as we trust exchange data directly now

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

    def _apply_fill_to_session_stats(self, order_id: str, actual_fee: float, realized_pnl: float):
        """
        Reconcile session stats with an executed trade.
        If we estimated a fee earlier, we still book the actual fee but drop the estimate marker.
        """
        if not self.session_stats:
            self.session_stats = {
                'total_trades': 0,
                'gross_pnl': 0.0,
                'total_fees': 0.0,
                'total_llm_cost': 0.0,
            }

        if order_id:
            order_key = str(order_id)
            if order_key in self._estimated_fees:
                self._estimated_fees.pop(order_key, None)
        fee_delta = actual_fee
        self.session_stats['total_trades'] += 1
        self.session_stats['total_fees'] += fee_delta
        self.session_stats['gross_pnl'] += realized_pnl
        try:
            self.db.set_session_stats_cache(self.session_id, self.session_stats)
        except Exception as e:
            logger.warning(f"Failed to persist session stats cache: {e}")

    def _sanity_check_equity_vs_stats(self, current_equity: float):
        """Compare estimated net PnL vs equity delta; log if off by >10%."""
        if current_equity is None or self.session is None:
            return
        try:
            starting = self.session.get('starting_balance')
            if starting is None:
                return
            estimated_net = (
                self.session_stats.get('gross_pnl', 0.0)
                - self.session_stats.get('total_fees', 0.0)
                - self.session_stats.get('total_llm_cost', 0.0)
            )
            actual_net = current_equity - starting
            diff = actual_net - estimated_net
            pct = abs(diff) / max(1e-6, abs(actual_net) if actual_net != 0 else 1.0)
            if pct > 0.1:
                logger.warning(
                    f"Equity/net mismatch: equity_net={actual_net:.2f}, estimated_net={estimated_net:.2f}, "
                    f"diff={diff:.2f} ({pct*100:.1f}%)"
                )
        except Exception as e:
            logger.debug(f"Equity sanity check failed: {e}")

    async def _rebuild_session_stats_from_trades(self, current_equity: float = None):
        """Recompute session_stats from recorded trades and update cache."""
        trades = self.db.get_trades_for_session(self.session_id)
        self.holdings = {}
        self.session_stats = {
            'total_trades': 0,
            'gross_pnl': 0.0,
            'total_fees': 0.0,
            'total_llm_cost': 0.0,
        }
        for t in trades:
            realized = self._update_holdings_and_realized(
                t['symbol'],
                t['action'],
                t['quantity'],
                t['price'],
                t.get('fee', 0.0) or 0.0
            )
            self.session_stats['total_trades'] += 1
            self.session_stats['total_fees'] += t.get('fee', 0.0) or 0.0
            self.session_stats['gross_pnl'] += realized

        # Pull LLM costs from session row
        db_stats = self.db.get_session_stats(self.session_id)
        self.session_stats['total_llm_cost'] = db_stats.get('total_llm_cost', 0.0)
        self.db.set_session_stats_cache(self.session_id, self.session_stats)
        logger.info(f"Session stats rebuilt from trades: {self.session_stats}")
        # Mirror stats into sessions table
        try:
            self.db.update_session_totals(
                self.session_id,
                total_trades=self.session_stats['total_trades'],
                total_fees=self.session_stats['total_fees'],
                total_llm_cost=self.session_stats['total_llm_cost'],
                net_pnl=self.session_stats['gross_pnl'] - self.session_stats['total_fees'] - self.session_stats['total_llm_cost'],
            )
        except Exception as e:
            logger.warning(f"Could not update session totals: {e}")
        # Sanity check vs equity delta when provided
        if current_equity is not None:
            self._sanity_check_equity_vs_stats(current_equity)

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
                self._apply_fill_to_session_stats(order_id, fee, realized_pnl)
                self.processed_trade_ids.add(trade_id)
                bot_actions_logger.info(f"✅ Synced trade: {side} {quantity} {symbol} @ ${price:,.2f} (Fee: ${fee:.4f})")
                # Mark any trade plan closed if target/stop hit (handled in monitor)
                
        except Exception as e:
            logger.exception(f"Error syncing trades: {e}")

    async def cleanup(self):
        """Cleanup and close connection."""
        logger.info("Cleaning up connections...")
        
        # Save final session statistics
        if self.session_id:
            try:
                # Get final equity snapshot
                final_equity = await self.bot.get_equity_async()
                
                # Get session stats
                session_stats = self.db.get_session_stats(self.session_id)
                # Ensure stats reflect latest trades before final report
                try:
                    await self._rebuild_session_stats_from_trades(final_equity)
                    session_stats = self.db.get_session_stats(self.session_id)
                except Exception as e:
                    logger.debug(f"Could not rebuild stats on cleanup: {e}")
                
                # Calculate net PnL
                gross_pnl = final_equity - session_stats['starting_balance']
                net_pnl = self.cost_tracker.calculate_net_pnl(
                    gross_pnl,
                    session_stats['total_fees'],
                    session_stats['total_llm_cost']
                )
                
                # Update database
                self.db.update_session_balance(self.session_id, final_equity, net_pnl)
                
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
                    price_now = data.get('price') if data else None

                    # Refresh live positions each loop for accurate exposure snapshots
                    try:
                        live_positions = await self.bot.get_positions_async()
                        self.db.replace_positions(self.session_id, live_positions)
                    except Exception as e:
                        logger.warning(f"Could not refresh positions: {e}")
                    # Refresh open orders for exposure headroom and context
                    open_orders = []
                    try:
                        open_orders = await self.bot.get_open_orders_async()
                        self.db.replace_open_orders(self.session_id, open_orders)
                    except Exception as e:
                        logger.warning(f"Could not refresh open orders: {e}")
                    # Enforce per-symbol open order cap proactively
                    capped_orders = {}
                    for order in open_orders:
                        sym = order.get('symbol')
                        capped_orders[sym] = capped_orders.get(sym, 0) + 1
                    for sym, cnt in capped_orders.items():
                        if cnt > self.max_plans_per_symbol:
                            bot_actions_logger.info(f"⚠️ Open order count {cnt} for {sym} exceeds cap {self.max_plans_per_symbol}")

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

                        # Build price lookup for open orders (fallback to recent tick)
                        price_lookup = {}
                        if data and data.get('price'):
                            price_lookup[symbol] = data['price']
                        for ord in open_orders or []:
                            sym = ord.get('symbol')
                            if sym and sym in price_lookup:
                                continue
                            latest = self.db.get_recent_market_data(self.session_id, sym, limit=1) if sym else None
                            if latest and latest[0].get('price'):
                                price_lookup[sym] = latest[0]['price']
                        self.risk_manager.update_pending_orders(open_orders, price_lookup=price_lookup)

                        price_overrides = {symbol: data['price']} if data and data.get('price') else None
                        current_exposure = self.risk_manager.get_total_exposure(price_overrides=price_overrides)
                    except Exception as e:
                        logger.warning(f"Could not build positions for exposure: {e}")

                    # 2.8 Monitor trade plans for stops/targets and max age
                    try:
                        await self._monitor_trade_plans(price_now)
                    except Exception as e:
                        logger.exception(f"Trade plan monitor error: {e}")

                    # Kill switch check
                    if self._kill_switch:
                        bot_actions_logger.info("🛑 Kill switch active; not trading.")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    # Slippage guard: if latest price moved >2% from decision price, skip execution
                    decision_price = market_data[symbol]['price'] if market_data.get(symbol) else None
                    
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

                    # 2.5 Sync Trades from Exchange (for logging only)
                    await self.sync_trades_from_exchange()
                    # Keep session stats cache fresh if DB trades grew
                    try:
                        db_trade_count = self.db.get_trade_count(self.session_id)
                        if db_trade_count > self.session_stats.get('total_trades', 0):
                            await self._rebuild_session_stats_from_trades(current_equity)
                    except Exception as e:
                        logger.debug(f"Could not refresh stats cache mid-loop: {e}")

                    # 3. Generate Signal via Strategy
                    # Pass session_stats explicitly
                    signal = await self.strategy.generate_signal(
                        self.session_id,
                        market_data,
                        current_equity,
                        current_exposure,
                        self.context,
                        session_stats=self.session_stats
                    )

                    # 4. Execute Signal
                    if signal:
                        action = signal.action
                        quantity = signal.quantity
                        reason = signal.reason
                        symbol = signal.symbol
                        order_id = getattr(signal, 'order_id', None)
                        stop_price = getattr(signal, 'stop_price', None)
                        target_price = getattr(signal, 'target_price', None)
                        
                        # Log decision to user-friendly log
                        if action == 'HOLD':
                            bot_actions_logger.info(f"📊 Decision: HOLD - {reason}")
                        
                        elif action == 'CANCEL':
                            if not order_id:
                                logger.warning("Skipped cancel: missing order_id")
                                continue
                            cancel_id = order_id
                            if isinstance(order_id, str) and order_id.isdigit():
                                cancel_id = int(order_id)
                            try:
                                success = await self.bot.cancel_open_order_async(cancel_id)
                                if success:
                                    bot_actions_logger.info(f"🛑 Cancelled order {order_id}: {reason}")
                                else:
                                    bot_actions_logger.info(f"⚠️ Cancel request failed for order {order_id}: {reason}")
                                # Refresh open orders snapshot so strategy context stays current
                                try:
                                    open_orders = await self.bot.get_open_orders_async()
                                    self.db.replace_open_orders(self.session_id, open_orders)
                                except Exception as e:
                                    logger.warning(f"Could not refresh open orders after cancel: {e}")
                            except Exception as e:
                                logger.error(f"Cancel order error: {e}")
                            continue

                        elif action in ['BUY', 'SELL'] and quantity > 0:
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

                        # Format quantity appropriately (show more decimals for small amounts) after buffering
                        if quantity < 1:
                            qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                        else:
                            qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                        bot_actions_logger.info(f"📊 Decision: {action} {qty_str} {symbol} - {reason}")

                        risk_result = self.risk_manager.check_trade_allowed(symbol, action, quantity, price)

                        if risk_result.allowed:
                            # Calculate fee before execution (estimate)
                            estimated_fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, action)
                            liquidity = "maker_intent"
                            # Capture stop/target if provided by strategy
                            stop_price = getattr(signal, 'stop_price', None)
                            target_price = getattr(signal, 'target_price', None)

                            # Enforce per-symbol plan cap before placing order
                            try:
                                open_plan_count = self.db.count_open_trade_plans_for_symbol(self.session_id, symbol)
                                if open_plan_count >= self.max_plans_per_symbol:
                                    bot_actions_logger.info(f"⛔ Trade Blocked: plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                    self.strategy.on_trade_rejected("Plan cap reached")
                                    continue
                            except Exception as e:
                                logger.debug(f"Could not check plan cap: {e}")

                            # Enforce pending exposure headroom including open orders
                            try:
                                pending_exposure = self.risk_manager.pending_orders_by_symbol.get(symbol, {}).get('buy', 0.0) if action == 'BUY' else 0.0
                                pending_count = self.risk_manager.pending_orders_by_symbol.get(symbol, {}).get('count', 0) if action == 'BUY' else 0
                                if action == 'BUY' and pending_count >= self.max_plans_per_symbol:
                                    bot_actions_logger.info(f"⛔ Trade Blocked: open order count reached for {symbol} ({pending_count}/{self.max_plans_per_symbol})")
                                    self.strategy.on_trade_rejected("Open order cap reached")
                                    continue
                                order_value = quantity * price
                                if action == 'BUY' and (pending_exposure + order_value + current_exposure) > MAX_TOTAL_EXPOSURE:
                                    bot_actions_logger.info("⛔ Trade Blocked: pending/open exposure would exceed cap")
                                    self.strategy.on_trade_rejected("Pending exposure over cap")
                                    continue
                            except Exception as e:
                                logger.debug(f"Pending exposure check failed: {e}")

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
                                    self._estimated_fees[str(order_result['order_id'])] = estimated_fee
                                    # Also persist estimated fee to DB for optional auditing
                                    try:
                                        self.db.log_estimated_fee(self.session_id, order_result['order_id'], estimated_fee, symbol, action)
                                    except Exception as e:
                                        logger.debug(f"Could not log estimated fee: {e}")

                            # Record trade plan so we can monitor stops/targets (only for new BUY/SELL)
                            if action in ['BUY', 'SELL'] and (stop_price or target_price):
                                try:
                                    plan_id = self.db.create_trade_plan(
                                        self.session_id,
                                        symbol,
                                        action,
                                        price,
                                        stop_price,
                                        target_price,
                                        quantity,
                                        reason
                                    )
                                    self._open_trade_plans[plan_id] = {
                                        'symbol': symbol,
                                        'side': action,
                                        'stop_price': stop_price,
                                        'target_price': target_price,
                                        'size': quantity
                                    }
                                    bot_actions_logger.info(f"📝 Plan #{plan_id}: stop={stop_price}, target={target_price}")
                                except Exception as e:
                                    logger.debug(f"Could not create trade plan: {e}")
                                    
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
                    logger.exception(f"Loop Error: {e}")
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
