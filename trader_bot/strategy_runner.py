import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import google.generativeai as genai

from trader_bot.config import (
    ACTIVE_EXCHANGE,
    AUTO_REPLACE_PLAN_ON_CAP,
    BREAK_GLASS_COOLDOWN_MIN,
    BREAK_GLASS_SIZE_FACTOR,
    CLIENT_ORDER_PREFIX,
    FEE_RATIO_COOLDOWN,
    GEMINI_API_KEY,
    HIGH_VOL_SIZE_FACTOR,
    COMMAND_RETENTION_DAYS,
    OHLCV_MAX_ROWS_PER_TIMEFRAME,
    OHLCV_MIN_CAPTURE_SPACING_SECONDS,
    LOOP_INTERVAL_SECONDS,
    MARKET_DATA_RETENTION_MINUTES,
    MAX_DAILY_LOSS,
    MAX_DAILY_LOSS_PERCENT,
    MAX_ORDER_VALUE,
    MAX_POSITIONS,
    MAX_SLIPPAGE_PCT,
    MAX_SPREAD_PCT,
    MAX_TOTAL_EXPOSURE,
    MED_VOL_SIZE_FACTOR,
    MIN_RR,
    MIN_TRADE_SIZE,
    MIN_TOP_OF_BOOK_NOTIONAL,
    LLM_MAX_SESSION_COST,
    ORDER_VALUE_BUFFER,
    LLM_PROVIDER,
    PLAN_MAX_AGE_MINUTES,
    PLAN_MAX_PER_SYMBOL,
    PLAN_TRAIL_TO_BREAKEVEN_PCT,
    PRIORITY_LOOKBACK_MIN,
    PRIORITY_MOVE_PCT,
    BOT_VERSION,
    TRADE_SYNC_CUTOFF_MINUTES,
    TRADING_MODE,
    EXCHANGE_ERROR_THRESHOLD,
    EXCHANGE_PAUSE_SECONDS,
    TOOL_ERROR_THRESHOLD,
    TOOL_PAUSE_SECONDS,
    TICKER_MAX_AGE_SECONDS,
    TICKER_MAX_LATENCY_MS,
    MAKER_PREFERENCE_DEFAULT,
    MAKER_PREFERENCE_OVERRIDES,
    ALLOWED_SYMBOLS,
)
from trader_bot.cost_tracker import CostTracker
from trader_bot.data_fetch_coordinator import DataFetchCoordinator
from trader_bot.database import TradingDatabase
from trader_bot.gemini_trader import GeminiTrader
from trader_bot.logger_config import setup_logging
from trader_bot.risk_manager import RiskManager
from trader_bot.strategy import LLMStrategy
from trader_bot.services.command_processor import CommandProcessor
from trader_bot.services.portfolio_tracker import PortfolioTracker
from trader_bot.services.plan_monitor import PlanMonitor, PlanMonitorConfig
from trader_bot.services.health_manager import HealthCircuitManager
from trader_bot.services.trade_action_handler import TradeActionHandler
from trader_bot.technical_analysis import TechnicalAnalysis
from trader_bot.trading_context import TradingContext
from trader_bot.utils import get_client_order_id

# Configure logging
bot_actions_logger = setup_logging()
telemetry_logger = logging.getLogger('telemetry')
logger = logging.getLogger(__name__)

# Configure Gemini (still needed for direct usage if any, but mostly in Strategy now)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class StrategyRunner:
    def __init__(self, execute_orders: bool = True):
        # Only instantiate the bot for the active exchange
        # Only instantiate the bot for the active exchange
        if ACTIVE_EXCHANGE == 'GEMINI':
            self.bot = GeminiTrader()
            self.exchange_name = 'GEMINI'
        else:
            # Default to Gemini if not specified or invalid
            logger.warning(f"Unknown ACTIVE_EXCHANGE '{ACTIVE_EXCHANGE}', defaulting to GEMINI")
            self.bot = GeminiTrader()
            self.exchange_name = 'GEMINI'
        
        self.risk_manager = RiskManager(self.bot)
        self.running = False
        self.execute_orders = execute_orders
        
        # Professional trading infrastructure
        self.db = TradingDatabase()
        self.cost_tracker = CostTracker(self.exchange_name, llm_provider=LLM_PROVIDER)
        self.technical_analysis = TechnicalAnalysis()
        self.session_id = None
        self.context = None
        self.session = None
        self.data_fetch_coordinator = None
        # Track estimated fees per order so we can reconcile with actual fills
        self._estimated_fees = {}  # order_id -> estimated fee
        self._kill_switch = False
        self.exchange_error_threshold = EXCHANGE_ERROR_THRESHOLD
        self.exchange_pause_seconds = EXCHANGE_PAUSE_SECONDS
        self.tool_error_threshold = TOOL_ERROR_THRESHOLD
        self.tool_pause_seconds = TOOL_PAUSE_SECONDS
        self.ticker_max_age_ms = TICKER_MAX_AGE_SECONDS * 1000
        self.ticker_max_latency_ms = TICKER_MAX_LATENCY_MS
        self.health_manager = HealthCircuitManager(
            record_health_state=self._record_health_state,
            exchange_error_threshold=self.exchange_error_threshold,
            exchange_pause_seconds=self.exchange_pause_seconds,
            tool_error_threshold=self.tool_error_threshold,
            tool_pause_seconds=self.tool_pause_seconds,
            ticker_max_age_ms=self.ticker_max_age_ms,
            ticker_max_latency_ms=self.ticker_max_latency_ms,
            monotonic=self._monotonic,
            actions_logger=bot_actions_logger,
            logger=logger,
        )
        self.portfolio_tracker = PortfolioTracker(self.db, logger=logger)
        self.holdings = self.portfolio_tracker.holdings
        self.action_handler = TradeActionHandler(
            db=self.db,
            bot=self.bot,
            risk_manager=self.risk_manager,
            cost_tracker=self.cost_tracker,
            portfolio_tracker=self.portfolio_tracker,
            prefer_maker=self._prefer_maker,
            health_manager=self.health_manager,
            emit_telemetry=self._emit_telemetry,
            log_execution_trace=self._log_execution_trace,
            actions_logger=bot_actions_logger,
            logger=logger,
        )
        self.maker_preference_default = MAKER_PREFERENCE_DEFAULT
        self.maker_preference_overrides = MAKER_PREFERENCE_OVERRIDES or {}
        # Seed a default stats container so background tasks don't crash before initialization completes
        self.session_stats = self.portfolio_tracker.session_stats
        
        # Initialize Strategy
        self.strategy = LLMStrategy(
            self.db, 
            self.technical_analysis, 
            self.cost_tracker,
            open_orders_provider=self.bot.get_open_orders_async,
            ohlcv_provider=self.bot.fetch_ohlcv,
            tool_coordinator=None,  # set post-connect when exchange is ready
        )
        # Wire action handler rejection callback once strategy exists
        self.action_handler.on_trade_rejected = getattr(self.strategy, "on_trade_rejected", None)
        
        # Trade syncing state
        self.order_reasons = {}  # order_id -> reason
        self.processed_trade_ids = set()
        self._last_ohlcv_capture: dict[tuple[str, str], float] = {}
        self._open_trade_plans = {}  # plan_id -> dict
        self.max_plan_age_minutes = PLAN_MAX_AGE_MINUTES
        self.day_end_flatten_hour_utc = None  # optional UTC hour to flatten plans
        self.max_plans_per_symbol = PLAN_MAX_PER_SYMBOL
        self.telemetry_logger = telemetry_logger
        self._apply_plan_trailing_pct = PLAN_TRAIL_TO_BREAKEVEN_PCT  # move stop to entry after move in favor
        self.shutdown_reason: str | None = None  # track why we stop
        self.ohlcv_min_capture_spacing_seconds = OHLCV_MIN_CAPTURE_SPACING_SECONDS
        self.ohlcv_retention_limit = OHLCV_MAX_ROWS_PER_TIMEFRAME
        self.command_processor = CommandProcessor(self.db)
        self.plan_monitor = self._build_plan_monitor()

    def _emit_telemetry(self, record: dict):
        """Emit structured telemetry as JSON line."""
        if not self.telemetry_logger:
            return
        try:
            self.telemetry_logger.info(json.dumps(record, default=str))
        except Exception as e:
            logger.debug(f"Telemetry emit failed: {e}")

    def _monotonic(self) -> float:
        """Wrapper to allow deterministic testing."""
        try:
            return asyncio.get_event_loop().time()
        except Exception:
            return 0.0

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert simple timeframe strings like '1m' or '1h' to seconds."""
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            if unit == 'm':
                return value * 60
            if unit == 'h':
                return value * 3600
            if unit == 'd':
                return value * 86400
        except Exception:
            return 0
        return 0

    def _build_plan_monitor(self) -> PlanMonitor:
        """Construct a plan monitor bound to the current dependencies."""
        return PlanMonitor(
            db=self.db,
            bot=self.bot,
            cost_tracker=self.cost_tracker,
            risk_manager=self.risk_manager,
            prefer_maker=self._prefer_maker,
            holdings_updater=self._update_holdings_and_realized,
            session_stats_applier=self._apply_fill_to_session_stats,
            max_total_exposure=MAX_TOTAL_EXPOSURE,
        )

    def _get_active_symbols(self) -> list[str]:
        """Return ordered list of symbols to monitor/trade."""
        symbols = []
        # 1) Configured symbols (ordered)
        symbols.extend([s for s in ALLOWED_SYMBOLS if s])
        # 2) Live state from DB snapshots
        try:
            positions = self.db.get_positions(self.session_id) if self.session_id else []
            symbols.extend([p.get("symbol") for p in positions or [] if p.get("symbol")])
        except Exception:
            pass
        try:
            orders = self.db.get_open_orders(self.session_id) if self.session_id else []
            symbols.extend([o.get("symbol") for o in orders or [] if o.get("symbol")])
        except Exception:
            pass
        try:
            plans = self.db.get_open_trade_plans(self.session_id) if self.session_id else []
            symbols.extend([p.get("symbol") for p in plans or [] if p.get("symbol")])
        except Exception:
            pass

        deduped = []
        seen = set()
        for sym in symbols:
            if not sym:
                continue
            sym_up = sym.upper()
            if sym_up in seen:
                continue
            # Skip non-tradable cash-like symbols (e.g., "USD") that don't have an order book
            if "/" not in sym_up:
                continue
            seen.add(sym_up)
            deduped.append(sym_up)
        return deduped or ["BTC/USD"]

    def _get_rebuild_symbols(self) -> list[str]:
        """Return symbols to use when rebuilding stats from exchange history."""
        allowed = [s.upper() for s in ALLOWED_SYMBOLS if s and "/" in s]
        try:
            exchange_symbols = getattr(self.bot, "exchange", None)
            exchange_symbols = getattr(exchange_symbols, "symbols", []) or []
        except Exception:
            exchange_symbols = []

        symbols = allowed
        if exchange_symbols:
            # Only request symbols the venue supports to avoid noisy errors
            venue_set = {s.upper() for s in exchange_symbols}
            symbols = [s for s in allowed if s in venue_set]

        if not symbols:
            symbols = ["BTC/USD"]
        return symbols

    def _record_health_state(self, key: str, value: str, detail: dict = None):
        """Persist health state and emit telemetry."""
        detail_str = None
        if detail is not None:
            try:
                detail_str = json.dumps(detail, default=str)
            except Exception:
                detail_str = str(detail)
        try:
            self.db.set_health_state(key, value, detail_str)
        except Exception as exc:
            logger.debug(f"Could not persist health state {key}: {exc}")
        record = {
            "type": "health",
            "source": key,
            "status": value,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
        }
        self._emit_telemetry(record)

    def _record_operational_metrics(self, current_exposure: float, current_equity: float):
        """Emit health metrics for risk counters and LLM budget."""
        llm_cost = (self.session_stats or {}).get("total_llm_cost", 0.0) or 0.0
        try:
            sod = self.risk_manager.start_of_day_equity or 0.0
            loss_pct = (self.risk_manager.daily_loss / sod * 100) if sod else 0.0
            gross = (self.session_stats or {}).get("gross_pnl", 0.0) or 0.0
            fees = (self.session_stats or {}).get("total_fees", 0.0) or 0.0
            fee_ratio = fees / max(abs(gross), 1.0)
            risk_detail = {
                "daily_loss": self.risk_manager.daily_loss,
                "daily_loss_pct": loss_pct,
                "daily_loss_pct_limit": self.daily_loss_pct,
                "daily_loss_limit": MAX_DAILY_LOSS,
                "exposure": current_exposure,
                "exposure_limit": MAX_TOTAL_EXPOSURE,
                "exposure_headroom": max(0.0, MAX_TOTAL_EXPOSURE - current_exposure),
                "fee_ratio": fee_ratio,
                "gross_pnl": gross,
                "total_fees": fees,
                "total_llm_cost": llm_cost,
                "equity": current_equity,
            }
            self._record_health_state("risk_metrics", "ok", risk_detail)
        except Exception as exc:
            logger.debug(f"Could not emit risk metrics: {exc}")

        try:
            session_started = None
            if self.session:
                session_started = self.session.get("created_at") or self.session.get("date")
            burn_stats = self.cost_tracker.calculate_llm_burn(
                total_llm_cost=llm_cost,
                session_started=session_started,
                budget=LLM_MAX_SESSION_COST,
            )
            budget_status = "ok"
            if burn_stats.get("remaining_budget", 0.0) <= 0:
                budget_status = "cap_hit"
            elif burn_stats.get("pct_of_budget", 0.0) >= 0.8:
                budget_status = "near_cap"
            self._record_health_state("llm_budget", budget_status, burn_stats)
        except Exception as exc:
            logger.debug(f"Could not emit LLM budget metrics: {exc}")

    async def _reconcile_exchange_state(self):
        """
        Reconcile positions and open orders against the live exchange at startup.
        Ensures DB snapshots and risk manager state reflect actual venue state.
        """
        if not self.session_id:
            return
        try:
            live_positions = await self.bot.get_positions_async()
        except Exception as exc:
            self._record_health_state("restart_recovery", "error", {"stage": "positions", "error": str(exc)})
            logger.warning(f"Could not fetch live positions during recovery: {exc}")
            return

        try:
            live_orders = await self.bot.get_open_orders_async()
            live_orders = self._filter_our_orders(live_orders)
        except Exception as exc:
            self._record_health_state("restart_recovery", "error", {"stage": "open_orders", "error": str(exc)})
            logger.warning(f"Could not fetch live open orders during recovery: {exc}")
            return

        # Load existing snapshots
        try:
            db_positions = self.db.get_positions(self.session_id) or []
            db_orders = self.db.get_open_orders(self.session_id) or []
        except Exception as exc:
            self._record_health_state("restart_recovery", "error", {"stage": "db_read", "error": str(exc)})
            logger.warning(f"Could not load DB snapshots during recovery: {exc}")
            return

        # Replace snapshots with live state
        try:
            self.db.replace_positions(self.session_id, live_positions)
            self.db.replace_open_orders(self.session_id, live_orders)
        except Exception as exc:
            self._record_health_state("restart_recovery", "error", {"stage": "db_write", "error": str(exc)})
            logger.warning(f"Could not persist reconciled snapshots: {exc}")
            return

        # Update risk manager with live state
        try:
            positions_dict = {}
            for pos in live_positions or []:
                symbol = pos.get("symbol")
                if not symbol:
                    continue
                mark = pos.get("current_price") or pos.get("avg_price") or 0.0
                positions_dict[symbol] = {"quantity": pos.get("quantity", 0.0), "current_price": mark}
            self.risk_manager.update_positions(positions_dict)
            self.risk_manager.update_pending_orders(live_orders, price_lookup=None)
        except Exception as exc:
            logger.debug(f"Risk manager update after recovery failed: {exc}")

        detail = {
            "positions_before": len(db_positions),
            "positions_after": len(live_positions or []),
            "orders_before": len(db_orders),
            "orders_after": len(live_orders or []),
        }
        self._record_health_state("restart_recovery", "ok", detail)
        bot_actions_logger.info(
            f"🧹 Startup reconciliation applied: positions {detail['positions_before']}→{detail['positions_after']}, "
            f"open orders {detail['orders_before']}→{detail['orders_after']}"
        )

    def _log_execution_trace(self, trace_id: int, execution_result: dict):
        """Attach execution outcome to LLM trace when available."""
        try:
            self.db.update_llm_trace_execution(trace_id, execution_result)
        except Exception as e:
            logger.debug(f"Could not update LLM trace {trace_id}: {e}")

    def _apply_exchange_trades_for_rebuild(self, trades: list) -> dict:
        """Delegate trade replay to portfolio tracker (kept for compatibility)."""
        self.portfolio_tracker.set_session(self.session_id)
        stats = self.portfolio_tracker.apply_exchange_trades_for_rebuild(trades)
        self.session_stats = self.portfolio_tracker.session_stats
        return stats

    def _apply_volatility_sizing(self, quantity: float, regime_flags: dict) -> float:
        """Scale quantity based on volatility regime."""
        if not regime_flags or quantity <= 0:
            return quantity
        vol_flag = regime_flags.get('volatility', '')
        if 'high' in vol_flag:
            return quantity * HIGH_VOL_SIZE_FACTOR
        if 'medium' in vol_flag:
            return quantity * MED_VOL_SIZE_FACTOR
        return quantity

    def _filter_our_orders(self, orders: list) -> list:
        """Only keep open orders with our client id prefix."""
        filtered = []
        for order in orders or []:
            client_oid = get_client_order_id(order)
            if client_oid and client_oid.startswith(CLIENT_ORDER_PREFIX):
                filtered.append(order)
        return filtered

    async def _reconcile_open_orders(self):
        """
        Refresh open order snapshot using live exchange data and drop any DB orders
        that no longer exist on the venue.
        """
        if not self.session_id:
            return
        try:
            exchange_orders = await self.bot.get_open_orders_async()
            exchange_orders = self._filter_our_orders(exchange_orders)
        except Exception as e:
            logger.warning(f"Could not fetch open orders for reconciliation: {e}")
            return

        try:
            db_orders = self.db.get_open_orders(self.session_id)
        except Exception as e:
            logger.warning(f"Could not load open orders from DB for reconciliation: {e}")
            db_orders = []

        db_ids = {str(o.get('order_id')) for o in db_orders if o.get('order_id')}
        exch_ids = {str(o.get('order_id') or o.get('id')) for o in exchange_orders if o.get('order_id') or o.get('id')}
        stale = db_ids - exch_ids
        if stale:
            bot_actions_logger.info(f"🧹 Removed {len(stale)} stale open orders not on exchange")

        try:
            self.db.replace_open_orders(self.session_id, exchange_orders)
        except Exception as e:
            logger.warning(f"Could not refresh open orders snapshot: {e}")

    def _passes_rr_filter(self, action: str, price: float, stop_price: float, target_price: float) -> bool:
        """Delegate RR filter to action handler for compatibility."""
        return self.action_handler.passes_rr_filter(action, price, stop_price, target_price, min_rr=MIN_RR)

    def _slippage_within_limit(self, decision_price: float, latest_price: float, market_data_point: dict = None):
        """Delegate slippage cap check to action handler for compatibility."""
        return self.action_handler.slippage_within_limit(
            decision_price,
            latest_price,
            market_data_point or {},
            max_slippage_pct=MAX_SLIPPAGE_PCT,
            max_spread_pct=MAX_SPREAD_PCT,
            min_top_of_book_notional=MIN_TOP_OF_BOOK_NOTIONAL,
        )

    def _prefer_maker(self, symbol: str) -> bool:
        """Determine maker intent based on overrides, else default."""
        if not symbol:
            return self.maker_preference_default
        symbol_up = symbol.upper()
        if symbol_up in self.maker_preference_overrides:
            return self.maker_preference_overrides[symbol_up]
        return self.maker_preference_default

    def _stacking_block(
        self,
        action: str,
        symbol: str,
        open_plan_count: int,
        pending_data: dict,
        position_qty: float,
    ) -> bool:
        """Delegate stacking guard to action handler for compatibility."""
        return self.action_handler.stacking_block(action, open_plan_count, pending_data, position_qty)

    async def _handle_update_plan(self, signal, telemetry_record, trace_id):
        """Handle UPDATE_PLAN intents via action handler."""
        plan_id = getattr(signal, 'plan_id', None)
        stop_price = getattr(signal, 'stop_price', None)
        target_price = getattr(signal, 'target_price', None)
        reason = getattr(signal, 'reason', '') or 'Update plan'
        result = await self.action_handler.handle_update_plan(plan_id, stop_price, target_price, reason, trace_id)
        if result is not None:
            telemetry_record.update(result)

    async def _handle_partial_close(self, signal, telemetry_record, trace_id, market_data, current_exposure):
        """Handle PARTIAL_CLOSE intents via action handler."""
        plan_id = getattr(signal, 'plan_id', None)
        close_fraction = getattr(signal, 'close_fraction', None) or 0.0
        symbol = signal.symbol
        price = market_data.get(symbol, {}).get('price') if market_data else None
        result = await self.action_handler.handle_partial_close(
            session_id=self.session_id,
            plan_id=plan_id,
            close_fraction=close_fraction,
            symbol=symbol,
            price=price,
            current_exposure=current_exposure,
            trace_id=trace_id,
        )
        if result is not None:
            telemetry_record.update(result)

    async def _handle_close_position(self, signal, telemetry_record, trace_id, market_data):
        """Handle CLOSE_POSITION intents via action handler."""
        symbol = signal.symbol
        price = market_data.get(symbol, {}).get('price') if market_data else None
        result = await self.action_handler.handle_close_position(
            session_id=self.session_id,
            symbol=symbol,
            price=price,
            trace_id=trace_id,
        )
        if result is not None:
            telemetry_record.update(result)

    async def _handle_signal(self, signal, market_data, open_orders, current_equity, current_exposure):
        """Helper for tests to exercise action handling paths."""
        # This wraps a subset of the loop logic for specific actions.
        # Build a minimal telemetry record
        telemetry_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "symbol": signal.symbol,
            "action": signal.action,
            "plan_id": getattr(signal, 'plan_id', None),
            "close_fraction": getattr(signal, 'close_fraction', None),
            "duration_minutes": getattr(signal, 'duration_minutes', None),
        }
        trace_id = getattr(signal, 'trace_id', None)
        if signal.action == 'UPDATE_PLAN':
            await self._handle_update_plan(signal, telemetry_record, trace_id)
        elif signal.action == 'PARTIAL_CLOSE':
            await self._handle_partial_close(signal, telemetry_record, trace_id, market_data, current_exposure)
        elif signal.action == 'CLOSE_POSITION':
            await self._handle_close_position(signal, telemetry_record, trace_id, market_data)
        elif signal.action == 'PAUSE_TRADING':
            duration = getattr(signal, 'duration_minutes', None) or 5
            pause_seconds = max(0, duration * 60)
            pause_until = self.health_manager.request_pause(pause_seconds)
            telemetry_record["status"] = "paused"
            telemetry_record["pause_seconds"] = pause_seconds
            telemetry_record["pause_until"] = pause_until
            self._emit_telemetry(telemetry_record)


    async def _capture_ohlcv(self, symbol: str):
        """Fetch multi-timeframe OHLCV for the active symbol and persist."""
        if not hasattr(self.bot, "fetch_ohlcv"):
            return
        timeframes = ['1m', '5m', '1h', '1d']
        now = self._monotonic()
        for tf in timeframes:
            try:
                tf_seconds = self._timeframe_to_seconds(tf)
                min_spacing = max(self.ohlcv_min_capture_spacing_seconds, tf_seconds or 0)
                last_key = (symbol, tf)
                last_capture = self._last_ohlcv_capture.get(last_key)
                if last_capture is not None and (now - last_capture) < min_spacing:
                    continue

                bars = await self.bot.fetch_ohlcv(symbol, timeframe=tf, limit=50)
                self.db.log_ohlcv_batch(self.session_id, symbol, tf, bars)
                if self.ohlcv_retention_limit:
                    try:
                        self.db.prune_ohlcv(self.session_id, symbol, tf, self.ohlcv_retention_limit)
                    except Exception as exc:
                        logger.debug(f"OHLCV prune failed for {symbol} {tf}: {exc}")
                self._last_ohlcv_capture[last_key] = now
            except Exception as e:
                logger.debug(f"OHLCV fetch failed for {symbol} {tf}: {e}")

    def _apply_order_value_buffer(self, quantity: float, price: float):
        """Delegate order value buffer to action handler for compatibility."""
        return self.action_handler.apply_order_value_buffer(quantity, price)

    def _liquidity_ok(self, market_data_point: dict) -> bool:
        """Delegate liquidity check to action handler for compatibility."""
        return self.action_handler.liquidity_ok(
            market_data_point,
            max_spread_pct=MAX_SPREAD_PCT,
            min_top_of_book_notional=MIN_TOP_OF_BOOK_NOTIONAL,
        )

    async def _monitor_trade_plans(self, price_lookup: dict, open_orders: list):
        """Delegate to the standalone PlanMonitor service."""
        config = PlanMonitorConfig(
            max_plan_age_minutes=self.max_plan_age_minutes,
            day_end_flatten_hour_utc=self.day_end_flatten_hour_utc,
            trail_to_breakeven_pct=self._apply_plan_trailing_pct,
        )
        self.plan_monitor.refresh_bindings(
            bot=self.bot,
            db=self.db,
            cost_tracker=self.cost_tracker,
            risk_manager=self.risk_manager,
            prefer_maker=self._prefer_maker,
            holdings_updater=self._update_holdings_and_realized,
            session_stats_applier=self._apply_fill_to_session_stats,
        )
        await self.plan_monitor.monitor(
            self.session_id,
            price_lookup=price_lookup,
            open_orders=open_orders,
            config=config,
        )

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

        # Initialize tool coordinator after exchange connection
        if getattr(self.bot, "exchange", None):
            self.data_fetch_coordinator = DataFetchCoordinator(
                self.bot.exchange,
                error_callback=self.health_manager.record_tool_failure,
                success_callback=self.health_manager.record_tool_success,
            )
            # Wire into strategy
            self.strategy.tool_coordinator = self.data_fetch_coordinator
        
        # Get initial equity (full account value)
        initial_equity = await self.bot.get_equity_async()
        while initial_equity is None:
            logger.warning("Could not fetch initial equity; retrying in 5s...")
            await asyncio.sleep(5)
            initial_equity = await self.bot.get_equity_async()
            
        logger.info(f"{self.exchange_name} Equity: {initial_equity}")
        
        # Create or load today's trading session (DB still used for logging/IDs)
        self.session_id = self.db.get_or_create_session(starting_balance=initial_equity, bot_version=BOT_VERSION)
        self.session = self.db.get_session(self.session_id)
        self.portfolio_tracker.set_session(self.session_id)
        # In PAPER mode, reset starting_balance baseline to current equity to avoid mismatch against old sandbox inventories
        if TRADING_MODE == 'PAPER' and self.session.get('starting_balance') != initial_equity:
            try:
                self.db.update_session_starting_balance(self.session_id, initial_equity)
                self.session['starting_balance'] = initial_equity
                logger.info(f"Sandbox Mode: Reset starting_balance to current equity ${initial_equity:,.2f}")
            except Exception as e:
                logger.warning(f"Could not reset starting_balance for sandbox: {e}")
        
        # Clear any old pending commands from previous sessions
        self.db.clear_old_commands()
        try:
            self.db.prune_commands(COMMAND_RETENTION_DAYS)
        except Exception as e:
            logger.debug(f"Could not prune commands: {e}")
        
        # Initialize trading context
        self.context = TradingContext(self.db, self.session_id)
        # Drop any stale open orders lingering from prior runs and sync with venue
        await self._reconcile_exchange_state()

        # Initialize session stats with persistence awareness
        logger.info("Initializing session stats...")
        cached_stats = self.db.get_session_stats_cache(self.session_id)
        if cached_stats:
            self.portfolio_tracker.session_stats = {
                'total_trades': cached_stats.get('total_trades', 0),
                'gross_pnl': cached_stats.get('gross_pnl', 0.0),
                'total_fees': cached_stats.get('total_fees', 0.0),
                'total_llm_cost': cached_stats.get('total_llm_cost', 0.0),
            }
            self.session_stats = self.portfolio_tracker.session_stats
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
            
            # 2. Fetch trades for the day across active symbols
            symbols = self._get_rebuild_symbols()
            trades = []
            for sym in symbols:
                sym_trades = await self.bot.get_trades_from_timestamp(sym, start_ts_ms)
                trades.extend(sym_trades)
            
            # 3. Rebuild Holdings and Stats
            self._apply_exchange_trades_for_rebuild(trades)

            # Load LLM costs from DB (since exchange doesn't track this)
            db_stats = self.db.get_session_stats(self.session_id)
            self.session_stats['total_llm_cost'] = db_stats.get('total_llm_cost', 0.0)
            self.portfolio_tracker.session_stats = self.session_stats
            self.db.set_session_stats_cache(self.session_id, self.session_stats)
            logger.info(f"Session Stats Rebuilt: {self.session_stats}")

        # Seed risk manager with persisted start-of-day equity to survive restarts
        start_equity = None
        try:
            persisted_baseline = self.db.get_start_of_day_equity(self.session_id)
        except Exception as e:
            logger.warning(f"Could not fetch persisted start-of-day equity: {e}")
            persisted_baseline = None

        if TRADING_MODE == 'PAPER':
            # In sandbox, always reset start-of-day equity to current to avoid false daily loss triggers
            start_equity = initial_equity
            logger.info(f"Sandbox Mode: Resetting start_of_day_equity to current: ${start_equity:,.2f}")
        elif persisted_baseline is not None:
            start_equity = persisted_baseline
        elif self.session and self.session.get('starting_balance') is not None:
            start_equity = self.session.get('starting_balance')
        else:
            # Prefer latest broker equity snapshot if available
            latest_equity = self.db.get_latest_equity(self.session_id)
            start_equity = latest_equity if latest_equity is not None else initial_equity

        # Persist baseline if it's new so restarts retain loss guard
        try:
            if start_equity is not None and persisted_baseline is None:
                self.db.set_start_of_day_equity(self.session_id, start_equity)
        except Exception as e:
            logger.warning(f"Could not persist start-of-day equity: {e}")

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
        """Delegate holdings/PnL updates to portfolio tracker (kept for compatibility)."""
        realized = self.portfolio_tracker.update_holdings_and_realized(symbol, action, quantity, price, fee)
        self.session_stats = self.portfolio_tracker.session_stats
        return realized

    def _apply_trade_to_holdings(self, symbol: str, action: str, quantity: float, price: float):
        """Delegate to portfolio tracker."""
        self.portfolio_tracker.apply_trade_to_holdings(symbol, action, quantity, price)
        self.session_stats = self.portfolio_tracker.session_stats

    def _load_holdings_from_db(self):
        """Rebuild holdings via portfolio tracker."""
        self.portfolio_tracker.set_session(self.session_id)
        self.portfolio_tracker.load_holdings_from_db()
        self.session_stats = self.portfolio_tracker.session_stats

    def _apply_fill_to_session_stats(self, order_id: str, actual_fee: float, realized_pnl: float):
        """Delegate session accounting to portfolio tracker (kept for compatibility)."""
        self.portfolio_tracker.set_session(self.session_id)
        self.portfolio_tracker.apply_fill_to_session_stats(order_id, actual_fee, realized_pnl, estimated_fee_map=self._estimated_fees)
        self.session_stats = self.portfolio_tracker.session_stats

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
            detail = {
                "starting_balance": starting,
                "estimated_net": estimated_net,
                "equity_net": actual_net,
                "diff": diff,
                "diff_pct": pct * 100,
            }
            if pct > 0.1:
                logger.warning(
                    f"Equity/net mismatch: equity_net={actual_net:.2f}, estimated_net={estimated_net:.2f}, "
                    f"diff={diff:.2f} ({pct*100:.1f}%)"
                )
                self._record_health_state("equity_sanity", "mismatch", detail)
            else:
                self._record_health_state("equity_sanity", "ok", detail)
        except Exception as e:
            logger.debug(f"Equity sanity check failed: {e}")

    async def _rebuild_session_stats_from_trades(self, current_equity: float = None):
        """Recompute session_stats from recorded trades and update cache via portfolio tracker."""
        self.portfolio_tracker.set_session(self.session_id)
        self.session_stats = self.portfolio_tracker.rebuild_session_stats_from_trades(current_equity)
        logger.info(f"Session stats rebuilt from trades: {self.session_stats}")
        if current_equity is not None:
            self._sanity_check_equity_vs_stats(current_equity)

    def _set_shutdown_reason(self, reason: str):
        """Keep the first shutdown reason to surface in logs."""
        if not self.shutdown_reason:
            self.shutdown_reason = reason

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
                        fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, 'SELL', liquidity=result.get('liquidity', 'taker'))
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
            self._set_shutdown_reason("flatten-all failed")
            self._kill_switch = True

    async def sync_trades_from_exchange(self):
        """Sync recent trades from exchange to DB."""
        if not self.session_id:
            return

        new_processed: set[tuple[str, str | None]] = set()
        if not self.processed_trade_ids:
            try:
                persisted_ids = self.db.get_processed_trade_ids(self.session_id)
                self.processed_trade_ids.update(persisted_ids or set())
            except Exception as exc:
                logger.debug(f"Could not load processed trade ids: {exc}")

        try:
            symbols = set()
            try:
                symbols.update(self.db.get_distinct_trade_symbols(self.session_id) or [])
                symbols.update({p.get('symbol') for p in self.db.get_positions(self.session_id) or [] if p.get('symbol')})
                symbols.update({p.get('symbol') for p in self.db.get_open_trade_plans(self.session_id) or [] if p.get('symbol')})
                symbols.update({o.get('symbol') for o in self.db.get_open_orders(self.session_id) or [] if o.get('symbol')})
            except Exception:
                pass
            symbols = {s for s in symbols if isinstance(s, str) and '/' in s}
            if not symbols:
                symbols = {'BTC/USD'}

            since_iso = self.db.get_latest_trade_timestamp(self.session_id)
            since_ms = None
            if since_iso:
                try:
                    since_dt = datetime.fromisoformat(since_iso)
                    since_ms = int(since_dt.timestamp() * 1000) - 5000
                except Exception:
                    since_ms = None
            cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=TRADE_SYNC_CUTOFF_MINUTES)

            for symbol in symbols:
                cursor_since = since_ms
                while True:
                    trades = await self.bot.get_my_trades_async(symbol, since=cursor_since, limit=100)
                    filtered_trades = []
                    for t in trades:
                        client_oid = get_client_order_id(t)
                        if not client_oid:
                            continue
                        if not client_oid.startswith(CLIENT_ORDER_PREFIX):
                            continue
                        t["_client_oid"] = client_oid
                        filtered_trades.append(t)
                    trades = filtered_trades

                    if not trades:
                        break

                    latest_ts = None
                    for t in trades:
                        client_oid = t.get('_client_oid') or get_client_order_id(t)
                        trade_id = str(t['id'])
                        if trade_id in self.processed_trade_ids:
                            continue

                        existing = self.db.conn.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,)).fetchone()
                        if existing:
                            self.processed_trade_ids.add(trade_id)
                            new_processed.add((trade_id, client_oid))
                            continue

                        ts_ms = t.get('timestamp')
                        if ts_ms:
                            trade_dt = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
                            if trade_dt < cutoff_dt:
                                self.processed_trade_ids.add(trade_id)
                                new_processed.add((trade_id, client_oid))
                                continue

                        order_id = t.get('order')
                        side = t['side'].upper()
                        price = t['price']
                        quantity = t['amount']
                        fee = t.get('fee', {}).get('cost', 0.0)

                        info = t.get('info') or {}
                        liquidity = (
                            t.get('liquidity')
                            or info.get('liquidity')
                            or info.get('fillLiquidity')
                            or info.get('liquidityIndicator')
                            or 'unknown'
                        )
                        if liquidity:
                            liquidity = str(liquidity).lower()

                        plan_reason = None
                        try:
                            plan_reason = self.db.get_trade_plan_reason_by_order(self.session_id, order_id, client_oid)
                        except Exception:
                            plan_reason = None
                        reason = self.order_reasons.get(str(order_id)) or plan_reason
                        if not reason:
                            # Skip trades we cannot attribute to our client IDs/reasons
                            self.processed_trade_ids.add(trade_id)
                            new_processed.add((trade_id, client_oid))
                            continue

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
                            trade_id=trade_id,
                            timestamp=t.get('datetime')
                        )
                        self._apply_fill_to_session_stats(order_id, fee, realized_pnl)
                        self.processed_trade_ids.add(trade_id)
                        new_processed.add((trade_id, client_oid))
                        bot_actions_logger.info(f"✅ Synced trade: {side} {quantity} {symbol} @ ${price:,.2f} (Fee: ${fee:.4f})")

                        ts = t.get('timestamp')
                        if ts is not None:
                            latest_ts = max(latest_ts or ts, ts)

                    if latest_ts is None or len(trades) < 100:
                        break
                    cursor_since = latest_ts + 1

        except Exception as e:
            logger.exception(f"Error syncing trades: {e}")
        finally:
            if new_processed:
                try:
                    self.db.record_processed_trade_ids(self.session_id, new_processed)
                except Exception as exc:
                    logger.debug(f"Could not persist processed trade ids: {exc}")

    async def cleanup(self):
        """Cleanup and close connection."""
        logger.info(f"Cleaning up connections... (reason: {self.shutdown_reason or 'unspecified'})")
        bot_actions_logger.info(f"🧹 Cleanup starting (reason: {self.shutdown_reason or 'unspecified'})")
        
        # Save final session statistics
        if self.session_id:
            try:
                # Get final equity snapshot
                final_equity = await self.bot.get_equity_async()
                
                # Get session stats and rebuild to ensure consistency
                session_stats = self.db.get_session_stats(self.session_id)
                try:
                    await self._rebuild_session_stats_from_trades(final_equity)
                    session_stats = self.db.get_session_stats(self.session_id)
                except Exception as e:
                    logger.debug(f"Could not rebuild stats on cleanup: {e}")
                
                # Calculate PnL using fee-exclusive realized and separate fees
                gross_pnl = session_stats.get('gross_pnl', 0.0) or 0.0
                net_pnl = gross_pnl - (session_stats.get('total_fees', 0.0) or 0.0) - (session_stats.get('total_llm_cost', 0.0) or 0.0)
                equity_delta = final_equity - session_stats['starting_balance']
                
                # Update database
                self.db.update_session_balance(self.session_id, final_equity, net_pnl)
                
                # Log summary to bot.log
                bot_actions_logger.info("=" * 50)
                bot_actions_logger.info("📊 SESSION SUMMARY")
                bot_actions_logger.info("=" * 50)
                bot_actions_logger.info(f"Total Trades: {session_stats['total_trades']}")
                bot_actions_logger.info(f"Gross PnL (ex-fee): ${gross_pnl:,.2f}")
                bot_actions_logger.info(f"Trading Fees: ${session_stats['total_fees']:.2f}")
                bot_actions_logger.info(f"LLM Costs: ${session_stats['total_llm_cost']:.4f}")
                bot_actions_logger.info(f"Net PnL: ${net_pnl:,.2f}")
                bot_actions_logger.info(f"Equity Delta (broker): ${equity_delta:,.2f}")
                
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

    async def run_loop(self, max_loops: int | None = None):
        """Main autonomous loop."""
        try:
            await self.initialize()
            self.running = True
            loops = 0
            
            while self.running:
                if self._kill_switch:
                    if not self.shutdown_reason:
                        self._set_shutdown_reason("kill switch")
                    bot_actions_logger.info("🛑 Kill switch active; exiting main loop.")
                    self.running = False
                    break
                try:
                    if max_loops is not None and loops >= max_loops:
                        break
                    exchange_error_seen = False
                    # 0. Check for pending commands from dashboard
                    command_result = await self.command_processor.process(
                        close_positions_cb=self._close_all_positions_safely,
                        stop_cb=self._set_shutdown_reason,
                    )
                    if command_result.stop_requested:
                        self.running = False
                        if command_result.shutdown_reason and not self.shutdown_reason:
                            self.shutdown_reason = command_result.shutdown_reason
                        break
                    
                    # 1. Update Equity / PnL
                    try:
                        current_equity = await self.bot.get_equity_async()
                    except Exception as e:
                        logger.warning(f"Could not fetch equity: {e}; skipping loop iteration.")
                        self.health_manager.record_exchange_failure("get_equity_async", e)
                        exchange_error_seen = True
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue
                    
                    if current_equity is None:
                        logger.warning("Could not fetch equity; skipping loop iteration to avoid false loss triggers.")
                        self.health_manager.record_exchange_failure("get_equity_async", "none")
                        exchange_error_seen = True
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

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
                            self._set_shutdown_reason(f"daily loss {loss_percent:.2f}% > {limit_pct}%")
                            bot_actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded ({loss_percent:.2f}%)")
                            # Attempt to flatten positions before stopping
                            await self._close_all_positions_safely()
                            self._kill_switch = True
                            break
                    # Check absolute daily loss
                    if self.risk_manager.daily_loss > MAX_DAILY_LOSS:
                        if TRADING_MODE == 'PAPER':
                            # In sandbox, large balances make absolute loss limits ($500) too tight.
                            # We rely on the percentage check above for safety.
                            logger.warning(f"Sandbox: Absolute daily loss exceeded (${self.risk_manager.daily_loss:.2f} > ${MAX_DAILY_LOSS:.2f}), but continuing loop.")
                        else:
                            logger.error(f"Max daily loss exceeded: ${self.risk_manager.daily_loss:.2f} > ${MAX_DAILY_LOSS:.2f}. Stopping loop.")
                            self._set_shutdown_reason(f"daily loss ${self.risk_manager.daily_loss:.2f} > ${MAX_DAILY_LOSS:.2f}")
                            bot_actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded (${self.risk_manager.daily_loss:.2f})")
                            await self._close_all_positions_safely()
                            self._kill_switch = True
                            break

                    # 2. Fetch Market Data
                    now_monotonic = asyncio.get_event_loop().time()
                    if self.health_manager.should_pause(now_monotonic):
                        remaining = self.health_manager.pause_remaining(now_monotonic)
                        bot_actions_logger.info(f"⏸️ Trading paused for {remaining:.0f}s")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    symbols = self._get_active_symbols()
                    market_data = {}
                    primary_symbol = symbols[0]
                    primary_fetch_failed = False

                    for sym in symbols:
                        ticker_started = self._monotonic()
                        try:
                            md = await self.bot.get_market_data_async(sym)
                        except Exception as e:
                            logger.warning(f"Market data fetch failed for {sym}: {e}")
                            self.health_manager.record_exchange_failure("get_market_data_async", e)
                            exchange_error_seen = True
                            if sym == primary_symbol:
                                primary_fetch_failed = True
                                break
                            continue
                        ticker_ended = self._monotonic()
                        if md is not None:
                            md["_latency_ms"] = (ticker_ended - ticker_started) * 1000
                            md["_fetched_monotonic"] = ticker_ended
                            md["fetched_at"] = datetime.now(timezone.utc).isoformat()
                        market_data[sym] = md

                        # Log market data to database
                        if md and self.session_id:
                            try:
                                self.db.log_market_data(
                                    self.session_id,
                                    sym,
                                    md.get('price'),
                                    md.get('bid'),
                                    md.get('ask'),
                                    md.get('volume') or 0.0,
                                    spread_pct=md.get('spread_pct'),
                                    bid_size=md.get('bid_size'),
                                    ask_size=md.get('ask_size'),
                                    ob_imbalance=md.get('ob_imbalance'),
                                )
                            except Exception as e:
                                logger.warning(f"Could not log market data: {e}")
                    if self.session_id:
                        try:
                            self.db.prune_market_data(self.session_id, MARKET_DATA_RETENTION_MINUTES)
                        except Exception as e:
                            logger.debug(f"Could not prune market data: {e}")

                    if primary_fetch_failed or not market_data.get(primary_symbol):
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    primary_data = market_data.get(primary_symbol)
                    stale, freshness_detail = self.health_manager.is_stale_market_data(primary_data)
                    if stale:
                        bot_actions_logger.info("⏸️ Skipping loop: market data stale or too latent")
                        self._record_health_state("market_data", "stale", freshness_detail)
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue
                    else:
                        self._record_health_state("market_data", "ok", freshness_detail)

                    # Capture multi-timeframe OHLCV for richer context (primary symbol only)
                    try:
                        await self._capture_ohlcv(primary_symbol)
                    except Exception as e:
                        logger.debug(f"Could not capture OHLCV: {e}")

                    # Microstructure filter: skip trading when spread/liquidity are poor (primary symbol)
                    if primary_data and not self._liquidity_ok(primary_data):
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    # Refresh live positions each loop for accurate exposure snapshots
                    try:
                        live_positions = await self.bot.get_positions_async()
                        self.db.replace_positions(self.session_id, live_positions)
                    except Exception as e:
                        logger.warning(f"Could not refresh positions: {e}")
                        self.health_manager.record_exchange_failure("get_positions_async", e)
                        exchange_error_seen = True
                    # Refresh open orders for exposure headroom and context
                    open_orders = []
                    try:
                        open_orders = await self.bot.get_open_orders_async()
                        open_orders = self._filter_our_orders(open_orders)
                        self.db.replace_open_orders(self.session_id, open_orders)
                    except Exception as e:
                        logger.warning(f"Could not refresh open orders: {e}")
                        self.health_manager.record_exchange_failure("get_open_orders_async", e)
                        exchange_error_seen = True
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
                    price_lookup = {}
                    try:
                        positions_data = self.db.get_positions(self.session_id)
                        for pos in positions_data:
                            sym = pos['symbol']
                            current_price = pos.get('current_price') or pos.get('avg_price') or 0

                            # Prefer most recent market tick
                            recent_data = self.db.get_recent_market_data(self.session_id, sym, limit=1)
                            if recent_data and recent_data[0].get('price'):
                                current_price = recent_data[0]['price']

                            # Prefer live price when we have it
                            if market_data.get(sym) and market_data[sym].get('price'):
                                current_price = market_data[sym]['price']

                            if current_price:
                                positions_dict[sym] = {
                                    'quantity': pos['quantity'],
                                    'current_price': current_price
                                }
                        self.risk_manager.update_positions(positions_dict)

                        # Build price lookup for open orders (fallback to recent tick)
                        for sym, md in market_data.items():
                            if md and md.get('price'):
                                price_lookup[sym] = md['price']
                        for ord in open_orders or []:
                            sym = ord.get('symbol')
                            if sym and sym in price_lookup:
                                continue
                            latest = self.db.get_recent_market_data(self.session_id, sym, limit=1) if sym else None
                            if latest and latest[0].get('price'):
                                price_lookup[sym] = latest[0]['price']
                        self.risk_manager.update_pending_orders(open_orders, price_lookup=price_lookup)

                        price_overrides = {sym: md.get('price') for sym, md in market_data.items() if md and md.get('price')}
                        price_overrides = price_overrides or None
                        current_exposure = self.risk_manager.get_total_exposure(price_overrides=price_overrides)
                        self._record_operational_metrics(current_exposure, current_equity)
                    except Exception as e:
                        logger.warning(f"Could not build positions for exposure: {e}")

                    # 2.8 Monitor trade plans for stops/targets and max age
                    try:
                        await self._monitor_trade_plans(price_lookup=price_lookup, open_orders=open_orders)
                    except Exception as e:
                        logger.exception(f"Trade plan monitor error: {e}")

                    # Kill switch check
                    if self._kill_switch:
                        if not self.shutdown_reason:
                            self._set_shutdown_reason("kill switch")
                        bot_actions_logger.info("🛑 Kill switch active; exiting main loop.")
                        self.running = False
                        break

                    # Slippage guard: if latest price moved >2% from decision price, skip execution
                    decision_price = market_data.get(primary_symbol, {}).get('price')

                    # 2.5 Sync Trades from Exchange (for logging only)
                    try:
                        await self.sync_trades_from_exchange()
                    except Exception as e:
                        logger.warning(f"Trade sync failed: {e}")
                        self.health_manager.record_exchange_failure("sync_trades_from_exchange", e)
                        exchange_error_seen = True
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
                        trace_id = getattr(signal, 'trace_id', None)
                        regime_flags = getattr(signal, 'regime_flags', {}) or {}
                        telemetry_record = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "session_id": self.session_id,
                            "exchange": self.exchange_name,
                            "symbol": symbol,
                            "action": action,
                            "quantity": quantity,
                            "reason": reason,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "plan_id": getattr(signal, 'plan_id', None),
                            "size_factor": getattr(signal, 'size_factor', None),
                            "close_fraction": getattr(signal, 'close_fraction', None),
                            "equity": current_equity,
                            "exposure": current_exposure,
                            "open_orders": len(open_orders) if open_orders is not None else None,
                            "trace_id": trace_id,
                            "regime_flags": regime_flags,
                        }
                        
                        # Log decision to user-friendly log
                        if action == 'HOLD':
                            bot_actions_logger.info(f"📊 Decision: HOLD - {reason}")
                            telemetry_record["status"] = "hold"
                            self._log_execution_trace(trace_id, {"status": "hold", "reason": reason})
                            self._emit_telemetry(telemetry_record)
                            continue
                        
                        elif action == 'CANCEL':
                            if not order_id:
                                logger.warning("Skipped cancel: missing order_id")
                                telemetry_record["status"] = "cancel_missing_id"
                                telemetry_record["error"] = "missing order_id"
                                self._log_execution_trace(trace_id, {"status": "cancel_missing_id"})
                                self._emit_telemetry(telemetry_record)
                                continue
                            cancel_id = order_id
                            if isinstance(order_id, str) and order_id.isdigit():
                                cancel_id = int(order_id)
                            try:
                                success = await self.bot.cancel_open_order_async(cancel_id)
                                if success:
                                    bot_actions_logger.info(f"🛑 Cancelled order {order_id}: {reason}")
                                    telemetry_record["status"] = "cancelled"
                                else:
                                    bot_actions_logger.info(f"⚠️ Cancel request failed for order {order_id}: {reason}")
                                    telemetry_record["status"] = "cancel_failed"
                                # Refresh open orders snapshot so strategy context stays current
                                try:
                                    open_orders = await self.bot.get_open_orders_async()
                                    self.db.replace_open_orders(self.session_id, open_orders)
                                except Exception as e:
                                    logger.warning(f"Could not refresh open orders after cancel: {e}")
                            except Exception as e:
                                logger.error(f"Cancel order error: {e}")
                                telemetry_record["status"] = "cancel_error"
                                telemetry_record["error"] = str(e)
                            self._log_execution_trace(trace_id, telemetry_record)
                            self._emit_telemetry(telemetry_record)
                            continue

                        elif action == 'UPDATE_PLAN':
                            await self._handle_update_plan(signal, telemetry_record, trace_id)
                            continue

                        elif action == 'PARTIAL_CLOSE':
                            await self._handle_partial_close(signal, telemetry_record, trace_id, market_data, current_exposure)
                            continue

                        elif action == 'CLOSE_POSITION':
                            await self._handle_close_position(signal, telemetry_record, trace_id, market_data)
                            continue

                        elif action == 'PAUSE_TRADING':
                            duration = getattr(signal, 'duration_minutes', None) or 5
                            pause_seconds = max(0, duration * 60)
                            pause_until = self.health_manager.request_pause(pause_seconds)
                            telemetry_record["status"] = "paused"
                            telemetry_record["pause_seconds"] = pause_seconds
                            telemetry_record["pause_until"] = pause_until
                            self._emit_telemetry(telemetry_record)
                            bot_actions_logger.info(f"⏸️ Trading paused for {pause_seconds/60:.1f} minutes by LLM request")
                            continue

                        elif action in ['BUY', 'SELL'] and quantity > 0:
                            # Get price for risk checks and execution
                            md = market_data.get(symbol)
                            price = md.get('price') if md else None
                            if price is None:
                                price = price_lookup.get(symbol)
                            
                            if not price:
                                logger.warning("Skipped trade: missing price data")
                                continue

                        # Volatility sizing adjustment
                        adjusted_quantity = self._apply_volatility_sizing(quantity, regime_flags)
                        telemetry_record["vol_scaled_qty"] = adjusted_quantity

                        # Guardrails: clip size to sit under the max order cap minus buffer
                        quantity = self._apply_order_value_buffer(adjusted_quantity, price)

                        if quantity <= 0:
                            logger.warning("Skipped trade: buffered quantity became non-positive")
                            continue

                        # Format quantity appropriately (show more decimals for small amounts) after buffering
                        if quantity < 1:
                            qty_str = f"{quantity:.6f}".rstrip('0').rstrip('.')
                        else:
                            qty_str = f"{quantity:.4f}".rstrip('0').rstrip('.')
                        bot_actions_logger.info(f"📊 Decision: {action} {qty_str} {symbol} - {reason}")

                        prefer_maker = self._prefer_maker(symbol)
                        telemetry_record["prefer_maker"] = prefer_maker

                        risk_result = self.risk_manager.check_trade_allowed(symbol, action, quantity, price)
                        telemetry_record["risk_allowed"] = risk_result.allowed
                        telemetry_record["risk_reason"] = risk_result.reason

                        if risk_result.allowed:
                            # RR filter when stop/target provided
                            if not self._passes_rr_filter(action, price, stop_price, target_price):
                                bot_actions_logger.info(f"⛔ Trade Blocked: RR below {MIN_RR}")
                                telemetry_record["status"] = "rr_blocked"
                                self._log_execution_trace(trace_id, telemetry_record)
                                self._emit_telemetry(telemetry_record)
                                self.strategy.on_trade_rejected("RR below threshold")
                                continue

                            # Calculate fee before execution (estimate)
                            liquidity_hint = "maker" if prefer_maker else "taker"
                            estimated_fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, action, liquidity=liquidity_hint)
                            liquidity = "maker_intent" if prefer_maker else "taker_intent"
                            # Capture stop/target if provided by strategy
                            stop_price = getattr(signal, 'stop_price', None)
                            target_price = getattr(signal, 'target_price', None)

                            # Enforce per-symbol plan cap before placing order
                            try:
                                open_plan_count = self.db.count_open_trade_plans_for_symbol(self.session_id, symbol)
                                if open_plan_count >= self.max_plans_per_symbol:
                                    if AUTO_REPLACE_PLAN_ON_CAP:
                                        try:
                                            plans = self.db.get_open_trade_plans(self.session_id)
                                            candidates = [p for p in plans if p.get('symbol') == symbol]
                                            if candidates:
                                                victim = sorted(candidates, key=lambda p: p.get('opened_at'))[0]
                                                self.db.update_trade_plan_status(victim['id'], status='cancelled', closed_at=datetime.now(timezone.utc).isoformat(), reason="auto_replace_on_cap")
                                                bot_actions_logger.info(f"♻️ Replaced plan {victim['id']} to make room for new {action} on {symbol}")
                                            else:
                                                bot_actions_logger.info(f"♻️ Auto-replace: no candidate plan found for {symbol}")
                                        except Exception as e:
                                            logger.warning(f"Auto-replace plan failed: {e}")
                                            bot_actions_logger.info(f"⛔ Trade Blocked: plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                            self.strategy.on_trade_rejected(f"Plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                            continue
                                    else:
                                        bot_actions_logger.info(f"⛔ Trade Blocked: plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                        self.strategy.on_trade_rejected(f"Plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                        continue
                            except Exception as e:
                                logger.debug(f"Could not check plan cap: {e}")
                            # Enforce pending exposure headroom including open orders, stacking, and per-symbol caps
                            try:
                                pending_data = self.risk_manager.pending_orders_by_symbol.get(symbol, {})
                                pending_exposure = pending_data.get('buy', 0.0) if action == 'BUY' else pending_data.get('sell', 0.0)
                                pending_count = pending_data.get('count_buy', 0) if action == 'BUY' else pending_data.get('count_sell', 0)
                                if action == 'BUY' and pending_count >= self.max_plans_per_symbol:
                                    bot_actions_logger.info(f"⛔ Trade Blocked: open order count reached for {symbol} ({pending_count}/{self.max_plans_per_symbol})")
                                    self.strategy.on_trade_rejected("Open order cap reached")
                                    telemetry_record["status"] = "risk_blocked"
                                    telemetry_record["risk_reason"] = "Open order cap reached"
                                    self._log_execution_trace(trace_id, telemetry_record)
                                    self._emit_telemetry(telemetry_record)
                                    continue
                                position_qty = (self.risk_manager.positions or {}).get(symbol, {}).get('quantity', 0.0) or 0.0
                                if self._stacking_block(action, symbol, open_plan_count, pending_data, position_qty):
                                    bot_actions_logger.info(f"⛔ Trade Blocked: stacking same-side risk on {symbol}")
                                    self.strategy.on_trade_rejected("Stacking blocked")
                                    telemetry_record["status"] = "risk_blocked"
                                    telemetry_record["risk_reason"] = "Stacking blocked"
                                    self._log_execution_trace(trace_id, telemetry_record)
                                    self._emit_telemetry(telemetry_record)
                                    continue
                                order_value = quantity * price
                                if action == 'BUY' and (pending_exposure + order_value + current_exposure) > MAX_TOTAL_EXPOSURE:
                                    bot_actions_logger.info("⛔ Trade Blocked: pending/open exposure would exceed cap")
                                    self.strategy.on_trade_rejected("Pending exposure over cap")
                                    telemetry_record["status"] = "risk_blocked"
                                    telemetry_record["risk_reason"] = "Pending exposure over cap"
                                    self._log_execution_trace(trace_id, telemetry_record)
                                    self._emit_telemetry(telemetry_record)
                                    continue
                            except Exception as e:
                                logger.debug(f"Pending exposure check failed: {e}")

                            # Slippage guard: refresh price and compare vs decision snapshot
                            try:
                                latest_md = await self.bot.get_market_data_async(symbol)
                                latest_price = latest_md.get('price') if latest_md else price
                            except Exception:
                                latest_price = price

                            ok_slip, move_pct = self._slippage_within_limit(price, latest_price, latest_md or md)
                            telemetry_record["slippage_pct"] = move_pct
                            if not ok_slip:
                                bot_actions_logger.info(f"⏸️ Skipping trade: slippage {move_pct:.2f}% > {MAX_SLIPPAGE_PCT:.2f}%")
                                telemetry_record["status"] = "slippage_blocked"
                                self._log_execution_trace(trace_id, telemetry_record)
                                self._emit_telemetry(telemetry_record)
                                self.strategy.on_trade_rejected("Slippage over limit")
                                continue

                            bot_actions_logger.info(f"✅ Executing: {action} {qty_str} {symbol} at ${price:,.2f} (est. fee: ${estimated_fee:.4f})")

                            if not self.execute_orders:
                                bot_actions_logger.info("👁️ Shadow mode: skipping live order placement")
                                continue
                            
                            # Execute trade
                            retries = 0
                            order_result = None
                            while retries < 2 and order_result is None:
                                try:
                                    order_result = await asyncio.wait_for(
                                        self.bot.place_order_async(symbol, action, quantity, prefer_maker=prefer_maker),
                                        timeout=15
                                    )
                                except asyncio.TimeoutError:
                                    logger.error("Order placement timed out")
                                    self.health_manager.record_exchange_failure("place_order_async", "timeout")
                                    exchange_error_seen = True
                                    await self.health_manager.maybe_reconnect(self.bot)
                                    retries += 1
                                except Exception as e:
                                    logger.error(f"Order placement error: {e}")
                                    self.health_manager.record_exchange_failure("place_order_async", e)
                                    exchange_error_seen = True
                                    await self.health_manager.maybe_reconnect(self.bot)
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
                                        reason,
                                        entry_order_id=order_result.get('order_id') if order_result else None,
                                        entry_client_order_id=order_result.get('client_order_id') if order_result else None
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

                            telemetry_record["status"] = order_result.get('status') if isinstance(order_result, dict) else "order_unknown"
                            telemetry_record["order_result"] = order_result
                            self._log_execution_trace(trace_id, telemetry_record)
                            self._emit_telemetry(telemetry_record)

                        else:
                            telemetry_record["status"] = "risk_blocked"
                            bot_actions_logger.info(f"⛔ Trade Blocked: {risk_result.reason}")
                            self.strategy.on_trade_rejected(risk_result.reason)
                            self._log_execution_trace(trace_id, telemetry_record)
                            self._emit_telemetry(telemetry_record)

                    if not exchange_error_seen:
                        self.health_manager.reset_exchange_errors()
                    # 5. Sleep
                    logger.info(f"Sleeping for {LOOP_INTERVAL_SECONDS} seconds...")
                    await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                    loops += 1

                except KeyboardInterrupt:
                    logger.info("Stopping loop...")
                    self._set_shutdown_reason("KeyboardInterrupt")
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
        runner._set_shutdown_reason(f"signal {sig.name if hasattr(sig, 'name') else sig}")
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
