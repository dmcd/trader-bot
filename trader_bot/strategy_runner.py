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
    MAX_ORDER_VALUE,
    MAX_POSITIONS,
    MAX_SLIPPAGE_PCT,
    MAX_SPREAD_PCT,
    MAX_TOTAL_EXPOSURE,
    MED_VOL_SIZE_FACTOR,
    MIN_RR,
    MIN_TRADE_SIZE,
    MIN_TOP_OF_BOOK_NOTIONAL,
    LLM_MAX_PORTFOLIO_COST,
    ORDER_VALUE_BUFFER,
    LLM_PROVIDER,
    PLAN_MAX_AGE_MINUTES,
    PLAN_MAX_PER_SYMBOL,
    PLAN_OVERNIGHT_MAX_WIDEN_PCT,
    PLAN_OVERNIGHT_WIDEN_ABS,
    PLAN_OVERNIGHT_WIDEN_ENABLED,
    PLAN_OVERNIGHT_WIDEN_PCT,
    PLAN_AUTO_REARM_ON_RESTART,
    PLAN_TRAIL_TO_BREAKEVEN_PCT,
    PRIORITY_LOOKBACK_MIN,
    PRIORITY_MOVE_PCT,
    SANDBOX_IGNORE_INITIAL_POSITIONS,
    IB_BASE_CURRENCY,
    BOT_VERSION,
    PORTFOLIO_BASE_CURRENCY,
    PORTFOLIO_DAY_TIMEZONE,
    PORTFOLIO_NAME,
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
    IB_ALLOWED_INSTRUMENT_TYPES,
    IB_EQUITY_MAX_SPREAD_PCT,
    IB_EQUITY_MIN_QUOTE_SIZE,
    IB_EQUITY_MIN_TOP_OF_BOOK_NOTIONAL,
    IB_FX_MAX_SPREAD_PCT,
    IB_FX_MIN_TOP_OF_BOOK_NOTIONAL,
)
from trader_bot.cost_tracker import CostTracker
from trader_bot.data_fetch_coordinator import DataFetchCoordinator
from trader_bot.database import TradingDatabase
from trader_bot.gemini_trader import GeminiTrader
from trader_bot.ib_trader import IBTrader
from trader_bot.logger_config import setup_logging, set_logging_context
from trader_bot.risk_manager import RiskManager
from trader_bot.strategy import LLMStrategy
from trader_bot.services.command_processor import CommandProcessor
from trader_bot.services.portfolio_tracker import PortfolioTracker
from trader_bot.services.plan_monitor import PlanMonitor, PlanMonitorConfig
from trader_bot.services.health_manager import HealthCircuitManager
from trader_bot.services.trade_action_handler import TradeActionHandler
from trader_bot.services.market_data_service import MarketDataService
from trader_bot.services.resync_service import ResyncService
from trader_bot.services.strategy_orchestrator import StrategyOrchestrator
from trader_bot.technical_analysis import TechnicalAnalysis
from trader_bot.trading_context import TradingContext
from trader_bot.utils import get_client_order_id
from trader_bot.symbols import infer_instrument_type, normalize_symbol

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
        if ACTIVE_EXCHANGE == 'GEMINI':
            self.bot = GeminiTrader()
            self.exchange_name = 'GEMINI'
        elif ACTIVE_EXCHANGE == 'IB':
            self.bot = IBTrader()
            self.exchange_name = 'IB'
        else:
            # Default to Gemini if not specified or invalid
            logger.warning(f"Unknown ACTIVE_EXCHANGE '{ACTIVE_EXCHANGE}', defaulting to GEMINI")
            self.bot = GeminiTrader()
            self.exchange_name = 'GEMINI'
        self.sandbox_ignore_positions = TRADING_MODE == 'PAPER' and SANDBOX_IGNORE_INITIAL_POSITIONS
        self._sandbox_position_baseline = {}
        resolved_base_currency = PORTFOLIO_BASE_CURRENCY
        if self.exchange_name == 'IB' and IB_BASE_CURRENCY:
            resolved_base_currency = IB_BASE_CURRENCY
        self.base_currency = resolved_base_currency
        fx_rate_provider = getattr(self.bot, "get_cached_fx_rate", None)
        self.risk_manager = RiskManager(
            self.bot,
            ignore_baseline_positions=self.sandbox_ignore_positions,
            base_currency=resolved_base_currency,
            fx_rate_provider=fx_rate_provider,
        )
        self.running = False
        self.execute_orders = execute_orders
        
        # Professional trading infrastructure
        self.db = TradingDatabase()
        self.portfolio_id, self.run_id = self.db.ensure_active_portfolio(
            name=PORTFOLIO_NAME,
            base_currency=self.base_currency,
            bot_version=BOT_VERSION,
        )
        set_logging_context(self.portfolio_id, self.run_id)
        self.portfolio = self.db.get_portfolio(self.portfolio_id)
        portfolio_base_currency = (self.portfolio or {}).get("base_currency") or self.base_currency
        self.base_currency = portfolio_base_currency
        self.risk_manager.set_portfolio(self.portfolio_id)
        self.risk_manager.set_base_currency(portfolio_base_currency, fx_rate_provider=fx_rate_provider)
        self.cost_tracker = CostTracker(self.exchange_name, llm_provider=LLM_PROVIDER)
        self.technical_analysis = TechnicalAnalysis()
        self.context = None
        self.starting_equity: float | None = None
        self._equity_baseline_ts: str | None = None
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
        self.portfolio_tracker = PortfolioTracker(self.db, portfolio_id=self.portfolio_id, logger=logger)
        self.holdings = self.portfolio_tracker.holdings
        self.market_data_service = MarketDataService(
            db=self.db,
            bot=self.bot,
            portfolio_id=self.portfolio_id,
            monotonic=self._monotonic,
            ohlcv_min_capture_spacing_seconds=OHLCV_MIN_CAPTURE_SPACING_SECONDS,
            ohlcv_retention_limit=OHLCV_MAX_ROWS_PER_TIMEFRAME,
            logger=logger,
        )
        self.resync_service = ResyncService(
            db=self.db,
            bot=self.bot,
            risk_manager=self.risk_manager,
            holdings_updater=self._update_holdings_and_realized,
            portfolio_stats_applier=self._apply_fill_to_portfolio_stats,
            record_health_state=self._record_health_state,
            logger=logger,
            portfolio_id=self.portfolio_id,
        )
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
            portfolio_id=self.portfolio_id,
        )
        self.maker_preference_default = MAKER_PREFERENCE_DEFAULT
        self.maker_preference_overrides = MAKER_PREFERENCE_OVERRIDES or {}
        # Seed a default stats container so background tasks don't crash before initialization completes
        self.portfolio_stats = self.portfolio_tracker.portfolio_stats
        
        # Initialize Strategy
        self.strategy = LLMStrategy(
            self.db, 
            self.technical_analysis, 
            self.cost_tracker,
            open_orders_provider=self.bot.get_open_orders_async,
            ohlcv_provider=self.bot.fetch_ohlcv,
            tool_coordinator=None,  # set post-connect when exchange is ready
            portfolio_id=self.portfolio_id,
            run_id=self.run_id,
        )
        # Wire action handler rejection callback once strategy exists
        self.action_handler.on_trade_rejected = getattr(self.strategy, "on_trade_rejected", None)
        
        # Trade syncing state
        self.order_reasons = {}  # order_id -> reason
        self.processed_trade_ids = set()
        self._open_trade_plans = {}  # plan_id -> dict
        self.max_plan_age_minutes = PLAN_MAX_AGE_MINUTES
        self.max_plans_per_symbol = PLAN_MAX_PER_SYMBOL
        self.telemetry_logger = telemetry_logger
        self._apply_plan_trailing_pct = PLAN_TRAIL_TO_BREAKEVEN_PCT  # move stop to entry after move in favor
        self.plan_overnight_widen_enabled = PLAN_OVERNIGHT_WIDEN_ENABLED
        self.plan_overnight_widen_pct = PLAN_OVERNIGHT_WIDEN_PCT
        self.plan_overnight_widen_abs = PLAN_OVERNIGHT_WIDEN_ABS
        self.plan_overnight_max_widen_pct = PLAN_OVERNIGHT_MAX_WIDEN_PCT
        self.plan_auto_rearm_on_restart = PLAN_AUTO_REARM_ON_RESTART
        self.shutdown_reason: str | None = None  # track why we stop
        self.ohlcv_min_capture_spacing_seconds = OHLCV_MIN_CAPTURE_SPACING_SECONDS
        self.ohlcv_retention_limit = OHLCV_MAX_ROWS_PER_TIMEFRAME
        self.command_processor = CommandProcessor(self.db)
        self.plan_monitor = self._build_plan_monitor()
        self.orchestrator = StrategyOrchestrator(
            command_processor=self.command_processor,
            plan_monitor=self.plan_monitor,
            risk_manager=self.risk_manager,
            health_manager=self.health_manager,
            record_operational_metrics=self._record_operational_metrics,
            loop_interval_seconds=LOOP_INTERVAL_SECONDS,
            logger=logger,
            actions_logger=bot_actions_logger,
            portfolio_id=self.portfolio_id,
        )

    def _emit_telemetry(self, record: dict):
        """Emit structured telemetry as JSON line."""
        if not self.telemetry_logger:
            return
        try:
            record.setdefault("run_id", self.run_id)
            record.setdefault("portfolio_id", self.portfolio_id)
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
        """Delegate timeframe parsing to market data service (compatibility wrapper)."""
        return self.market_data_service.timeframe_to_seconds(timeframe)

    def _build_plan_monitor(self) -> PlanMonitor:
        """Construct a plan monitor bound to the current dependencies."""
        return PlanMonitor(
            db=self.db,
            bot=self.bot,
            cost_tracker=self.cost_tracker,
            risk_manager=self.risk_manager,
            prefer_maker=self._prefer_maker,
            holdings_updater=self._update_holdings_and_realized,
            portfolio_stats_applier=self._apply_fill_to_portfolio_stats,
            max_total_exposure=MAX_TOTAL_EXPOSURE,
            portfolio_id=self.portfolio_id,
        )

    def _plan_monitor_config(self) -> PlanMonitorConfig:
        """Build plan monitor config from runner settings."""
        return PlanMonitorConfig(
            max_plan_age_minutes=self.max_plan_age_minutes,
            trail_to_breakeven_pct=self._apply_plan_trailing_pct,
            overnight_widen_enabled=self.plan_overnight_widen_enabled,
            overnight_widen_pct=self.plan_overnight_widen_pct,
            overnight_widen_abs=self.plan_overnight_widen_abs,
            overnight_widen_max_pct=self.plan_overnight_max_widen_pct,
            auto_rearm_on_restart=self.plan_auto_rearm_on_restart,
            portfolio_day_timezone=PORTFOLIO_DAY_TIMEZONE,
        )

    def _refresh_resync_bindings(self):
        """Keep resync service aligned with current db/bot/portfolio for tests and stubs."""
        self.resync_service.db = self.db
        self.resync_service.bot = self.bot
        self.resync_service.risk_manager = self.risk_manager
        self.resync_service.trade_sync_cutoff_minutes = TRADE_SYNC_CUTOFF_MINUTES
        if hasattr(self.resync_service, "set_portfolio"):
            self.resync_service.set_portfolio(self.portfolio_id)

    def _refresh_orchestrator_bindings(self):
        """Keep orchestrator dependencies in sync when tests swap stubs."""
        self.orchestrator.command_processor = self.command_processor
        self.orchestrator.plan_monitor = self.plan_monitor
        self.orchestrator.risk_manager = self.risk_manager
        self.orchestrator.health_manager = self.health_manager
        self.orchestrator.portfolio_id = self.portfolio_id
        self.plan_monitor.portfolio_id = self.portfolio_id

    def _get_active_symbols(self) -> list[str]:
        """Return ordered list of symbols to monitor/trade."""
        symbols = []
        # 1) Configured symbols (ordered)
        symbols.extend([s for s in ALLOWED_SYMBOLS if s])
        # 2) Live state from DB snapshots
        try:
            positions = self.db.get_positions_for_portfolio(self.portfolio_id) if self.portfolio_id else []
            symbols.extend([p.get("symbol") for p in positions or [] if p.get("symbol")])
        except Exception:
            pass
        try:
            orders = self.db.get_open_orders_for_portfolio(self.portfolio_id) if self.portfolio_id else []
            symbols.extend([o.get("symbol") for o in orders or [] if o.get("symbol")])
        except Exception:
            pass
        try:
            plans = self.db.get_open_trade_plans_for_portfolio(self.portfolio_id) if self.portfolio_id else []
            symbols.extend([p.get("symbol") for p in plans or [] if p.get("symbol")])
        except Exception:
            pass

        deduped = []
        seen = set()
        for sym in symbols:
            if not sym:
                continue
            try:
                canonical = normalize_symbol(sym)
            except ValueError:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            deduped.append(canonical)
        return deduped or ["BTC/USD"]

    def _get_rebuild_symbols(self) -> list[str]:
        """Return symbols to use when rebuilding stats from exchange history."""
        allowed = []
        for sym in ALLOWED_SYMBOLS:
            try:
                allowed.append(normalize_symbol(sym))
            except ValueError:
                continue
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

    def _get_sync_symbols(self) -> set[str]:
        """Collect symbols from DB state for trade sync."""
        symbols = set()
        try:
            symbols.update(self.db.get_distinct_trade_symbols_for_portfolio(self.portfolio_id) or [])
            symbols.update({p.get('symbol') for p in self.db.get_positions_for_portfolio(self.portfolio_id) or [] if p.get('symbol')})
            symbols.update({p.get('symbol') for p in self.db.get_open_trade_plans_for_portfolio(self.portfolio_id) or [] if p.get('symbol')})
            symbols.update({o.get('symbol') for o in self.db.get_open_orders_for_portfolio(self.portfolio_id) or [] if o.get('symbol')})
        except Exception:
            pass
        normalized = set()
        for sym in symbols:
            if not isinstance(sym, str):
                continue
            try:
                normalized.add(normalize_symbol(sym))
            except ValueError:
                continue
        symbols = normalized
        if not symbols:
            symbols = {"BTC/USD"}
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
            "portfolio_id": self.portfolio_id,
            "run_id": self.run_id,
        }
        self._emit_telemetry(record)

    def _record_operational_metrics(
        self,
        current_exposure: float,
        current_equity: float,
        per_symbol_exposure: dict | None = None,
    ):
        """Emit health metrics for risk counters and LLM budget."""
        llm_cost = (self.portfolio_stats or {}).get("total_llm_cost", 0.0) or 0.0
        try:
            gross = (self.portfolio_stats or {}).get("gross_pnl", 0.0) or 0.0
            fees = (self.portfolio_stats or {}).get("total_fees", 0.0) or 0.0
            net = gross - fees - llm_cost
            fee_ratio = fees / max(abs(gross), 1.0)
            try:
                self.portfolio_tracker.update_exposure_notional(current_exposure)
            except Exception as exc:
                logger.debug(f"Could not persist exposure to stats cache: {exc}")
            risk_detail = {
                "exposure": current_exposure,
                "exposure_limit": MAX_TOTAL_EXPOSURE,
                "exposure_headroom": max(0.0, MAX_TOTAL_EXPOSURE - current_exposure),
                "per_symbol_exposure": per_symbol_exposure or {},
                "fee_ratio": fee_ratio,
                "gross_pnl": gross,
                "total_fees": fees,
                "total_llm_cost": llm_cost,
                "equity": current_equity,
                "net_pnl": net,
            }
            self._record_health_state("risk_metrics", "ok", risk_detail)
        except Exception as exc:
            logger.debug(f"Could not emit risk metrics: {exc}")

        try:
            portfolio_started = self._equity_baseline_ts
            if portfolio_started is None and self.portfolio:
                portfolio_started = self.portfolio.get("created_at")
            if portfolio_started is None and self.portfolio_id is not None:
                try:
                    baseline = self.db.get_first_equity_snapshot_for_portfolio(self.portfolio_id)
                    if baseline:
                        portfolio_started = baseline.get("timestamp")
                except Exception as exc:
                    logger.debug(f"Could not load equity baseline timestamp: {exc}")
            burn_stats = self.cost_tracker.calculate_llm_burn(
                total_llm_cost=llm_cost,
                run_started=portfolio_started,
                budget=LLM_MAX_PORTFOLIO_COST,
            )
            budget_status = "ok"
            if burn_stats.get("remaining_budget", 0.0) <= 0:
                budget_status = "cap_hit"
            elif burn_stats.get("pct_of_budget", 0.0) >= 0.8:
                budget_status = "near_cap"
            self._record_health_state("llm_budget", budget_status, burn_stats)
        except Exception as exc:
            logger.debug(f"Could not emit LLM budget metrics: {exc}")

    def _seed_equity_baseline(self, initial_equity: float) -> float:
        """
        Determine the portfolio baseline using the earliest equity snapshot.
        Logs the current equity to snapshots so restarts have a persisted reference.
        """
        baseline = initial_equity
        baseline_ts = None
        if self.portfolio_id is not None:
            try:
                first_snapshot = self.db.get_first_equity_snapshot_for_portfolio(self.portfolio_id)
                if first_snapshot:
                    baseline = first_snapshot.get("equity", initial_equity)
                    baseline_ts = first_snapshot.get("timestamp")
            except Exception as exc:
                logger.debug(f"Could not load baseline snapshot: {exc}")
            try:
                snapshot_ts = datetime.now(timezone.utc).isoformat()
                self.db.log_equity_snapshot_for_portfolio(self.portfolio_id, initial_equity, timestamp=snapshot_ts)
                baseline_ts = baseline_ts or snapshot_ts
            except Exception as exc:
                logger.debug(f"Could not log initial equity snapshot: {exc}")
        self._equity_baseline_ts = baseline_ts
        return baseline

    async def _reconcile_exchange_state(self):
        """
        Reconcile positions and open orders against the live exchange at startup.
        Ensures DB snapshots and risk manager state reflect actual venue state.
        """
        self._refresh_resync_bindings()
        await self.resync_service.reconcile_exchange_state()

    def _log_execution_trace(self, trace_id: int, execution_result: dict):
        """Attach execution outcome to LLM trace when available."""
        try:
            self.db.update_llm_trace_execution(trace_id, execution_result)
        except Exception as e:
            logger.debug(f"Could not update LLM trace {trace_id}: {e}")

    def _apply_exchange_trades_for_rebuild(self, trades: list) -> dict:
        """Delegate trade replay to portfolio tracker (kept for compatibility)."""
        stats = self.portfolio_tracker.apply_exchange_trades_for_rebuild(trades)
        self.portfolio_stats = self.portfolio_tracker.portfolio_stats
        return stats

    async def _write_end_of_day_snapshot(self, equity: float):
        """Capture equity, positions, and open plans for restart continuity."""
        if not self.portfolio_id:
            return
        positions: list[dict] = []
        try:
            positions = await self.bot.get_positions_async()
            self.db.replace_positions_for_portfolio(self.portfolio_id, positions or [])
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug(f"Could not fetch live positions for EOD snapshot: {exc}")
            try:
                positions = self.db.get_positions_for_portfolio(self.portfolio_id) or []
            except Exception:
                positions = []

        try:
            open_plans = self.db.get_open_trade_plans_for_portfolio(self.portfolio_id) or []
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug(f"Could not load open plans for EOD snapshot: {exc}")
            open_plans = []

        tz_name = getattr(self.db, "portfolio_day_timezone", None)
        timestamp = datetime.now(timezone.utc)
        self.db.log_end_of_day_snapshot_for_portfolio(
            self.portfolio_id,
            equity=equity,
            positions=positions,
            plans=open_plans,
            timestamp=timestamp,
            timezone_name=tz_name,
            run_id=self.run_id,
        )

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
        return self.resync_service.filter_our_orders(orders)

    async def _reconcile_open_orders(self):
        """
        Refresh open order snapshot using live exchange data and drop any DB orders
        that no longer exist on the venue.
        """
        self._refresh_resync_bindings()
        await self.resync_service.reconcile_open_orders()

    def _passes_rr_filter(self, action: str, price: float, stop_price: float, target_price: float) -> bool:
        """Delegate RR filter to action handler for compatibility."""
        return self.action_handler.passes_rr_filter(action, price, stop_price, target_price, min_rr=MIN_RR)

    def _slippage_within_limit(self, decision_price: float, latest_price: float, market_data_point: dict = None):
        """Delegate slippage cap check to action handler for compatibility."""
        max_spread_pct, min_top_of_book_notional, _ = self._microstructure_thresholds(market_data_point or {})
        return self.action_handler.slippage_within_limit(
            decision_price,
            latest_price,
            market_data_point or {},
            max_slippage_pct=MAX_SLIPPAGE_PCT,
            max_spread_pct=max_spread_pct,
            min_top_of_book_notional=min_top_of_book_notional,
        )

    def _capture_sandbox_position_baseline(self, positions: list | None):
        """
        Record the initial sandbox inventory so exposure snapshots only reflect
        positions opened/closed during this run.
        """
        if not self.sandbox_ignore_positions or not positions:
            return

        updated = False
        for pos in positions:
            sym = pos.get('symbol')
            if not sym:
                continue
            qty = pos.get('quantity', 0.0) or 0.0
            if sym not in self._sandbox_position_baseline:
                self._sandbox_position_baseline[sym] = qty
                updated = True

        if updated:
            self.risk_manager.set_position_baseline(self._sandbox_position_baseline)
            if self.context and hasattr(self.context, "set_position_baseline"):
                self.context.set_position_baseline(self._sandbox_position_baseline)

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
            "portfolio_id": self.portfolio_id,
            "run_id": self.run_id,
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
        """Delegate OHLCV capture to market data service (compatibility wrapper)."""
        # Refresh mutable settings/clock for tests
        self.market_data_service.db = self.db
        self.market_data_service.bot = self.bot
        self.market_data_service.monotonic = self._monotonic
        self.market_data_service.ohlcv_min_capture_spacing_seconds = self.ohlcv_min_capture_spacing_seconds
        self.market_data_service.ohlcv_retention_limit = self.ohlcv_retention_limit
        self.market_data_service.set_portfolio(self.portfolio_id)
        await self.market_data_service.capture_ohlcv(symbol)

    def _apply_order_value_buffer(self, quantity: float, price: float, symbol: str | None = None):
        """Delegate order value buffer to action handler for compatibility."""
        return self.action_handler.apply_order_value_buffer(quantity, price, symbol=symbol)

    def _merge_ib_order_metadata(self, telemetry_record: dict, order_result: dict | None):
        if self.exchange_name != "IB" or not isinstance(order_result, dict):
            return
        mapping = {
            "contract_id": "ib_contract_id",
            "exchange": "ib_exchange",
            "primary_exchange": "ib_primary_exchange",
            "commission_source": "ib_commission_source",
            "instrument_type": "instrument_type",
        }
        for source, target in mapping.items():
            val = order_result.get(source)
            if val is not None:
                telemetry_record[target] = val

    def _microstructure_thresholds(self, market_data_point: dict | None) -> tuple[float, float, float | None]:
        md = market_data_point or {}
        instrument_type = md.get("instrument_type")
        symbol = md.get("symbol")

        if self.exchange_name == "IB" and not instrument_type and symbol:
            try:
                normalized = normalize_symbol(symbol)
                base, quote = normalized.split("/", 1)
                instrument_type = infer_instrument_type(
                    base,
                    quote,
                    allowed_instrument_types=IB_ALLOWED_INSTRUMENT_TYPES,
                    base_currency=IB_BASE_CURRENCY,
                )
            except Exception:
                instrument_type = None

        if self.exchange_name == "IB":
            if instrument_type == "STK":
                return IB_EQUITY_MAX_SPREAD_PCT, IB_EQUITY_MIN_TOP_OF_BOOK_NOTIONAL, IB_EQUITY_MIN_QUOTE_SIZE
            if instrument_type == "FX":
                return IB_FX_MAX_SPREAD_PCT, IB_FX_MIN_TOP_OF_BOOK_NOTIONAL, None

        return MAX_SPREAD_PCT, MIN_TOP_OF_BOOK_NOTIONAL, None

    def _liquidity_ok(self, market_data_point: dict) -> bool:
        """Delegate liquidity check to action handler for compatibility."""
        max_spread_pct, min_top_of_book_notional, min_quote_size = self._microstructure_thresholds(market_data_point)
        return self.action_handler.liquidity_ok(
            market_data_point,
            max_spread_pct=max_spread_pct,
            min_top_of_book_notional=min_top_of_book_notional,
            min_quote_size=min_quote_size,
        )

    async def _monitor_trade_plans(self, price_lookup: dict, open_orders: list):
        """Delegate to the standalone PlanMonitor service."""
        config = self._plan_monitor_config()
        refresh_bindings = lambda: self.plan_monitor.refresh_bindings(  # noqa: E731
            bot=self.bot,
            db=self.db,
            cost_tracker=self.cost_tracker,
            risk_manager=self.risk_manager,
            prefer_maker=self._prefer_maker,
            holdings_updater=self._update_holdings_and_realized,
            portfolio_stats_applier=self._apply_fill_to_portfolio_stats,
            portfolio_id=self.portfolio_id,
        )
        await self.orchestrator.monitor_trade_plans(
            price_lookup=price_lookup,
            open_orders=open_orders,
            config=config,
            refresh_bindings_cb=refresh_bindings,
            portfolio_id=self.portfolio_id,
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
            f"Max positions {MAX_POSITIONS}"
        )
        
        # Connect to the active exchange
        await self.bot.connect_async()

        # Initialize tool coordinator after exchange connection
        exchange_for_tools = getattr(self.bot, "exchange", None)
        if not exchange_for_tools and hasattr(self.bot, "fetch_ohlcv"):
            exchange_for_tools = self.bot

        if exchange_for_tools:
            self.data_fetch_coordinator = DataFetchCoordinator(
                exchange_for_tools,
                error_callback=self.health_manager.record_tool_failure,
                success_callback=self.health_manager.record_tool_success,
                portfolio_id=self.portfolio_id,
                run_id=self.run_id,
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
        
        # Seed trackers for this portfolio (no legacy session wiring)
        self.starting_equity = self._seed_equity_baseline(initial_equity)
        self.risk_manager.set_baseline(self.starting_equity, timestamp=self._equity_baseline_ts)
        self.portfolio_tracker.set_portfolio(self.portfolio_id)
        self.resync_service.set_portfolio(self.portfolio_id)

        # Clear any old pending commands from previous runs
        self.db.clear_old_commands()
        try:
            self.db.prune_commands(COMMAND_RETENTION_DAYS)
        except Exception as e:
            logger.debug(f"Could not prune commands: {e}")

        # Initialize trading context
        self.context = TradingContext(self.db, self.portfolio_id, run_id=self.run_id)
        # Load persisted snapshots for restart resilience before hitting the exchange
        bootstrap_state = self.resync_service.bootstrap_snapshots()
        try:
            open_plans = (bootstrap_state or {}).get("open_plans") if bootstrap_state is not None else []
            if open_plans:
                refreshed_plans = self.plan_monitor.rearm_after_restart(
                    open_plans=open_plans,
                    config=self._plan_monitor_config(),
                    now=datetime.now(timezone.utc),
                )
                bootstrap_state["open_plans"] = refreshed_plans
        except Exception as exc:
            logger.debug(f"Could not auto-rearm plan monitors on restart: {exc}")
        if bootstrap_state["positions"] or bootstrap_state["open_orders"]:
            logger.info(
                f"Restored {len(bootstrap_state['positions'])} positions and "
                f"{len(bootstrap_state['open_orders'])} open orders from portfolio snapshots"
            )
        # Drop any stale open orders lingering from prior runs and sync with venue
        await self._reconcile_exchange_state()

        # Initialize portfolio stats with persistence awareness
        logger.info("Initializing portfolio stats...")
        _, cache_hit = self.portfolio_tracker.load_cached_stats()
        if cache_hit:
            self.portfolio_stats = dict(self.portfolio_tracker.portfolio_stats)
            logger.info(f"Loaded portfolio stats from cache: {self.portfolio_stats}")
        else:
            logger.info("No cached portfolio stats found; rebuilding from exchange trades...")
            start_ts_ms = 0
            try:
                latest_trade_ts = self.db.get_latest_trade_timestamp_for_portfolio(self.portfolio_id)
                if latest_trade_ts:
                    parsed = datetime.fromisoformat(latest_trade_ts)
                    start_ts_ms = int(parsed.timestamp() * 1000)
            except Exception as exc:
                logger.debug(f"Could not derive rebuild start timestamp: {exc}")

            symbols = self._get_rebuild_symbols()
            trades = []
            for sym in symbols:
                sym_trades = await self.bot.get_trades_from_timestamp(sym, start_ts_ms)
                trades.extend(sym_trades)

            self._apply_exchange_trades_for_rebuild(trades)

            db_stats = self.db.get_portfolio_stats(self.portfolio_id)
            self.portfolio_stats['total_llm_cost'] = db_stats.get('total_llm_cost', 0.0)
            self.portfolio_tracker.portfolio_stats = self.portfolio_stats
            try:
                self.portfolio_tracker._persist_stats_cache()
            except Exception as exc:
                logger.debug(f"Could not persist rebuilt stats cache: {exc}")
            logger.info(f"Portfolio Stats Rebuilt: {self.portfolio_stats}")

        # Seed risk manager with current equity for telemetry
        self.risk_manager.update_equity(initial_equity)

        # No need to reconcile_exchange_state in the old way; we just trust the exchange now.
        # But we might want to log initial positions to DB for debugging.
        live_positions = await self.bot.get_positions_async()
        self._capture_sandbox_position_baseline(live_positions)
        self.db.replace_positions_for_portfolio(self.portfolio_id, live_positions)
        bot_actions_logger.info(f"💰 Starting Equity: ${initial_equity:,.2f}")

    # reconcile_exchange_state removed as we trust exchange data directly now

    def _update_holdings_and_realized(self, symbol: str, action: str, quantity: float, price: float, fee: float) -> float:
        """Delegate holdings/PnL updates to portfolio tracker (kept for compatibility)."""
        realized = self.portfolio_tracker.update_holdings_and_realized(symbol, action, quantity, price, fee)
        self.portfolio_stats = self.portfolio_tracker.portfolio_stats
        return realized

    def _apply_trade_to_holdings(self, symbol: str, action: str, quantity: float, price: float):
        """Delegate to portfolio tracker."""
        self.portfolio_tracker.apply_trade_to_holdings(symbol, action, quantity, price)
        self.portfolio_stats = self.portfolio_tracker.portfolio_stats

    def _load_holdings_from_db(self):
        """Rebuild holdings via portfolio tracker."""
        self.portfolio_tracker.set_portfolio(self.portfolio_id)
        self.portfolio_tracker.load_holdings_from_db()
        self.portfolio_stats = self.portfolio_tracker.portfolio_stats

    def _apply_fill_to_portfolio_stats(self, order_id: str, actual_fee: float, realized_pnl: float):
        """Delegate portfolio accounting to portfolio tracker."""
        self.portfolio_tracker.set_portfolio(self.portfolio_id)
        self.portfolio_tracker.apply_fill_to_portfolio_stats(order_id, actual_fee, realized_pnl, estimated_fee_map=self._estimated_fees)
        self.portfolio_stats = self.portfolio_tracker.portfolio_stats

    def _sanity_check_equity_vs_stats(self, current_equity: float):
        """Compare estimated net PnL vs equity delta; log if off by >10%."""
        if current_equity is None:
            return
        try:
            starting = self.starting_equity
            if starting is None and self.portfolio_id is not None:
                try:
                    baseline = self.db.get_first_equity_snapshot_for_portfolio(self.portfolio_id)
                    if baseline:
                        starting = baseline.get("equity")
                except Exception as exc:
                    logger.debug(f"Could not load equity baseline for sanity check: {exc}")
            if starting is None:
                return
            estimated_net = (
                self.portfolio_stats.get('gross_pnl', 0.0)
                - self.portfolio_stats.get('total_fees', 0.0)
                - self.portfolio_stats.get('total_llm_cost', 0.0)
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

    async def _rebuild_portfolio_stats_from_trades(self, current_equity: float = None):
        """Recompute portfolio_stats from recorded trades and update cache via portfolio tracker."""
        self.portfolio_tracker.set_portfolio(self.portfolio_id)
        self.portfolio_stats = self.portfolio_tracker.rebuild_portfolio_stats_from_trades(current_equity)
        logger.info(f"Portfolio stats rebuilt from trades: {self.portfolio_stats}")
        if current_equity is not None:
            self._sanity_check_equity_vs_stats(current_equity)

    def _set_shutdown_reason(self, reason: str):
        """Keep the first shutdown reason to surface in logs."""
        if not self.shutdown_reason:
            self.shutdown_reason = reason

    async def _close_all_positions_safely(self):
        """Attempt to flatten all positions using market-ish orders."""
        try:
            positions = self.db.get_positions_for_portfolio(self.portfolio_id)
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
                    result = await self.bot.place_order_async(
                        symbol,
                        'SELL',
                        quantity,
                        prefer_maker=False,
                        force_market=True,
                    )
                    if result:
                        fee = self.cost_tracker.calculate_trade_fee(symbol, quantity, price, 'SELL', liquidity=result.get('liquidity', 'taker'))
                        realized_pnl = self._update_holdings_and_realized(symbol, 'SELL', quantity, price, fee)
                        self.db.log_trade_for_portfolio(
                            self.portfolio_id,
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
        """Sync recent trades from exchange to DB via resync service."""
        if not self.portfolio_id:
            return
        self._refresh_resync_bindings()
        await self.resync_service.sync_trades_from_exchange(
            processed_trade_ids=self.processed_trade_ids,
            order_reasons=self.order_reasons,
            plan_reason_lookup=lambda portfolio_id, order_id, client_oid: self.db.get_trade_plan_reason_by_order_for_portfolio(
                portfolio_id, order_id=order_id, client_order_id=client_oid
            ),
            get_symbols=lambda: self._get_sync_symbols(),
        )

    async def cleanup(self):
        """Cleanup and close connection."""
        logger.info(f"Cleaning up connections... (reason: {self.shutdown_reason or 'unspecified'})")
        bot_actions_logger.info(f"🧹 Cleanup starting (reason: {self.shutdown_reason or 'unspecified'})")
        
        # Save final portfolio statistics
        if self.portfolio_id:
            try:
                # Get final equity snapshot
                final_equity = await self.bot.get_equity_async()
                
                # Get portfolio stats and rebuild to ensure consistency
                portfolio_stats = self.db.get_portfolio_stats(self.portfolio_id)
                try:
                    await self._rebuild_portfolio_stats_from_trades(final_equity)
                    portfolio_stats = self.db.get_portfolio_stats(self.portfolio_id)
                except Exception as e:
                    logger.debug(f"Could not rebuild stats on cleanup: {e}")
                
                # Calculate PnL using fee-exclusive realized and separate fees
                gross_pnl = portfolio_stats.get('gross_pnl', 0.0) or 0.0
                net_pnl = gross_pnl - (portfolio_stats.get('total_fees', 0.0) or 0.0) - (portfolio_stats.get('total_llm_cost', 0.0) or 0.0)
                equity_delta = final_equity - (self.starting_equity or final_equity)
                try:
                    self.db.log_equity_snapshot_for_portfolio(self.portfolio_id, final_equity)
                except Exception as exc:
                    logger.debug(f"Could not persist final equity snapshot: {exc}")
                try:
                    await self._write_end_of_day_snapshot(final_equity)
                except Exception as exc:
                    logger.debug(f"Could not persist end-of-day snapshot: {exc}")
                
                # Log summary to bot.log
                bot_actions_logger.info("=" * 50)
                bot_actions_logger.info("📊 PORTFOLIO SUMMARY")
                bot_actions_logger.info("=" * 50)
                bot_actions_logger.info(f"Total Trades: {portfolio_stats['total_trades']}")
                bot_actions_logger.info(f"Gross PnL (ex-fee): ${gross_pnl:,.2f}")
                bot_actions_logger.info(f"Trading Fees: ${portfolio_stats['total_fees']:.2f}")
                bot_actions_logger.info(f"LLM Costs: ${portfolio_stats['total_llm_cost']:.4f}")
                bot_actions_logger.info(f"Net PnL: ${net_pnl:,.2f}")
                bot_actions_logger.info(f"Equity Delta (broker): ${equity_delta:,.2f}")
                
                if net_pnl > 0:
                    bot_actions_logger.info("✅ Profitable run!")
                else:
                    bot_actions_logger.info("❌ Unprofitable run")
                bot_actions_logger.info("=" * 50)
                
            except Exception as e:
                logger.error(f"Error saving portfolio stats: {e}")
        
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
            self._refresh_orchestrator_bindings()
            await self.orchestrator.start(self.initialize)
            self.running = True
            loops = 0

            async def sleep_loop():
                """Sleep for the configured interval and bump loop counter."""
                nonlocal loops
                logger.info(f"Sleeping for {LOOP_INTERVAL_SECONDS} seconds...")
                await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                loops += 1
            
            while self.running and self.orchestrator.running:
                if self._kill_switch:
                    if not self.shutdown_reason:
                        self._set_shutdown_reason("kill switch")
                    bot_actions_logger.info("🛑 Kill switch active; exiting main loop.")
                    self.orchestrator.request_stop(self.shutdown_reason)
                    self.running = False
                    break
                try:
                    if max_loops is not None and loops >= max_loops:
                        break
                    exchange_error_seen = False
                    # 0. Check for pending commands from dashboard
                    command_result = await self.orchestrator.process_commands(
                        close_positions_cb=self._close_all_positions_safely,
                        stop_cb=self._set_shutdown_reason,
                    )
                    if command_result.stop_requested:
                        self.orchestrator.request_stop(command_result.shutdown_reason)
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

                    if self.portfolio_id is not None:
                        try:
                            self.db.log_equity_snapshot_for_portfolio(self.portfolio_id, current_equity)
                        except Exception as e:
                            logger.warning(f"Could not log equity snapshot: {e}")
                    self.risk_manager.update_equity(current_equity)
                    
                    risk_result = await self.orchestrator.enforce_risk_budget(
                        current_equity=current_equity,
                        close_positions_cb=self._close_all_positions_safely,
                        set_shutdown_reason=self._set_shutdown_reason,
                    )
                    if risk_result.should_stop:
                        self._kill_switch = risk_result.kill_switch
                        if risk_result.shutdown_reason and not self.shutdown_reason:
                            self.shutdown_reason = risk_result.shutdown_reason
                        self.orchestrator.request_stop(self.shutdown_reason)
                        self.running = False
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

                    for sym in symbols:
                        ticker_started = self._monotonic()
                        try:
                            md = await self.bot.get_market_data_async(sym)
                        except Exception as e:
                            logger.warning(f"Market data fetch failed for {sym}: {e}")
                            self.health_manager.record_exchange_failure("get_market_data_async", e)
                            exchange_error_seen = True
                            market_data[sym] = None
                            continue
                        ticker_ended = self._monotonic()
                        if md is not None:
                            md["_latency_ms"] = (ticker_ended - ticker_started) * 1000
                            md["_fetched_monotonic"] = ticker_ended
                            md["fetched_at"] = datetime.now(timezone.utc).isoformat()
                        market_data[sym] = md

                        # Log market data to database
                        if md and self.portfolio_id is not None:
                            try:
                                self.db.log_market_data_for_portfolio(
                                    self.portfolio_id,
                                    sym,
                                    md.get('price'),
                                    md.get('bid'),
                                    md.get('ask'),
                                    md.get('volume'),
                                    spread_pct=md.get('spread_pct'),
                                    bid_size=md.get('bid_size'),
                                    ask_size=md.get('ask_size'),
                                    ob_imbalance=md.get('ob_imbalance'),
                                )
                            except Exception as e:
                                logger.warning(f"Could not log market data: {e}")
                    if self.portfolio_id is not None:
                        try:
                            self.db.prune_market_data_for_portfolio(self.portfolio_id, MARKET_DATA_RETENTION_MINUTES)
                        except Exception as e:
                            logger.debug(f"Could not prune market data: {e}")
                    if not market_data:
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    fresh_market, stale_market = self.orchestrator.evaluate_market_health(market_data)
                    if not fresh_market:
                        stale_list = ", ".join(sorted(stale_market.keys())) or "all symbols"
                        bot_actions_logger.info(f"⏸️ Skipping loop: market data stale or missing for {stale_list}")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue
                    symbols = [sym for sym in symbols if sym in fresh_market]
                    market_data = {sym: market_data[sym] for sym in symbols if sym in fresh_market}

                    liquid_symbols = []
                    illiquid_symbols = []
                    for sym in symbols:
                        md = market_data.get(sym) or {}
                        if self._liquidity_ok(md):
                            liquid_symbols.append(sym)
                        else:
                            illiquid_symbols.append(sym)
                    if illiquid_symbols:
                        bot_actions_logger.info(f"💧 Skipping illiquid symbols: {', '.join(sorted(illiquid_symbols))}")
                    if not liquid_symbols:
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue
                    symbols = liquid_symbols
                    market_data = {sym: market_data[sym] for sym in symbols}

                    # Capture multi-timeframe OHLCV for each active symbol
                    for sym in symbols:
                        try:
                            await self._capture_ohlcv(sym)
                        except Exception as e:
                            logger.debug(f"Could not capture OHLCV for {sym}: {e}")

                    primary_symbol = symbols[0] if symbols else None
                    if not primary_symbol:
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    primary_data = market_data.get(primary_symbol)

                    # Refresh live positions each loop for accurate exposure snapshots
                    try:
                        live_positions = await self.bot.get_positions_async()
                        self._capture_sandbox_position_baseline(live_positions)
                        self.db.replace_positions_for_portfolio(self.portfolio_id, live_positions)
                    except Exception as e:
                        logger.warning(f"Could not refresh positions: {e}")
                        self.health_manager.record_exchange_failure("get_positions_async", e)
                        exchange_error_seen = True
                    # Refresh open orders for exposure headroom and context
                    open_orders = []
                    try:
                        open_orders = await self.bot.get_open_orders_async()
                        open_orders = self._filter_our_orders(open_orders)
                        self.db.replace_open_orders_for_portfolio(self.portfolio_id, open_orders)
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
                        positions_data = self.db.get_positions_for_portfolio(self.portfolio_id)
                        for pos in positions_data:
                            sym = pos['symbol']
                            current_price = pos.get('current_price') or pos.get('avg_price') or 0

                            # Prefer most recent market tick
                            recent_data = self.db.get_recent_market_data_for_portfolio(self.portfolio_id, sym, limit=1)
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
                            latest = self.db.get_recent_market_data_for_portfolio(self.portfolio_id, sym, limit=1) if sym else None
                            if latest and latest[0].get('price'):
                                price_lookup[sym] = latest[0]['price']
                        self.risk_manager.update_pending_orders(open_orders, price_lookup=price_lookup)

                        price_overrides = {sym: md.get('price') for sym, md in market_data.items() if md and md.get('price')}
                        price_overrides = price_overrides or None
                        current_exposure, per_symbol_exposure = self.risk_manager.compute_exposure(price_overrides=price_overrides)
                        self.orchestrator.emit_operational_metrics(
                            current_exposure,
                            current_equity,
                            per_symbol_exposure=per_symbol_exposure,
                        )
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
                        self.orchestrator.request_stop(self.shutdown_reason)
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
                    # Keep portfolio stats cache fresh if DB trades grew
                    try:
                        db_trade_count = self.db.get_trade_count_for_portfolio(self.portfolio_id)
                        if db_trade_count > self.portfolio_stats.get('total_trades', 0):
                            await self._rebuild_portfolio_stats_from_trades(current_equity)
                    except Exception as e:
                        logger.debug(f"Could not refresh stats cache mid-loop: {e}")

                    # 3. Generate Signal via Strategy
                    # Pass portfolio_stats explicitly
                    signal = await self.strategy.generate_signal(
                        market_data,
                        current_equity,
                        current_exposure,
                        self.context,
                        portfolio_stats=self.portfolio_stats
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
                            "portfolio_id": self.portfolio_id,
                            "run_id": self.run_id,
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
                            await sleep_loop()
                            continue
                        
                        elif action == 'CANCEL':
                            if not order_id:
                                logger.warning("Skipped cancel: missing order_id")
                                telemetry_record["status"] = "cancel_missing_id"
                                telemetry_record["error"] = "missing order_id"
                                self._log_execution_trace(trace_id, {"status": "cancel_missing_id"})
                                self._emit_telemetry(telemetry_record)
                                await sleep_loop()
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
                                    self.db.replace_open_orders_for_portfolio(self.portfolio_id, open_orders)
                                except Exception as e:
                                    logger.warning(f"Could not refresh open orders after cancel: {e}")
                            except Exception as e:
                                logger.error(f"Cancel order error: {e}")
                                telemetry_record["status"] = "cancel_error"
                                telemetry_record["error"] = str(e)
                            self._log_execution_trace(trace_id, telemetry_record)
                            self._emit_telemetry(telemetry_record)
                            await sleep_loop()
                            continue

                        elif action == 'UPDATE_PLAN':
                            await self._handle_update_plan(signal, telemetry_record, trace_id)
                            await sleep_loop()
                            continue

                        elif action == 'PARTIAL_CLOSE':
                            await self._handle_partial_close(signal, telemetry_record, trace_id, market_data, current_exposure)
                            await sleep_loop()
                            continue

                        elif action == 'CLOSE_POSITION':
                            await self._handle_close_position(signal, telemetry_record, trace_id, market_data)
                            await sleep_loop()
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
                            await sleep_loop()
                            continue

                        elif action in ['BUY', 'SELL'] and quantity > 0:
                            # Get price for risk checks and execution
                            md = market_data.get(symbol)
                            price = md.get('price') if md else None
                            if price is None:
                                price = price_lookup.get(symbol)
                            
                            if not price:
                                logger.warning("Skipped trade: missing price data")
                                await sleep_loop()
                                continue

                        # Volatility sizing adjustment
                        adjusted_quantity = self._apply_volatility_sizing(quantity, regime_flags)
                        telemetry_record["vol_scaled_qty"] = adjusted_quantity

                        # Guardrails: clip size to sit under the max order cap minus buffer
                        quantity = self._apply_order_value_buffer(adjusted_quantity, price, symbol)

                        if quantity <= 0:
                            logger.warning("Skipped trade: buffered quantity became non-positive")
                            await sleep_loop()
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
                                await sleep_loop()
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
                                open_plan_count = self.db.count_open_trade_plans_for_symbol_for_portfolio(self.portfolio_id, symbol)
                                if open_plan_count >= self.max_plans_per_symbol:
                                    if AUTO_REPLACE_PLAN_ON_CAP:
                                        try:
                                            plans = self.db.get_open_trade_plans_for_portfolio(self.portfolio_id)
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
                                            await sleep_loop()
                                            continue
                                    else:
                                        bot_actions_logger.info(f"⛔ Trade Blocked: plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                        self.strategy.on_trade_rejected(f"Plan cap reached for {symbol} ({open_plan_count}/{self.max_plans_per_symbol})")
                                        await sleep_loop()
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
                                    await sleep_loop()
                                    continue
                                position_qty = (self.risk_manager.positions or {}).get(symbol, {}).get('quantity', 0.0) or 0.0
                                if self._stacking_block(action, symbol, open_plan_count, pending_data, position_qty):
                                    bot_actions_logger.info(f"⛔ Trade Blocked: stacking same-side risk on {symbol}")
                                    self.strategy.on_trade_rejected("Stacking blocked")
                                    telemetry_record["status"] = "risk_blocked"
                                    telemetry_record["risk_reason"] = "Stacking blocked"
                                    self._log_execution_trace(trace_id, telemetry_record)
                                    self._emit_telemetry(telemetry_record)
                                    await sleep_loop()
                                    continue
                                order_value = quantity * price
                                if action == 'BUY' and (pending_exposure + order_value + current_exposure) > MAX_TOTAL_EXPOSURE:
                                    bot_actions_logger.info("⛔ Trade Blocked: pending/open exposure would exceed cap")
                                    self.strategy.on_trade_rejected("Pending exposure over cap")
                                    telemetry_record["status"] = "risk_blocked"
                                    telemetry_record["risk_reason"] = "Pending exposure over cap"
                                    self._log_execution_trace(trace_id, telemetry_record)
                                    self._emit_telemetry(telemetry_record)
                                    await sleep_loop()
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
                                await sleep_loop()
                                continue

                            bot_actions_logger.info(f"✅ Executing: {action} {qty_str} {symbol} at ${price:,.2f} (est. fee: ${estimated_fee:.4f})")

                            if not self.execute_orders:
                                bot_actions_logger.info("👁️ Shadow mode: skipping live order placement")
                                await sleep_loop()
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

                            # Record trade plan so we can monitor stops/targets (only for new BUY/SELL)
                            if action in ['BUY', 'SELL'] and (stop_price or target_price):
                                try:
                                    plan_id = self.db.create_trade_plan_for_portfolio(
                                        self.portfolio_id,
                                        symbol,
                                        action,
                                        price,
                                        stop_price,
                                        target_price,
                                        quantity,
                                        reason,
                                        entry_order_id=order_result.get('order_id') if order_result else None,
                                        entry_client_order_id=order_result.get('client_order_id') if order_result else None,
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
                                    self.db.replace_open_orders_for_portfolio(self.portfolio_id, open_orders)
                                except Exception as e:
                                    logger.warning(f"Could not snapshot open orders: {e}")

                            telemetry_record["status"] = order_result.get('status') if isinstance(order_result, dict) else "order_unknown"
                            telemetry_record["order_result"] = order_result
                            self._merge_ib_order_metadata(telemetry_record, order_result)
                            self._log_execution_trace(trace_id, telemetry_record)
                            self._emit_telemetry(telemetry_record)

                        else:
                            telemetry_record["status"] = "risk_blocked"
                            bot_actions_logger.info(f"⛔ Trade Blocked: {risk_result.reason}")
                            self.strategy.on_trade_rejected(risk_result.reason)
                            self._log_execution_trace(trace_id, telemetry_record)
                            self._emit_telemetry(telemetry_record)
                            await sleep_loop()
                            continue

                    if not exchange_error_seen:
                        self.health_manager.reset_exchange_errors()
                    # 5. Sleep
                    await sleep_loop()

                except KeyboardInterrupt:
                    logger.info("Stopping loop...")
                    self._set_shutdown_reason("KeyboardInterrupt")
                    self.orchestrator.request_stop(self.shutdown_reason)
                    self.running = False
                    break
                except Exception as e:
                    logger.exception(f"Loop Error: {e}")
                    await asyncio.sleep(5)
        finally:
            # Always cleanup, even if there's an exception or break
            await self.orchestrator.cleanup(self.cleanup)

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
