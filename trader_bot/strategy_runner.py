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
    LOOP_INTERVAL_SECONDS,
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
    ACTIVE_SYMBOLS,
)
from trader_bot.cost_tracker import CostTracker
from trader_bot.data_fetch_coordinator import DataFetchCoordinator
from trader_bot.database import TradingDatabase
from trader_bot.gemini_trader import GeminiTrader
from trader_bot.logger_config import setup_logging
from trader_bot.risk_manager import RiskManager
from trader_bot.strategy import LLMStrategy
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
        # Simple in-memory holdings tracker for realized PnL
        self.holdings = {}  # symbol -> {'qty': float, 'avg_cost': float}
        # Track estimated fees per order so we can reconcile with actual fills
        self._estimated_fees = {}  # order_id -> estimated fee
        self._last_reconnect = 0.0
        self._kill_switch = False
        self._exchange_error_streak = 0
        self._tool_error_streak = 0
        self.exchange_error_threshold = EXCHANGE_ERROR_THRESHOLD
        self.exchange_pause_seconds = EXCHANGE_PAUSE_SECONDS
        self.tool_error_threshold = TOOL_ERROR_THRESHOLD
        self.tool_pause_seconds = TOOL_PAUSE_SECONDS
        self.ticker_max_age_ms = TICKER_MAX_AGE_SECONDS * 1000
        self.ticker_max_latency_ms = TICKER_MAX_LATENCY_MS
        self._exchange_health = "ok"
        self._tool_health = "ok"
        self.maker_preference_default = MAKER_PREFERENCE_DEFAULT
        self.maker_preference_overrides = MAKER_PREFERENCE_OVERRIDES or {}
        # Seed a default stats container so background tasks don't crash before initialization completes
        self.session_stats = {
            'total_trades': 0,
            'gross_pnl': 0.0,
            'total_fees': 0.0,
            'total_llm_cost': 0.0,
        }
        
        # Initialize Strategy
        self.strategy = LLMStrategy(
            self.db, 
            self.technical_analysis, 
            self.cost_tracker,
            open_orders_provider=self.bot.get_open_orders_async,
            ohlcv_provider=self.bot.fetch_ohlcv,
            tool_coordinator=None,  # set post-connect when exchange is ready
        )
        
        # Trade syncing state
        self.order_reasons = {}  # order_id -> reason
        self.processed_trade_ids = set()
        self._open_trade_plans = {}  # plan_id -> dict
        self.max_plan_age_minutes = PLAN_MAX_AGE_MINUTES
        self.day_end_flatten_hour_utc = None  # optional UTC hour to flatten plans
        self.max_plans_per_symbol = PLAN_MAX_PER_SYMBOL
        self.telemetry_logger = telemetry_logger
        self._apply_plan_trailing_pct = PLAN_TRAIL_TO_BREAKEVEN_PCT  # move stop to entry after move in favor
        self._pause_until = None  # timestamp (monotonic seconds) when pause expires
        self.shutdown_reason: str | None = None  # track why we stop

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

    def _get_active_symbols(self) -> list[str]:
        """Return ordered list of symbols to monitor/trade."""
        symbols = []
        # 1) Configured symbols (ordered)
        symbols.extend([s for s in ACTIVE_SYMBOLS if s])
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
            seen.add(sym_up)
            deduped.append(sym_up)
        return deduped or ["BTC/USD"]

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

    def _record_exchange_failure(self, context: str, error: Exception | str = None):
        """Track consecutive exchange failures and trigger auto-pause."""
        self._exchange_error_streak += 1
        detail = {
            "context": context,
            "error": str(error) if error is not None else None,
            "streak": self._exchange_error_streak,
        }
        if self._exchange_health != "degraded":
            self._exchange_health = "degraded"
            self._record_health_state("exchange_circuit", "degraded", detail)
        if self._exchange_error_streak >= self.exchange_error_threshold:
            pause_until = self._monotonic() + self.exchange_pause_seconds
            self._pause_until = max(self._pause_until or 0, pause_until)
            detail.update(
                {
                    "pause_seconds": self.exchange_pause_seconds,
                    "tripped_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._record_health_state("exchange_circuit", "tripped", detail)
            bot_actions_logger.info(
                f"🛑 Exchange circuit breaker tripped after {self.exchange_error_threshold} failures; pausing for {self.exchange_pause_seconds}s"
            )
            self._exchange_error_streak = 0
            self._exchange_health = "tripped"

    def _reset_exchange_errors(self):
        """Mark exchange channel healthy and clear streak."""
        if self._exchange_error_streak > 0 or self._exchange_health != "ok":
            self._exchange_error_streak = 0
            self._exchange_health = "ok"
            self._record_health_state("exchange_circuit", "ok", {"note": "recovered"})

    def _record_tool_failure(self, request: Any = None, error: Exception | str = None, context: str = None):
        """Track consecutive tool failures and trigger auto-pause."""
        self._tool_error_streak += 1
        detail = {
            "request": getattr(request, "id", None) if request is not None else None,
            "tool": getattr(request, "tool", None).value if getattr(request, "tool", None) else None,
            "context": context,
            "error": str(error) if error is not None else None,
            "streak": self._tool_error_streak,
        }
        if self._tool_health != "degraded":
            self._tool_health = "degraded"
            self._record_health_state("tool_circuit", "degraded", detail)
        if self._tool_error_streak >= self.tool_error_threshold:
            pause_until = self._monotonic() + self.tool_pause_seconds
            self._pause_until = max(self._pause_until or 0, pause_until)
            detail.update(
                {
                    "pause_seconds": self.tool_pause_seconds,
                    "tripped_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._record_health_state("tool_circuit", "tripped", detail)
            bot_actions_logger.info(
                f"🛑 Tool circuit breaker tripped after {self.tool_error_threshold} failures; pausing for {self.tool_pause_seconds}s"
            )
            self._tool_error_streak = 0
            self._tool_health = "tripped"

    def _record_tool_success(self):
        """Mark tool path healthy and clear streak."""
        if self._tool_error_streak > 0 or self._tool_health != "ok":
            self._tool_error_streak = 0
            self._tool_health = "ok"
            self._record_health_state("tool_circuit", "ok", {"note": "recovered"})

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

    def _is_stale_market_data(self, data: dict) -> tuple[bool, dict]:
        """Return (stale, detail) using latency and age thresholds."""
        if not data:
            return True, {"reason": "empty"}
        detail: dict[str, Any] = {}
        now_mono = self._monotonic()
        latency_ms = data.get("_latency_ms")
        if latency_ms is not None:
            detail["latency_ms"] = latency_ms
        fetched_mono = data.get("_fetched_monotonic")
        age_ms = None
        if fetched_mono is not None:
            age_ms = max(0.0, (now_mono - fetched_mono) * 1000)
        ts_field = data.get("timestamp") or data.get("ts")
        if ts_field is not None:
            ts_ms = ts_field if ts_field > 1e12 else ts_field * 1000
            wall_age = max(0.0, (time.time() * 1000) - ts_ms)
            age_ms = age_ms if age_ms is not None else wall_age
            detail["data_age_ms"] = wall_age
        if age_ms is None and fetched_mono is not None:
            age_ms = max(0.0, (now_mono - fetched_mono) * 1000)
        if age_ms is not None:
            detail["age_ms"] = age_ms
        if latency_ms is not None and latency_ms > self.ticker_max_latency_ms:
            detail["reason"] = "latency"
            return True, detail
        if age_ms is not None and age_ms > self.ticker_max_age_ms:
            detail["reason"] = "age"
            return True, detail
        return False, detail

    def _log_execution_trace(self, trace_id: int, execution_result: dict):
        """Attach execution outcome to LLM trace when available."""
        try:
            self.db.update_llm_trace_execution(trace_id, execution_result)
        except Exception as e:
            logger.debug(f"Could not update LLM trace {trace_id}: {e}")

    def _extract_fee_cost(self, fee_field: Any) -> float:
        """Normalize fee representations (dict, list, scalar) to a float cost."""
        if fee_field is None:
            return 0.0
        if isinstance(fee_field, dict):
            try:
                return float(fee_field.get('cost') or 0.0)
            except (TypeError, ValueError):
                return 0.0
        if isinstance(fee_field, (list, tuple)):
            total = 0.0
            for entry in fee_field:
                total += self._extract_fee_cost(entry)
            return total
        try:
            return float(fee_field)
        except (TypeError, ValueError):
            logger.debug(f"Unknown fee format during rebuild: {fee_field}")
            return 0.0

    def _normalize_exchange_trade(self, trade: dict) -> tuple[str, str, float, float, float] | None:
        """Ensure required fields exist and types are sane for rebuild steps."""
        if not isinstance(trade, dict):
            logger.warning("Skipping malformed trade during rebuild: not a dict")
            return None
        try:
            symbol = trade.get('symbol')
            side_raw = trade.get('side')
            quantity = trade.get('amount')
            price = trade.get('price')
            if not symbol or side_raw is None or quantity is None or price is None:
                logger.warning("Skipping malformed trade during rebuild: missing required fields")
                return None
            try:
                side = str(side_raw).upper()
                quantity_val = float(quantity)
                price_val = float(price)
            except (TypeError, ValueError) as exc:
                logger.warning(f"Skipping malformed trade during rebuild: {exc}")
                return None
            if quantity_val <= 0 or price_val <= 0:
                logger.warning("Skipping malformed trade during rebuild: non-positive quantity or price")
                return None
            fee_cost = self._extract_fee_cost(trade.get('fee'))
            return symbol, side, quantity_val, price_val, fee_cost
        except Exception as exc:
            logger.warning(f"Skipping malformed trade during rebuild: {exc}")
            return None

    def _apply_exchange_trades_for_rebuild(self, trades: list) -> dict:
        """
        Rebuild holdings and session stats from a list of exchange trades.
        Malformed entries are skipped with warnings instead of breaking the rebuild.
        """
        self.holdings = {}
        self.session_stats = {
            'total_trades': 0,
            'gross_pnl': 0.0,
            'total_fees': 0.0,
            'total_llm_cost': 0.0
        }
        skipped = 0
        for trade in trades or []:
            normalized = self._normalize_exchange_trade(trade)
            if not normalized:
                skipped += 1
                continue
            symbol, side, quantity, price, fee_cost = normalized
            try:
                realized = self._update_holdings_and_realized(symbol, side, quantity, price, fee_cost)
            except Exception as exc:
                logger.warning(f"Could not apply trade during rebuild for {symbol}: {exc}")
                skipped += 1
                continue

            self.session_stats['total_trades'] += 1
            self.session_stats['total_fees'] += fee_cost
            self.session_stats['gross_pnl'] += realized

        if skipped:
            logger.warning(f"Skipped {skipped} malformed trades while rebuilding stats")
        return self.session_stats

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
        """Require minimum risk/reward when both stop and target are provided."""
        if not price or stop_price is None or target_price is None:
            return True
        if action == 'BUY':
            risk = price - stop_price
            reward = target_price - price
        else:
            risk = stop_price - price
            reward = price - target_price
        if risk <= 0 or reward <= 0:
            return False
        rr = reward / risk
        return rr >= MIN_RR

    def _compute_slippage_cap(self, market_data_point: dict) -> float:
        """Derive a slippage cap that tightens on thin books or wide spreads."""
        cap = MAX_SLIPPAGE_PCT
        if not market_data_point:
            return cap

        spread_pct = market_data_point.get("spread_pct")
        bid = market_data_point.get("bid")
        ask = market_data_point.get("ask")
        bid_size = market_data_point.get("bid_size")
        ask_size = market_data_point.get("ask_size")

        top_notional = None
        if bid and ask and bid_size and ask_size:
            top_notional = min(bid * bid_size, ask * ask_size)

        if top_notional is not None:
            if top_notional < MIN_TOP_OF_BOOK_NOTIONAL:
                cap *= 0.25
            elif top_notional < MIN_TOP_OF_BOOK_NOTIONAL * 2:
                cap *= 0.5

        if spread_pct is not None and spread_pct > 0:
            factor = max(0.3, min(1.0, MAX_SPREAD_PCT / max(spread_pct, 1e-9)))
            cap *= factor

        return max(cap, MAX_SLIPPAGE_PCT * 0.1)

    def _slippage_within_limit(self, decision_price: float, latest_price: float, market_data_point: dict = None):
        """Return (allowed, move_pct) based on dynamic slippage cap."""
        if not decision_price or not latest_price:
            return True, 0.0
        move_pct = abs(latest_price - decision_price) / decision_price * 100
        cap = self._compute_slippage_cap(market_data_point or {})
        return move_pct <= cap, move_pct

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
        """
        Prevent stacking same-direction risk when we already have a position and
        pending orders/plans on the symbol.
        """
        if action != 'BUY':
            return False
        if position_qty <= 0:
            return False
        pending_same_side = (pending_data.get('count_buy', 0) or 0) > 0
        has_plans = open_plan_count > 0
        return pending_same_side or has_plans

    async def _handle_update_plan(self, signal, telemetry_record, trace_id):
        """Handle UPDATE_PLAN intents."""
        plan_id = getattr(signal, 'plan_id', None)
        stop_price = getattr(signal, 'stop_price', None)
        target_price = getattr(signal, 'target_price', None)
        reason = getattr(signal, 'reason', '') or 'Update plan'
        if not plan_id:
            telemetry_record["status"] = "update_plan_missing_id"
            self._emit_telemetry(telemetry_record)
            return
        try:
            self.db.update_trade_plan_prices(plan_id, stop_price=stop_price, target_price=target_price, reason=reason)
            telemetry_record["status"] = "plan_updated"
            self._log_execution_trace(trace_id, telemetry_record)
            self._emit_telemetry(telemetry_record)
            bot_actions_logger.info(f"✏️ Plan {plan_id} updated: stop={stop_price}, target={target_price}")
        except Exception as e:
            telemetry_record["status"] = "plan_update_error"
            telemetry_record["error"] = str(e)
            self._log_execution_trace(trace_id, telemetry_record)
            self._emit_telemetry(telemetry_record)
            logger.error(f"Plan update failed: {e}")

    async def _handle_partial_close(self, signal, telemetry_record, trace_id, market_data, current_exposure):
        """Handle PARTIAL_CLOSE intents."""
        plan_id = getattr(signal, 'plan_id', None)
        close_fraction = getattr(signal, 'close_fraction', None) or 0.0
        symbol = signal.symbol
        if not plan_id or close_fraction <= 0 or close_fraction > 1:
            telemetry_record["status"] = "partial_close_invalid"
            telemetry_record["error"] = "invalid plan_id or fraction"
            self._emit_telemetry(telemetry_record)
            return
        try:
            open_plans = self.db.get_open_trade_plans(self.session_id)
            plan = next((p for p in open_plans if p.get('id') == plan_id), None)
        except Exception:
            plan = None
        if not plan:
            telemetry_record["status"] = "partial_close_missing_plan"
            self._emit_telemetry(telemetry_record)
            return
        plan_side = plan.get('side', 'BUY').upper()
        plan_size = plan.get('size', 0.0) or 0.0
        close_qty = max(0.0, plan_size * close_fraction)
        if close_qty <= 0:
            telemetry_record["status"] = "partial_close_zero_qty"
            self._emit_telemetry(telemetry_record)
            return
        flatten_action = 'SELL' if plan_side == 'BUY' else 'BUY'
        price = market_data.get(symbol, {}).get('price') if market_data else None
        qty_for_risk = self._apply_order_value_buffer(close_qty, price or 0)
        risk_result = self.risk_manager.check_trade_allowed(symbol, flatten_action, qty_for_risk, price or 0)
        if not risk_result.allowed:
            telemetry_record["status"] = "partial_close_blocked"
            telemetry_record["risk_reason"] = risk_result.reason
            self.strategy.on_trade_rejected(risk_result.reason)
            self._emit_telemetry(telemetry_record)
            return
        bot_actions_logger.info(f"🔻 Partial close {close_fraction*100:.0f}% of plan {plan_id}: {flatten_action} {qty_for_risk} {symbol}")
        try:
            prefer_maker = self._prefer_maker(symbol)
            order_result = await self.bot.place_order_async(symbol, flatten_action, qty_for_risk, prefer_maker=prefer_maker)
            liquidity_tag = order_result.get('liquidity', 'taker') if order_result else 'taker'
            fee = self.cost_tracker.calculate_trade_fee(symbol, qty_for_risk, price or 0, flatten_action, liquidity=liquidity_tag)
            realized = self._update_holdings_and_realized(symbol, flatten_action, qty_for_risk, price or 0, fee)
            self.db.log_trade(
                self.session_id,
                symbol,
                flatten_action,
                qty_for_risk,
                price or 0,
                fee,
                f"Partial close plan {plan_id} ({close_fraction*100:.0f}%)",
                liquidity=order_result.get('liquidity') if order_result else 'taker',
                realized_pnl=realized,
            )
            self._apply_fill_to_session_stats(order_result.get('order_id') if order_result else None, fee, realized)
            remaining_size = max(plan_size - close_qty, 0.0)
            try:
                partial_reason = f"Partial close {close_fraction*100:.0f}%"
                if remaining_size <= 1e-9:
                    self.db.update_trade_plan_status(
                        plan_id,
                        status='closed',
                        closed_at=datetime.now(timezone.utc).isoformat(),
                        reason=partial_reason,
                    )
                else:
                    self.db.update_trade_plan_size(plan_id, size=remaining_size, reason=partial_reason)
            except Exception as exc:
                logger.debug(f"Could not update plan size after partial close: {exc}")
            telemetry_record["status"] = "partial_close_executed"
            telemetry_record["order_result"] = order_result
        except Exception as e:
            telemetry_record["status"] = "partial_close_error"
            telemetry_record["error"] = str(e)
            logger.error(f"Partial close failed: {e}")
        self._log_execution_trace(trace_id, telemetry_record)
        self._emit_telemetry(telemetry_record)

    async def _handle_close_position(self, signal, telemetry_record, trace_id, market_data):
        """Handle CLOSE_POSITION intents by flattening current holdings for symbol."""
        symbol = signal.symbol
        qty = 0.0
        price = market_data.get(symbol, {}).get('price') if market_data else None
        try:
            positions = self.db.get_positions(self.session_id)
            for pos in positions:
                if pos.get('symbol') == symbol:
                    qty = pos.get('quantity', 0.0) or 0.0
                    break
        except Exception:
            qty = 0.0
        if qty <= 0:
            telemetry_record["status"] = "close_position_none"
            self._emit_telemetry(telemetry_record)
            return
        action = 'SELL' if qty > 0 else 'BUY'
        qty_abs = abs(qty)
        qty_buffered = self._apply_order_value_buffer(qty_abs, price or 0)
        if qty_buffered <= 0:
            telemetry_record["status"] = "close_position_zero_after_buffer"
            self._emit_telemetry(telemetry_record)
            return
        risk_result = self.risk_manager.check_trade_allowed(symbol, action, qty_buffered, price or 0)
        if not risk_result.allowed:
            telemetry_record["status"] = "close_position_blocked"
            telemetry_record["risk_reason"] = risk_result.reason
            self._emit_telemetry(telemetry_record)
            return
        try:
            prefer_maker = self._prefer_maker(symbol)
            order_result = await self.bot.place_order_async(symbol, action, qty_buffered, prefer_maker=prefer_maker)
            liquidity_tag = order_result.get('liquidity', 'taker') if order_result else 'taker'
            fee = self.cost_tracker.calculate_trade_fee(symbol, qty_buffered, price or 0, action, liquidity=liquidity_tag)
            realized = self._update_holdings_and_realized(symbol, action, qty_buffered, price or 0, fee)
            self.db.log_trade(
                self.session_id,
                symbol,
                action,
                qty_buffered,
                price or 0,
                fee,
                f"Close position request ({qty})",
                liquidity=order_result.get('liquidity') if order_result else 'taker',
                realized_pnl=realized,
            )
            self._apply_fill_to_session_stats(order_result.get('order_id') if order_result else None, fee, realized)
            telemetry_record["status"] = "close_position_executed"
            telemetry_record["order_result"] = order_result
        except Exception as e:
            telemetry_record["status"] = "close_position_error"
            telemetry_record["error"] = str(e)
            logger.error(f"Close position failed: {e}")
        self._log_execution_trace(trace_id, telemetry_record)
        self._emit_telemetry(telemetry_record)

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
            self._pause_until = asyncio.get_event_loop().time() + pause_seconds
            telemetry_record["status"] = "paused"
            telemetry_record["pause_seconds"] = pause_seconds
            self._emit_telemetry(telemetry_record)


    async def _capture_ohlcv(self, symbol: str):
        """Fetch multi-timeframe OHLCV for the active symbol and persist."""
        if not hasattr(self.bot, "fetch_ohlcv"):
            return
        timeframes = ['1m', '5m', '1h', '1d']
        for tf in timeframes:
            try:
                bars = await self.bot.fetch_ohlcv(symbol, timeframe=tf, limit=50)
                self.db.log_ohlcv_batch(self.session_id, symbol, tf, bars)
            except Exception as e:
                logger.debug(f"OHLCV fetch failed for {symbol} {tf}: {e}")

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

    def _liquidity_ok(self, market_data_point: dict) -> bool:
        """Simple microstructure filters using spread and top-of-book depth."""
        if not market_data_point:
            return True

        spread_pct = market_data_point.get('spread_pct')
        bid = market_data_point.get('bid')
        ask = market_data_point.get('ask')
        bid_size = market_data_point.get('bid_size')
        ask_size = market_data_point.get('ask_size')

        if spread_pct is None and bid and ask:
            mid = (bid + ask) / 2
            if mid:
                spread_pct = ((ask - bid) / mid) * 100

        if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
            bot_actions_logger.info(f"⏸️ Skipping trade: spread {spread_pct:.3f}% > cap {MAX_SPREAD_PCT:.3f}%")
            return False

        if bid and ask and bid_size and ask_size:
            # Use the weaker side as liquidity floor
            min_notional = min(bid * bid_size, ask * ask_size)
            if min_notional < MIN_TOP_OF_BOOK_NOTIONAL:
                bot_actions_logger.info(
                    f"⏸️ Skipping trade: top-of-book notional ${min_notional:.2f} < ${MIN_TOP_OF_BOOK_NOTIONAL:.2f} floor"
                )
                return False

        return True

    async def _monitor_trade_plans(self, price_lookup: dict, open_orders: list):
        """
        Monitor open trade plans for stop/target hits and enforce max age/day-end flattening per symbol.
        price_lookup: symbol -> latest price
        open_orders: list of open orders from exchange
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

            open_orders_by_symbol = {}
            for o in open_orders or []:
                sym = o.get('symbol')
                if not sym:
                    continue
                open_orders_by_symbol.setdefault(sym, []).append(o)

            for plan in open_plans:
                plan_id = plan['id']
                side = plan['side'].upper()
                regime_flags = plan.get('regime_flags') or {}
                stop = plan.get('stop_price')
                target = plan.get('target_price')
                size = plan.get('size') or 0.0
                symbol = plan.get('symbol')
                price_now = (price_lookup or {}).get(symbol)
                entry = plan.get('entry_price') or price_now
                version = plan.get('version') or 1
                if not price_now or size <= 0:
                    continue

                should_close = False
                reason = None

                # Close stale plans with no position and no open orders for the symbol
                pos_qty = (self.risk_manager.positions or {}).get(symbol, {}).get('quantity', 0.0) or 0.0
                has_open_orders = bool(open_orders_by_symbol.get(symbol))
                if abs(pos_qty) < 1e-9 and not has_open_orders:
                    should_close = True
                    reason = "Plan closed: position flat and no open orders"

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
                    # Trail stop to entry after move in favor
                    if stop and price_now >= entry * (1 + self._apply_plan_trailing_pct) and stop < entry:
                        try:
                            self.db.update_trade_plan_prices(plan_id, stop_price=entry, reason="Trailed stop to breakeven")
                            bot_actions_logger.info(f"↩️ Trailed stop to breakeven for plan {plan_id} (v{version}→v{version+1})")
                            stop = entry
                        except Exception as e:
                            logger.debug(f"Could not trail stop for plan {plan_id}: {e}")
                    # Apply volatility-aware trailing: widen on low vol, tighten on high vol
                    try:
                        vol_flag = (plan.get('volatility') or regime_flags.get('volatility') or '').lower()
                    except Exception:
                        vol_flag = ''
                    trail_pct = self._apply_plan_trailing_pct
                    if vol_flag:
                        if 'low' in vol_flag:
                            trail_pct *= 1.5
                        elif 'high' in vol_flag:
                            trail_pct *= 0.7
                    if stop and price_now <= stop:
                        should_close = True
                        reason = f"Stop hit at ${price_now:,.2f}"
                    elif target and price_now >= target:
                        should_close = True
                        reason = f"Target hit at ${price_now:,.2f}"
                else:  # SELL plan (short)
                    if stop and price_now <= entry * (1 - self._apply_plan_trailing_pct) and stop > entry:
                        try:
                            self.db.update_trade_plan_prices(plan_id, stop_price=entry, reason="Trailed stop to breakeven")
                            bot_actions_logger.info(f"↩️ Trailed stop to breakeven for plan {plan_id} (v{version}→v{version+1})")
                            stop = entry
                        except Exception as e:
                            logger.debug(f"Could not trail stop for plan {plan_id}: {e}")
                    try:
                        vol_flag = (plan.get('volatility') or regime_flags.get('volatility') or '').lower()
                    except Exception:
                        vol_flag = ''
                    trail_pct = self._apply_plan_trailing_pct
                    if vol_flag:
                        if 'low' in vol_flag:
                            trail_pct *= 1.5
                        elif 'high' in vol_flag:
                            trail_pct *= 0.7
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
                        prefer_maker = self._prefer_maker(plan['symbol'])
                        order_result = await self.bot.place_order_async(plan['symbol'], action, size, prefer_maker=prefer_maker)
                        liquidity_tag = order_result.get('liquidity', 'taker') if order_result else 'taker'
                        fee = self.cost_tracker.calculate_trade_fee(plan['symbol'], size, price_now, action, liquidity=liquidity_tag)
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
                        self._apply_fill_to_session_stats(
                            order_result.get('order_id') if order_result else None,
                            fee,
                            realized,
                        )
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

        # Initialize tool coordinator after exchange connection
        if getattr(self.bot, "exchange", None):
            self.data_fetch_coordinator = DataFetchCoordinator(
                self.bot.exchange,
                error_callback=self._record_tool_failure,
                success_callback=self._record_tool_success,
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
        
        # Initialize trading context
        self.context = TradingContext(self.db, self.session_id)
        # Drop any stale open orders lingering from prior runs and sync with venue
        await self._reconcile_exchange_state()

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
            
            # 2. Fetch trades for the day across active symbols
            try:
                symbols = [s for s in getattr(self.bot.exchange, 'symbols', []) or [] if '/USD' in s]
            except Exception:
                symbols = []
            if not symbols:
                symbols = ['BTC/USD']
            trades = []
            for sym in symbols:
                sym_trades = await self.bot.get_trades_from_timestamp(sym, start_ts_ms)
                trades.extend(sym_trades)
            
            # 3. Rebuild Holdings and Stats
            self._apply_exchange_trades_for_rebuild(trades)

            # Load LLM costs from DB (since exchange doesn't track this)
            db_stats = self.db.get_session_stats(self.session_id)
            self.session_stats['total_llm_cost'] = db_stats.get('total_llm_cost', 0.0)
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
        """
        Maintain in-memory holdings to compute realized PnL per trade.
        Realized PnL is fee-exclusive so costs are handled exactly once in aggregates.
        """
        pos = self.holdings.get(symbol, {'qty': 0.0, 'avg_cost': 0.0})
        qty = pos['qty']
        avg_cost = pos['avg_cost']
        realized = 0.0

        if action == 'BUY':
            new_qty = qty + quantity
            new_avg = ((qty * avg_cost) + (quantity * price)) / new_qty if new_qty > 0 else 0.0
            self.holdings[symbol] = {'qty': new_qty, 'avg_cost': new_avg}
            realized = 0.0
        else:  # SELL
            realized = (price - avg_cost) * quantity
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
        if TRADING_MODE == 'PAPER':
            return
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
            try:
                fee_cost = self._extract_fee_cost(t.get('fee'))
                realized = self._update_holdings_and_realized(
                    t['symbol'],
                    t['action'],
                    t['quantity'],
                    t['price'],
                    fee_cost,
                )
                self.session_stats['total_trades'] += 1
                self.session_stats['total_fees'] += fee_cost
                self.session_stats['gross_pnl'] += realized
            except Exception as exc:
                logger.warning(f"Skipping trade during stats rebuild due to error: {exc}")

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
                try:
                    if max_loops is not None and loops >= max_loops:
                        break
                    exchange_error_seen = False
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
                    try:
                        current_equity = await self.bot.get_equity_async()
                    except Exception as e:
                        logger.warning(f"Could not fetch equity: {e}; skipping loop iteration.")
                        self._record_exchange_failure("get_equity_async", e)
                        exchange_error_seen = True
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue
                    
                    if current_equity is None:
                        logger.warning("Could not fetch equity; skipping loop iteration to avoid false loss triggers.")
                        self._record_exchange_failure("get_equity_async", "none")
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
                    if self._pause_until and now_monotonic < self._pause_until:
                        bot_actions_logger.info(f"⏸️ Trading paused for {self._pause_until - now_monotonic:.0f}s")
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
                            self._record_exchange_failure("get_market_data_async", e)
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

                    if primary_fetch_failed or not market_data.get(primary_symbol):
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    primary_data = market_data.get(primary_symbol)
                    stale, freshness_detail = self._is_stale_market_data(primary_data)
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
                        self._record_exchange_failure("get_positions_async", e)
                        exchange_error_seen = True
                    # Refresh open orders for exposure headroom and context
                    open_orders = []
                    try:
                        open_orders = await self.bot.get_open_orders_async()
                        open_orders = self._filter_our_orders(open_orders)
                        self.db.replace_open_orders(self.session_id, open_orders)
                    except Exception as e:
                        logger.warning(f"Could not refresh open orders: {e}")
                        self._record_exchange_failure("get_open_orders_async", e)
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
                    except Exception as e:
                        logger.warning(f"Could not build positions for exposure: {e}")

                    # 2.8 Monitor trade plans for stops/targets and max age
                    try:
                        await self._monitor_trade_plans(price_lookup=price_lookup, open_orders=open_orders)
                    except Exception as e:
                        logger.exception(f"Trade plan monitor error: {e}")

                    # Kill switch check
                    if self._kill_switch:
                        bot_actions_logger.info("🛑 Kill switch active; not trading.")
                        await asyncio.sleep(LOOP_INTERVAL_SECONDS)
                        continue

                    # Slippage guard: if latest price moved >2% from decision price, skip execution
                    decision_price = market_data.get(primary_symbol, {}).get('price')

                    # 2.5 Sync Trades from Exchange (for logging only)
                    try:
                        await self.sync_trades_from_exchange()
                    except Exception as e:
                        logger.warning(f"Trade sync failed: {e}")
                        self._record_exchange_failure("sync_trades_from_exchange", e)
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
                            self._pause_until = asyncio.get_event_loop().time() + pause_seconds
                            telemetry_record["status"] = "paused"
                            telemetry_record["pause_seconds"] = pause_seconds
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
                                    self._record_exchange_failure("place_order_async", "timeout")
                                    exchange_error_seen = True
                                    await self._reconnect_bot()
                                    retries += 1
                                except Exception as e:
                                    logger.error(f"Order placement error: {e}")
                                    self._record_exchange_failure("place_order_async", e)
                                    exchange_error_seen = True
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
                        self._reset_exchange_errors()
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
