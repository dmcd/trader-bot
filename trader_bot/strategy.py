import asyncio
import json
import math
import re
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone
from statistics import pstdev
from typing import Any, Dict, List, Optional
from types import SimpleNamespace

import google.generativeai as genai
import jsonschema
from pydantic import BaseModel, ValidationError
from abc import ABC, abstractmethod
from openai import OpenAI

from trader_bot.config import (
    ACTIVE_EXCHANGE,
    AUTO_REPLACE_PLAN_ON_CAP,
    BREAK_GLASS_COOLDOWN_MIN,
    FEE_RATIO_COOLDOWN,
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_MAX_CONSECUTIVE_ERRORS,
    LLM_MAX_SESSION_COST,
    LLM_MIN_CALL_INTERVAL_SECONDS,
    LLM_PROVIDER,
    LOOP_INTERVAL_SECONDS,
    MAX_DAILY_LOSS,
    MAX_DAILY_LOSS_PERCENT,
    MAX_ORDER_VALUE,
    MAX_POSITIONS,
    MAX_SPREAD_PCT,
    MAX_TOTAL_EXPOSURE,
    MIN_TRADE_INTERVAL_SECONDS,
    MIN_TRADE_SIZE,
    PLAN_MAX_PER_SYMBOL,
    PRIORITY_LOOKBACK_MIN,
    PRIORITY_MOVE_PCT,
    LLM_DECISION_BYTE_BUDGET,
    TOOL_DEFAULT_TIMEFRAMES,
    TOOL_MAX_BARS,
    TOOL_MAX_DEPTH,
    TOOL_MAX_JSON_BYTES,
    TOOL_MAX_TRADES,
    TRADING_MODE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from trader_bot.llm_tools import TOOL_PARAM_MODELS, ToolName, ToolRequest, ToolResponse

logger = logging.getLogger(__name__)
telemetry_logger = logging.getLogger('telemetry')
bot_actions_logger = logging.getLogger('bot_actions')

class StrategySignal:
    def __init__(self, action: str, symbol: str, quantity: float, reason: str, order_id=None, trace_id: int = None, regime_flags: Dict[str, str] = None):
        self.action = action.upper()
        self.symbol = symbol
        self.quantity = quantity
        self.reason = reason
        self.order_id = order_id
        self.trace_id = trace_id
        self.regime_flags = regime_flags or {}

    def __str__(self):
        return f"{self.action} {self.quantity} {self.symbol} ({self.reason})"


class _LLMResponse:
    """Lightweight wrapper to normalize LLM responses across providers."""

    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=prompt_tokens or 0,
            candidates_token_count=completion_tokens or 0,
        )
        self.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )


class BaseStrategy(ABC):
    def __init__(self, db, technical_analysis, cost_tracker):
        self.db = db
        self.ta = technical_analysis
        self.cost_tracker = cost_tracker

    @abstractmethod
    async def generate_signal(self, session_id: int, market_data: Dict[str, Any], current_equity: float, current_exposure: float, context: Any = None) -> Optional[StrategySignal]:
        """
        Analyze market data and return a trading signal.
        """
        pass

class LLMStrategy(BaseStrategy):
    def __init__(self, db, technical_analysis, cost_tracker, open_orders_provider=None, ohlcv_provider=None, tool_coordinator=None):
        super().__init__(db, technical_analysis, cost_tracker)
        self.system_prompt = self._load_system_prompt()
        self.llm_provider = (LLM_PROVIDER or "GEMINI").upper()
        if self.llm_provider not in {"GEMINI", "OPENAI"}:
            logger.warning("Unknown LLM_PROVIDER %s, defaulting to GEMINI", self.llm_provider)
            self.llm_provider = "GEMINI"

        self.llm_model = LLM_MODEL
        self._llm_ready = False
        self._openai_client = None
        self.model = None

        if self.llm_provider == "OPENAI":
            if OPENAI_API_KEY:
                client_kwargs = {"api_key": OPENAI_API_KEY}
                if OPENAI_BASE_URL:
                    client_kwargs["base_url"] = OPENAI_BASE_URL
                self._openai_client = OpenAI(**client_kwargs)
                self._llm_ready = True
            else:
                logger.warning("OPENAI_API_KEY not found. LLMStrategy will not work.")
        else:
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel(self.llm_model, system_instruction=self.system_prompt)
                self._llm_ready = True
            else:
                logger.warning("GEMINI_API_KEY not found. LLMStrategy will not work.")
        
        self.last_trade_ts = None
        self._last_break_glass = 0.0
        self.last_rejection_reason = None
        # Optional async callable that fetches live open orders from the active exchange
        self.open_orders_provider = open_orders_provider
        # Optional async callable that fetches OHLCV data
        self.ohlcv_provider = ohlcv_provider
        # Optional tool coordinator for LLM tool requests
        self.tool_coordinator = tool_coordinator
        self.prompt_template = self._load_prompt_template()
        self._last_llm_call_ts = 0.0
        self._consecutive_llm_errors = 0
        self._decision_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "CANCEL", "UPDATE_PLAN", "PARTIAL_CLOSE", "CLOSE_POSITION", "PAUSE_TRADING"]},
                "symbol": {"type": ["string", "null"]},
                "quantity": {"type": ["number", "null"], "minimum": 0},
                "reason": {"type": "string"},
                "order_id": {"type": ["string", "number", "null"]},
                "stop_price": {"type": ["number", "null"], "minimum": 0},
                "target_price": {"type": ["number", "null"], "minimum": 0},
                "plan_id": {"type": ["integer", "null"]},
                "size_factor": {"type": ["number", "null"], "minimum": 0},
                "close_fraction": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "duration_minutes": {"type": ["number", "null"], "minimum": 1}
            },
            "required": ["action", "symbol", "reason"],
            "additionalProperties": False
        }

    def _extract_json_payload(self, text: str) -> str:
        """Extract a JSON object from an LLM response that may include chatter."""
        if not text:
            return ""

        cleaned = text.strip()

        # Prefer an explicit fenced code block
        block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if block:
            return block.group(1).strip()

        # Fall back to the first brace-delimited object
        first_brace = cleaned.find('{')
        if first_brace != -1:
            last_brace = cleaned.rfind('}')
            while last_brace != -1 and last_brace >= first_brace:
                candidate = cleaned[first_brace:last_brace + 1].strip()
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    # Try an earlier closing brace in case we overshot
                    last_brace = cleaned.rfind('}', first_brace, last_brace)
        return cleaned

    def _is_choppy(self, symbol: str, market_data_point, recent_data):
        """
        Simple regime filter: skip when Bollinger width is tight and RSI is neutral.
        """
        try:
            if not recent_data or len(recent_data) < 20:
                return False
            indicators = self.ta.calculate_indicators(recent_data)
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

    def _priority_signal(self, session_id: int, symbol: str, context: Any = None) -> bool:
        """
        Detect breakout move to allow a break-glass trade inside cooldown.
        """
        try:
            lookback_points = PRIORITY_LOOKBACK_MIN * 60 // LOOP_INTERVAL_SECONDS + 2
            
            before_ts = None
            if context and hasattr(context, 'current_iso_time'):
                before_ts = context.current_iso_time
                
            recent = self.db.get_recent_market_data(session_id, symbol, limit=max(lookback_points, 10), before_timestamp=before_ts)
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

    def _fees_too_high(self, session_stats: Dict[str, Any]):
        """
        Check fee ratio vs gross PnL; returns True to pause when fees dominate.
        """
        try:
            if not isinstance(session_stats, dict):
                logger.warning(f"Invalid session_stats type: {type(session_stats)}")
                return False
                
            gross = session_stats.get('gross_pnl', 0) or 0
            fees = session_stats.get('total_fees', 0) or 0
            if gross == 0:
                return False
            ratio = (fees / abs(gross)) * 100
            return ratio >= FEE_RATIO_COOLDOWN
        except Exception as e:
            logger.warning(f"Fee ratio check failed: {e}")
            return False

    def _build_timeframe_summary(self, session_id: int, symbol: str) -> str:
        """Summarize multi-timeframe OHLCV into a concise string for the prompt."""
        if not hasattr(self.db, "get_recent_ohlcv"):
            return ""
        timeframes = ['1m', '5m', '1h', '1d']
        lines = []
        for tf in timeframes:
            try:
                bars = self.db.get_recent_ohlcv(session_id, symbol, tf, limit=50)
                if not bars or len(bars) < 2:
                    continue
                closes = [b.get('close') for b in bars if b.get('close') is not None]
                vols = [b.get('volume') for b in bars if b.get('volume') is not None]
                if not closes:
                    continue
                last = closes[0]
                first = closes[-1]
                change_pct = ((last - first) / first * 100) if first else None
                returns = []
                for i in range(1, len(closes)):
                    if closes[i]:
                        returns.append((closes[i-1] - closes[i]) / closes[i])
                vol_pct = (pstdev(returns) * 100) if len(returns) >= 2 else None
                avg_vol = sum(vols) / len(vols) if vols else None
                parts = [f"{tf}: last ${last:,.2f}"]
                if change_pct is not None:
                    parts.append(f"Δ{change_pct:+.2f}%")
                if vol_pct is not None:
                    parts.append(f"vol {vol_pct:.2f}%σ")
                if avg_vol is not None:
                    parts.append(f"avg vol {avg_vol:.2f}")
                lines.append(", ".join(parts))
            except Exception as e:
                logger.debug(f"TF summary failed for {symbol} {tf}: {e}")
        if not lines:
            return ""
        return "Multi-timeframe: " + " | ".join(lines)

    def _compute_regime_flags(self, session_id: int, symbol: str, market_data_point: Dict[str, Any], recent_bars: Dict[str, list]) -> Dict[str, str]:
        """Derive simple regime flags: volatility bucket, trend slope, liquidity."""
        flags = {}

        # Volatility bucket using 1h bars
        one_h = recent_bars.get('1h') or []
        if len(one_h) >= 10:
            closes = [b.get('close') for b in one_h if b.get('close') is not None]
            returns = []
            for i in range(1, len(closes)):
                if closes[i]:
                    returns.append((closes[i] - closes[i-1]) / closes[i-1])
            if len(returns) >= 5:
                vol_pct = pstdev(returns) * 100
                if vol_pct < 0.5:
                    flags['volatility'] = f"low ({vol_pct:.2f}%)"
                elif vol_pct < 1.5:
                    flags['volatility'] = f"medium ({vol_pct:.2f}%)"
                else:
                    flags['volatility'] = f"high ({vol_pct:.2f}%)"

        # Trend slope using 1h closes over ~12 bars
        if len(one_h) >= 6:
            closes = [b.get('close') for b in one_h if b.get('close') is not None][:12]
            if len(closes) >= 2:
                x = list(range(len(closes)))
                avg_x = sum(x) / len(x)
                avg_y = sum(closes) / len(closes)
                num = sum((xi - avg_x) * (yi - avg_y) for xi, yi in zip(x, closes))
                den = sum((xi - avg_x) ** 2 for xi in x) or 1
                slope = num / den
                last = closes[0]
                slope_pct = (slope / last * 100) if last else 0
                if slope_pct > 0.05:
                    flags['trend'] = f"up ({slope_pct:.2f}%/bar)"
                elif slope_pct < -0.05:
                    flags['trend'] = f"down ({slope_pct:.2f}%/bar)"
                else:
                    flags['trend'] = "flat"

        # Liquidity flag from top-of-book and spread
        spread_pct = market_data_point.get('spread_pct')
        bid_size = market_data_point.get('bid_size')
        ask_size = market_data_point.get('ask_size')
        if spread_pct is not None:
            if spread_pct > MAX_SPREAD_PCT:
                flags['liquidity'] = f"wide_spread ({spread_pct:.3f}%)"
            else:
                flags['liquidity'] = f"ok_spread ({spread_pct:.3f}%)"
        if bid_size and ask_size:
            min_notional = min(
                (market_data_point.get('bid') or 0) * bid_size,
                (market_data_point.get('ask') or 0) * ask_size
            )
            flags['depth'] = f"top_notional ${min_notional:,.2f}"

        return flags

    async def generate_signal(self, session_id: int, market_data: Dict[str, Any], current_equity: float, current_exposure: float, context: Any = None, session_stats: Dict[str, Any] = None) -> Optional[StrategySignal]:
        if not market_data:
            return None

        # Assuming single symbol focus for now as per original code
        symbol = list(market_data.keys())[0]
        data = market_data[symbol]

        # Clock for cooldowns and LLM throttling
        now_ts = asyncio.get_event_loop().time()

        # LLM cost/frequency guards (HOLD instead of burning tokens)
        total_llm_cost = (session_stats or {}).get('total_llm_cost', 0.0) if session_stats else 0.0
        if total_llm_cost >= LLM_MAX_SESSION_COST:
            logger.info(f"Skipping LLM call: session LLM cost ${total_llm_cost:.4f} exceeds cap ${LLM_MAX_SESSION_COST:.2f}")
            return StrategySignal('HOLD', symbol, 0, 'LLM cost cap hit')
        if self._last_llm_call_ts and (now_ts - self._last_llm_call_ts) < LLM_MIN_CALL_INTERVAL_SECONDS:
            logger.info("Skipping LLM call: min call interval not met")
            return StrategySignal('HOLD', symbol, 0, 'LLM call throttled')

        regime_flags = {}

        # 1. Fee Check
        if session_stats and self._fees_too_high(session_stats):
            logger.info("Skipping trading due to high fee ratio")
            return None

        # 2. Chop Check
        # Use ohlcv_provider if available, else fallback to DB (or skip)
        recent_data = []
        if self.ohlcv_provider:
            try:
                # Fetch 1m candles for chop check
                recent_data = await self.ohlcv_provider(symbol, timeframe='1m', limit=50)
            except Exception as e:
                logger.warning(f"OHLCV fetch failed: {e}")
        
        if not recent_data:
             # Fallback to DB if provider fails or not set
             # Use context-provided time for DB lookups if available
            before_ts = None
            if context and hasattr(context, 'current_iso_time'):
                before_ts = context.current_iso_time
            recent_data = self.db.get_recent_market_data(session_id, symbol, limit=50, before_timestamp=before_ts)

        if self._is_choppy(symbol, data, recent_data):
            logger.info("Skipping trade: market in chop")
            return None

        # 3. Timing / Priority Check
        # Use context-provided time override when available
        if context and hasattr(context, 'current_time'):
            now_ts = context.current_time
            
        can_trade = not self.last_trade_ts or (now_ts - self.last_trade_ts) >= MIN_TRADE_INTERVAL_SECONDS
        priority = self._priority_signal(session_id, symbol, context)
        allow_break_glass = priority and (now_ts - self._last_break_glass) >= (BREAK_GLASS_COOLDOWN_MIN * 60)

        if not can_trade and not allow_break_glass:
            return None

        # 4. Build Prompt
        open_orders = []
        if self.open_orders_provider:
            try:
                open_orders = await self.open_orders_provider()
            except Exception as e:
                logger.warning(f"Open order fetch failed; falling back to DB: {e}")

        if not open_orders:
            try:
                open_orders = self.db.get_open_orders(session_id)
            except Exception as e:
                logger.warning(f"DB open order fetch failed: {e}")
                open_orders = []

        last_trade_age = (now_ts - self.last_trade_ts) if self.last_trade_ts else None
        last_trade_age_str = f"{last_trade_age:.0f}s" if last_trade_age is not None else "n/a"
        fee_ratio_flag = "high" if (session_stats and self._fees_too_high(session_stats)) else "normal"
        priority_flag = "true" if priority and allow_break_glass else "false"
        spacing_flag = "clear" if can_trade else f"cooldown {MIN_TRADE_INTERVAL_SECONDS - (now_ts - self.last_trade_ts):.0f}s"
        
        order_cap_value = MAX_ORDER_VALUE
        exposure_cap = MAX_TOTAL_EXPOSURE
        exposure_now = current_exposure if current_exposure is not None else 0.0
        equity_now = current_equity if current_equity is not None else 0.0
        headroom = max(0.0, exposure_cap - exposure_now)
        pending_buy_exposure = 0.0
        open_order_counts = {}
        for order in open_orders or []:
            sym = order.get('symbol')
            open_order_counts[sym] = open_order_counts.get(sym, 0) + 1
            if (order.get('side') or '').upper() != 'BUY':
                continue
            qty = order.get('remaining')
            if qty is None:
                qty = order.get('amount', 0.0)
            px = order.get('price')
            if px and qty:
                pending_buy_exposure += px * qty
        logger.info(
            f"Exposure snapshot: exposure_now=${exposure_now:,.2f}, cap=${exposure_cap:,.2f}, headroom=${headroom:,.2f}, "
            f"open_orders={len(open_orders) if open_orders else 0}, equity=${equity_now:,.2f}"
        )

        open_order_count = len(open_orders)
        open_order_snippets = []
        for order in open_orders[:5]:
            side = (order.get('side') or '').upper() or 'N/A'
            sym = order.get('symbol') or 'Unknown'
            qty = order.get('amount') or 0
            remaining = order.get('remaining') if order.get('remaining') is not None else qty
            price = order.get('price')
            price_str = f"${price:,.2f}" if price else "mkt"
            order_id = order.get('order_id')
            id_str = f"#{order_id}" if order_id is not None else "#?"
            open_order_snippets.append(f"{id_str} {side} {qty:.4f} {sym} @ {price_str} (rem {remaining:.4f})")
        open_orders_summary = "; ".join(open_order_snippets) if open_order_snippets else "none"

        # Per-symbol plan counts to expose cap usage to the LLM
        plan_counts = {}
        try:
            open_plans = self.db.get_open_trade_plans(session_id)
            for plan in open_plans:
                sym = plan.get('symbol')
                if not sym:
                    continue
                plan_counts[sym] = plan_counts.get(sym, 0) + 1
        except Exception as exc:
            logger.debug(f"Could not fetch plan counts: {exc}")
        plan_counts_str = ", ".join(f"{sym}:{cnt}/{PLAN_MAX_PER_SYMBOL}" for sym, cnt in plan_counts.items()) or "none"

        prompt_context = (
            f"- Cooldown: {spacing_flag}\n"
            f"- Priority signal allowed: {priority_flag}\n"
            f"- Fee regime: {fee_ratio_flag}\n"
            f"- Last trade age: {last_trade_age_str}\n"
            f"- Equity: ${equity_now:,.2f}\n"
            f"- Exposure: ${exposure_now:,.2f} of ${exposure_cap:,.2f} (room ${headroom:,.2f})\n"
            f"- Pending BUY exposure: ${pending_buy_exposure:,.2f}\n"
            f"- Max open orders per symbol: {MAX_POSITIONS}\n"
            f"- Plans per symbol (used/cap): {plan_counts_str}\n"
            f"{'- Plan cap reached: you must UPDATE_PLAN/PARTIAL_CLOSE/CANCEL or rely on auto-replace if enabled.\n' if any(count >= PLAN_MAX_PER_SYMBOL for count in plan_counts.values()) else ''}"
            f"- Order cap: ${order_cap_value:.2f}\n"
            f"- Min trade size: ${MIN_TRADE_SIZE:.2f}\n"
            f"- Open orders: {open_order_count} ({open_orders_summary})"
        )

        decision_result = await self._get_llm_decision(
            session_id,
            market_data,
            current_equity,
            prompt_context,
            context,
            open_orders=open_orders,
            headroom=headroom,
            pending_buy_exposure=pending_buy_exposure,
            can_trade=can_trade,
            spacing_flag=spacing_flag,
            plan_counts=plan_counts,
        )
        trace_id = None
        decision_json = None
        if isinstance(decision_result, tuple):
            # New API returns (decision_json, trace_id)
            decision_json, trace_id = decision_result
        else:
            # Backward compatibility for tests/mocks returning only JSON
            decision_json = decision_result
        
        if decision_json:
            try:
                # Successful call; record timing and reset error counter
                self._last_llm_call_ts = now_ts
                self._consecutive_llm_errors = 0
                decision = json.loads(decision_json)
                # Validate shape
                try:
                    jsonschema.validate(decision, self._decision_schema)
                except Exception as e:
                    logger.error(f"LLM decision failed schema validation: {e}")
                    try:
                        self.db.log_llm_call(session_id, 0, 0, 0.0, f"schema_error:{str(e)}")
                    except Exception:
                        pass
                    return None
                action = decision.get('action')
                quantity = decision.get('quantity')
                if quantity is None:
                    quantity = 0
                reason = decision.get('reason')
                order_id = decision.get('order_id')
                stop_price = decision.get('stop_price')
                target_price = decision.get('target_price')

                if action in ['BUY', 'SELL'] and quantity > 0:
                    # Update state if we are signaling a trade
                    if allow_break_glass and not can_trade:
                         self._last_break_glass = now_ts

                    price = data.get('price') if data else None

                    if action == 'BUY':
                        max_order_value = min(order_cap_value, headroom)
                        if not price or max_order_value <= 0:
                            return StrategySignal('HOLD', symbol, 0, 'No exposure headroom')
                        quantity = self._clamp_quantity(quantity, price, headroom)
                    else:
                        # SELL: still clamp to order cap for safety
                        if price:
                            quantity = self._clamp_quantity(quantity, price, order_cap_value)

                    # Clamp stops/targets to reasonable bounds around price
                    stop_price = decision.get('stop_price')
                    target_price = decision.get('target_price')
                    if price:
                        band = price * 0.02  # 2% band
                        if action == 'BUY':
                            min_stop = max(0.0, price - band)
                            max_target = price + band
                            if stop_price is not None:
                                stop_price = max(min_stop, min(stop_price, price))  # stop stays below/at entry
                            if target_price is not None:
                                target_price = min(max_target, max(target_price, price))  # target at/above entry
                        else:  # SELL (short)
                            max_stop = price + band
                            min_target = max(0.0, price - band)
                            if stop_price is not None:
                                stop_price = min(max_stop, max(stop_price, price))  # stop stays above/at entry
                            if target_price is not None:
                                target_price = max(min_target, min(target_price, price))  # target at/below entry
                        # Telemetry for clamping
                        if decision.get('stop_price') != stop_price or decision.get('target_price') != target_price:
                            clamp_msg = f"clamped: stop {decision.get('stop_price')} -> {stop_price}, target {decision.get('target_price')} -> {target_price}"
                            logger.info(f"LLM stops/targets {clamp_msg}")
                            try:
                                self.db.log_llm_call(session_id, 0, 0, 0.0, clamp_msg)
                            except Exception:
                                pass

                    # We don't update last_trade_ts here, we let the runner do it upon execution success? 
                    # Actually better to update it here to prevent double signaling if execution takes time, 
                    # BUT runner might reject it. 
                    # Let's stick to the pattern: Strategy suggests, Runner executes. 
                    # Strategy state should ideally update on confirmation.
                    # For now, we'll leave state update to the caller or handle it optimistically?
                    # The original code checked `can_trade` in the loop.
                    
                    signal = StrategySignal(action, symbol, quantity, reason, trace_id=trace_id, regime_flags=regime_flags)
                    # Carry stop/target forward to runner via attributes
                    signal.stop_price = stop_price
                    signal.target_price = target_price
                    return signal
                elif action == 'CANCEL' and order_id:
                    return StrategySignal('CANCEL', symbol or '', 0, reason or 'Cancel open order', order_id=order_id, trace_id=trace_id, regime_flags=regime_flags)
                elif action == 'UPDATE_PLAN':
                    plan_id = decision.get('plan_id')
                    sig = StrategySignal('UPDATE_PLAN', symbol, 0, reason or 'Update plan', order_id=plan_id, trace_id=trace_id, regime_flags=regime_flags)
                    sig.plan_id = plan_id
                    sig.size_factor = decision.get('size_factor')
                    sig.stop_price = stop_price
                    sig.target_price = target_price
                    return sig
                elif action == 'PARTIAL_CLOSE':
                    plan_id = decision.get('plan_id')
                    close_fraction = decision.get('close_fraction', 0.0)
                    sig = StrategySignal('PARTIAL_CLOSE', symbol, 0, reason or 'Partial close', order_id=plan_id, trace_id=trace_id, regime_flags=regime_flags)
                    sig.plan_id = plan_id
                    sig.close_fraction = close_fraction
                    return sig
                elif action == 'CLOSE_POSITION':
                    sig = StrategySignal('CLOSE_POSITION', symbol, 0, reason or 'Close position', trace_id=trace_id, regime_flags=regime_flags)
                    return sig
                elif action == 'PAUSE_TRADING':
                    pause_symbol = decision.get('symbol')
                    sig = StrategySignal('PAUSE_TRADING', pause_symbol, 0, reason or 'Pause trading', trace_id=trace_id, regime_flags=regime_flags)
                    sig.duration_minutes = decision.get('duration_minutes')
                    return sig
                elif action == 'HOLD':
                    return StrategySignal('HOLD', symbol, 0, reason, trace_id=trace_id, regime_flags=regime_flags)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {decision_json}")
                self._consecutive_llm_errors += 1
        else:
            self._consecutive_llm_errors += 1

        if self._consecutive_llm_errors >= LLM_MAX_CONSECUTIVE_ERRORS:
            logger.info(f"LLM consecutive errors {self._consecutive_llm_errors} >= cap; returning HOLD")
            return StrategySignal('HOLD', symbol, 0, 'LLM errors')

        return None


    def _load_prompt_template(self) -> str:
        """Load template from adjacent file to make manual edits easy."""
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            return LLMStrategy._prompt_template_cache

        template_path = Path(__file__).with_name("llm_prompt_template.txt")
        template_text = template_path.read_text()
        LLMStrategy._prompt_template_cache = template_text
        return template_text

    def _load_system_prompt(self) -> str:
        """Load system prompt (static instructions) from adjacent file."""
        if hasattr(LLMStrategy, "_system_prompt_cache"):
            return LLMStrategy._system_prompt_cache
        system_path = Path(__file__).with_name("llm_system_prompt.txt")
        system_text = system_path.read_text()
        LLMStrategy._system_prompt_cache = system_text
        return system_text

    def _build_prompt(self, **kwargs) -> str:
        """Render prompt with a forgiving formatter so optional fields can be empty."""

        class _SafeDict(dict):
            def __missing__(self, key):
                return ""

        return self.prompt_template.format_map(_SafeDict(**kwargs))

    async def _invoke_llm(self, prompt: str, timeout: int = 30):
        """Invoke the LLM with a timeout."""
        if self.llm_provider == "OPENAI":
            return await asyncio.wait_for(
                asyncio.to_thread(self._call_openai, prompt),
                timeout=timeout,
            )

        return await asyncio.wait_for(
            asyncio.to_thread(self._call_gemini, prompt),
            timeout=timeout,
        )

    def _call_gemini(self, prompt: str):
        if not self.model:
            raise RuntimeError("Gemini model is not configured")
        return self.model.generate_content(prompt)

    def _call_openai(self, prompt: str):
        if not self._openai_client:
            raise RuntimeError("OpenAI client is not configured")

        response = self._openai_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        text = ""
        if response.choices:
            message = response.choices[0].message
            text = message.content if message else ""

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return _LLMResponse(text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    def _log_llm_usage(self, session_id: int, response: Any, response_text: str):
        """Record token usage (as reported by provider) and partial response to DB."""
        if not session_id:
            return
        try:
            usage = None
            input_tokens = None
            output_tokens = None

            # Prefer provider-reported usage as the single source of truth
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                input_tokens = getattr(usage, "prompt_token_count", None)
                output_tokens = getattr(usage, "candidates_token_count", None)

            if (input_tokens is None or output_tokens is None) and hasattr(response, "usage"):
                usage = response.usage
                input_tokens = input_tokens if input_tokens is not None else getattr(usage, "prompt_tokens", None)
                output_tokens = output_tokens if output_tokens is not None else getattr(usage, "completion_tokens", None)

            if input_tokens is None or output_tokens is None:
                # Provider did not supply usage; skip cost accrual rather than guessing
                try:
                    telemetry_logger.info(
                        json.dumps(
                            {
                                "type": "llm_usage_missing",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "session_id": session_id,
                                "note": "provider returned no token usage; cost not accrued",
                            }
                        )
                    )
                except Exception:
                    logger.debug("Failed to log missing usage telemetry")
                return

            cost = self.cost_tracker.calculate_llm_cost(input_tokens, output_tokens)
            self.db.log_llm_call(session_id, input_tokens, output_tokens, cost, response_text[:500])
        except Exception as e:
            logger.warning(f"Error tracking LLM usage: {e}")

    def _parse_tool_requests(self, payload: str) -> List[ToolRequest]:
        """Parse tool_requests JSON into validated ToolRequest objects."""
        if not payload:
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        tool_items = parsed.get("tool_requests") if isinstance(parsed, dict) else None
        if not tool_items:
            return []
        requests: List[ToolRequest] = []
        for item in tool_items:
            try:
                tool = item.get("tool") if isinstance(item, dict) else None
                target_model = None
                if tool:
                    try:
                        # Ensure we look up using the Enum if possible
                        target_model = TOOL_PARAM_MODELS.get(ToolName(tool))
                    except ValueError:
                        # Fallback or invalid tool string
                        target_model = TOOL_PARAM_MODELS.get(tool)
                
                params = item.get("params") if isinstance(item, dict) else None
                
                if target_model and params is not None:
                    if isinstance(params, BaseModel):
                        params = params.model_dump()
                    if isinstance(params, dict):
                        # 1. Handle timeframes being a single string instead of a list
                        if "timeframes" in params and isinstance(params["timeframes"], str):
                            # Split by comma if present, otherwise treat as single item list
                            tf_str = params["timeframes"]
                            params["timeframes"] = [t.strip() for t in tf_str.split(",")] if "," in tf_str else [tf_str]

                        allowed = set(target_model.model_fields.keys())
                        # Drop fields that do not belong to the target model
                        params = {k: v for k, v in params.items() if k in allowed}
                        
                        # Coerce numeric params that may be supplied as strings or dicts by the LLM.
                        for numeric_field in ("limit", "depth"):
                            if numeric_field in params:
                                value = params[numeric_field]
                                if isinstance(value, dict):
                                    # Pick the largest numeric value if a timeframe keyed dict slipped through.
                                    numeric_values = [
                                        v for v in value.values() if isinstance(v, (int, float))
                                    ]
                                    params[numeric_field] = max(numeric_values) if numeric_values else None
                                elif isinstance(value, str):
                                    # Handle "100" or "100.0"
                                    try:
                                        params[numeric_field] = int(float(value))
                                    except ValueError:
                                        params[numeric_field] = None
                        
                        # Remove any None coercions to let defaults apply.
                        params = {k: v for k, v in params.items() if v is not None}
                        item = dict(item)
                        item["params"] = params
                
                requests.append(ToolRequest.model_validate(item))
            except ValidationError as exc:
                # Log the specific item that failed for better debugging
                logger.warning(f"Tool request validation failed for item: {json.dumps(item)} - Error: {exc}")
                # Don't spam the main bot log, but keep it in debug or a specific logger if needed
                # bot_actions_logger.info("⚠️ LLM tool request rejected (see console.log for details)")
        return requests

    def _enforce_prompt_budget(self, prompt: str, budget: int = LLM_DECISION_BYTE_BUDGET) -> str:
        """
        Trim prompt sections to respect a byte budget; removes memory/context blocks first,
        then hard-truncates as a last resort.
        """
        if budget <= 0:
            return prompt
        def _bytes(s: str) -> int:
            try:
                return len(s.encode("utf-8"))
            except Exception:
                return len(s)

        if _bytes(prompt) <= budget:
            return prompt

        trimmed = prompt

        # Strip memory block if present
        trimmed_mem = re.sub(
            r"MEMORY \(recent plans/decisions\):\n.*?(?:\n{2,}|\Z)",
            "MEMORY: trimmed for budget\n\n",
            trimmed,
            flags=re.DOTALL,
        )
        if _bytes(trimmed_mem) <= budget:
            return trimmed_mem
        trimmed = trimmed_mem

        # Strip large context block if present
        trimmed_ctx = re.sub(
            r"CONTEXT:\n.*?(RULES:|\Z)",
            "CONTEXT: trimmed for budget\n\\1",
            trimmed,
            flags=re.DOTALL,
        )
        if _bytes(trimmed_ctx) <= budget:
            return trimmed_ctx
        trimmed = trimmed_ctx

        # Final hard clamp
        encoded = trimmed.encode("utf-8")
        if len(encoded) <= budget:
            return trimmed
        head = encoded[: budget - 100]  # leave room for marker
        marker = b"\n[TRIMMED]"
        if len(marker) > 100:
            marker = b"[TRIMMED]"
        clamped = head + marker
        try:
            return clamped.decode("utf-8", errors="ignore")
        except Exception:
            return trimmed[: max(0, budget // 2)]

    async def _get_llm_decision(self, session_id, market_data, current_equity, prompt_context=None, trading_context=None, open_orders=None, headroom: float = 0.0, pending_buy_exposure: float = 0.0, can_trade: bool = True, spacing_flag: str = "", plan_counts: dict = None):
        """Asks the configured LLM for a trading decision and logs full prompt/response; returns (decision_json, trace_id)."""
        if not self._llm_ready:
            return None, None

        open_orders = open_orders or []

        equity_now = current_equity if current_equity is not None else 0.0

        available_symbols = list(market_data.keys())
        if not available_symbols:
            return None
        
        # Build market data summary
        market_summary = ""
        for symbol, data in market_data.items():
            if data:
                spread = data.get('spread_pct')
                ob_imb = data.get('ob_imbalance')
                vol = data.get('volume')
                spread_str = f", Spread {spread:.3f}%" if spread is not None else ""
                ob_str = f", OB Imb {ob_imb:+.2f}" if ob_imb is not None else ""
                vol_str = f", Vol {vol}" if vol is not None else ""
                market_summary += (
                    f"\n  - {symbol}: Price ${data.get('price', 'N/A')}, "
                    f"Bid ${data.get('bid', 'N/A')}, Ask ${data.get('ask', 'N/A')}"
                    f"{spread_str}{ob_str}{vol_str}"
                )

        # When tools are available, prefer them over inline snapshots
        if self.tool_coordinator:
            market_summary = "Tool responses will be the source of truth; inline market snapshot omitted to save tokens."
        
        symbol = available_symbols[0]
        context_summary = ""
        regime_flags = {}

        memory_block = ""
        if symbol and trading_context:
            try:
                context_summary = trading_context.get_context_summary(symbol, open_orders=open_orders)
                recent_bars = {
                    tf: self.db.get_recent_ohlcv(session_id, symbol, tf, limit=50)
                    for tf in ['1m', '5m', '1h', '1d']
                    if hasattr(self.db, "get_recent_ohlcv")
                }
                regime_flags = self._compute_regime_flags(session_id, symbol, market_data.get(symbol, {}), recent_bars)
                if hasattr(trading_context, "get_memory_snapshot"):
                    mem = trading_context.get_memory_snapshot()
                    if mem:
                        memory_block = f"MEMORY (recent plans/decisions):\n{mem}\n\n"
            except Exception as e:
                logger.warning(f"Error getting context/regime: {e}")
        
        is_crypto = ACTIVE_EXCHANGE == 'GEMINI' and any('/' in symbol for symbol in available_symbols)

        prompt_context_block = ""
        rules_block = ""
        if prompt_context:
            prompt_context_block = f"CONTEXT:\n{prompt_context}\n"
            rules_block = (
                "RULES:\n"
                "- If cooldown is active and priority signal is NOT allowed, you MUST return HOLD.\n"
                "- If priority signal allowed = true, you may trade despite cooldown but size must be reduced and within caps.\n"
                "- Always obey order cap and exposure cap; quantities must fit caps.\n"
                f"- Ensure trade value is at least ${MIN_TRADE_SIZE:.2f}.\n"
                "- If fee regime is high, avoid churn: prefer HOLD or maker-first trades.\n"
                "- Always use symbols from the Available Symbols list.\n"
                "- Factor existing open orders into sizing/direction to avoid over-allocation or duplicate legs.\n"
                "- You may return action=CANCEL with an order_id from the open orders list to pull a stale or unsafe order.\n"
                "- If plan cap for a symbol is reached, do NOT place new BUY/SELL. Use UPDATE_PLAN (with plan_id/stop/target/size_factor) or CANCEL/PARTIAL_CLOSE existing plans instead.\n"
            )

        mode_note = ""
        if TRADING_MODE == 'PAPER':
            mode_note = (
                "NOTE: You are running in SANDBOX/PAPER mode. The 'Portfolio Value' shown above represents "
                "the Profit/Loss (PnL) relative to the starting balance, NOT the total account value. It may be negative. "
                "This is expected. You still have sufficient capital to trade. Do NOT stop trading because of a negative Portfolio Value.\n"
            )

        rejection_note = ""
        if self.last_rejection_reason:
            rejection_note = (
                "IMPORTANT: Your previous order was REJECTED by the Risk Manager for the following reason:\n"
                f"'{self.last_rejection_reason}'\n"
                "You MUST adjust your strategy to avoid this rejection (e.g., lower quantity, check limits).\n"
            )
            self.last_rejection_reason = None

        response_instructions_planner = (
            "Return ONLY a JSON object with this shape:\n"
            "{\n"
            '  "tool_requests": [\n'
            '    {"id":"req1","tool":"get_market_data","params":{"symbol":"<symbol from available symbols>","timeframes":<subset of timeframes>,"limit":<bars up to '
            f'{TOOL_MAX_BARS}>}},\n'
            '    {"id":"req2","tool":"get_order_book","params":{"symbol":"<symbol>","depth":<levels up to '
            f'{TOOL_MAX_DEPTH}>}},\n'
            '    {"id":"req3","tool":"get_recent_trades","params":{"symbol":"<symbol>","limit":<trades up to '
            f'{TOOL_MAX_TRADES}>}}\n'
            "  ]\n"
            "}\n"
            "Use only allowed tools (get_market_data, get_order_book, get_recent_trades). "
            f"Stay within caps: max_bars={TOOL_MAX_BARS}, max_depth={TOOL_MAX_DEPTH}, max_trades={TOOL_MAX_TRADES}, max_json_bytes={TOOL_MAX_JSON_BYTES}. "
            "If no extra data is needed, return {\"tool_requests\":[]}. Do not propose actions here."
        )

        response_instructions_decision = (
            "Tool responses (if any) are provided below. Use them as the source of truth. "
            "Return ONLY a JSON object with the following format:\n"
            "{\n"
            '    "action": "BUY" | "SELL" | "HOLD" | "CANCEL" | "UPDATE_PLAN" | "PARTIAL_CLOSE" | "CLOSE_POSITION" | "PAUSE_TRADING",\n'
            '    "symbol": "<symbol from available symbols>",\n'
            '    "quantity": <number>,\n'
            '    "reason": "<short explanation>",\n'
            '    "order_id": "<order id to cancel when action=CANCEL>",\n'
            '    "stop_price": <number|null>,\n'
            '    "target_price": <number|null>,\n'
            '    "plan_id": <number|null>,          // required when action=UPDATE_PLAN or PARTIAL_CLOSE\n'
            '    "size_factor": <number|null>,      // optional for UPDATE_PLAN (e.g., 0.5 to halve size)\n'
            '    "close_fraction": <number|null>,   // optional for PARTIAL_CLOSE (0-1 fraction to close)\n'
            '    "duration_minutes": <number|null>  // optional for PAUSE_TRADING\n'
            "}\n"
            "Do not request tools in this step."
        )

        base_prompt = self._build_prompt(
            asset_class="crypto" if is_crypto else "stock",
            equity_line=f"${equity_now:,.2f} USD" if is_crypto else f"${equity_now:,.2f} AUD",
            market_summary=market_summary or "  - No market data available",
            max_order_value=f"${MAX_ORDER_VALUE:.2f} USD" if is_crypto else f"${MAX_ORDER_VALUE:.2f} AUD",
            min_trade_size=f"${MIN_TRADE_SIZE:.2f} USD" if is_crypto else f"${MIN_TRADE_SIZE:.2f} AUD",
            max_daily_loss=f"{MAX_DAILY_LOSS_PERCENT}% of portfolio" if is_crypto else f"${MAX_DAILY_LOSS:.2f} AUD",
            available_symbols=", ".join(available_symbols),
            quantity_guidance=(
                "For crypto trading, you can use FRACTIONAL quantities (e.g., 0.001 BTC). "
                f"Calculate the appropriate fractional amount to stay within the ${MAX_ORDER_VALUE:.2f} max order value."
                if is_crypto
                else "For stock trading, quantities must be WHOLE NUMBERS (integers)."
            ),
            context_block=f"{context_summary}\n\n" if context_summary else "",
            regime_flags=(", ".join(f"{k}={v}" for k, v in regime_flags.items()) if regime_flags else "none"),
            prompt_context_block=prompt_context_block,
            rules_block=f"{rules_block}\n" if rules_block else "",
            mode_note=mode_note,
            rejection_note=rejection_note,
            exposure_headroom=f"${headroom:,.2f}",
            pending_exposure_budget=f"${pending_buy_exposure:,.2f}",
            max_open_orders_per_symbol=MAX_POSITIONS,
            response_instructions=response_instructions_decision,
            memory_block=memory_block,
        )

        trace_id = None
        tool_requests: List[ToolRequest] = []
        tool_responses: List[ToolResponse] = []

        # Planner turn to request tools when coordinator is available
        if self.tool_coordinator:
            planner_prompt = self._build_prompt(
                asset_class="crypto" if is_crypto else "stock",
                equity_line=f"${equity_now:,.2f} USD" if is_crypto else f"${equity_now:,.2f} AUD",
                market_summary=market_summary or "  - No market data available",
                max_order_value=f"${MAX_ORDER_VALUE:.2f} USD" if is_crypto else f"${MAX_ORDER_VALUE:.2f} AUD",
                min_trade_size=f"${MIN_TRADE_SIZE:.2f} USD" if is_crypto else f"${MIN_TRADE_SIZE:.2f} AUD",
                max_daily_loss=f"{MAX_DAILY_LOSS_PERCENT}% of portfolio" if is_crypto else f"${MAX_DAILY_LOSS:.2f} AUD",
                available_symbols=", ".join(available_symbols),
                quantity_guidance=(
                    "For crypto trading, you can use FRACTIONAL quantities (e.g., 0.001 BTC). "
                    f"Calculate the appropriate fractional amount to stay within the ${MAX_ORDER_VALUE:.2f} max order value."
                    if is_crypto
                    else "For stock trading, quantities must be WHOLE NUMBERS (integers)."
                ),
                context_block=f"{context_summary}\n\n" if context_summary else "",
                regime_flags=(", ".join(f"{k}={v}" for k, v in regime_flags.items()) if regime_flags else "none"),
                prompt_context_block="",
                rules_block="",
                mode_note="",
                rejection_note="",
                exposure_headroom=f"${headroom:,.2f}",
                pending_exposure_budget=f"${pending_buy_exposure:,.2f}",
                max_open_orders_per_symbol=MAX_POSITIONS,
                response_instructions=response_instructions_planner,
                memory_block=memory_block,
            )
            try:
                telemetry_logger.info(
                    json.dumps(
                        {
                            "type": "llm_prompt",
                            "role": "planner",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "session_id": session_id,
                            "prompt": planner_prompt,
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to log planner prompt: {e}")

            try:
                planner_response = await self._invoke_llm(planner_prompt)
                self._log_llm_usage(session_id, planner_response, planner_response.text)
                planner_payload = self._extract_json_payload(planner_response.text)
                tool_requests = self._parse_tool_requests(planner_payload)
                if tool_requests:
                    summary = "; ".join(
                        f"{tr.tool.value}({getattr(tr.params, 'symbol', '')})"
                        for tr in tool_requests
                    )
                    bot_actions_logger.info(f"🛠️ LLM requested tools: {summary}")
                telemetry_logger.info(
                    json.dumps(
                        {
                            "type": "llm_tool_requests",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "session_id": session_id,
                            "tool_requests": [tr.model_dump() for tr in tool_requests],
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Planner turn failed: {e}")

            if tool_requests:
                try:
                    tool_responses = await self.tool_coordinator.handle_requests(tool_requests)
                    telemetry_logger.info(
                        json.dumps(
                            {
                                "type": "llm_tool_responses",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "session_id": session_id,
                                "tool_responses": [tr.model_dump() for tr in tool_responses],
                            }
                        )
                    )
                except Exception as e:
                    logger.warning(f"Tool fetch failed: {e}")
                    bot_actions_logger.info("⚠️ Tool fetch failed (check console.log for details)")

        decision_prompt = base_prompt
        if tool_responses:
            decision_prompt += "\n\nTOOL RESPONSES (JSON):\n"
            decision_prompt += json.dumps([r.model_dump() for r in tool_responses], default=str) + "\n"
            decision_prompt += "TOOL RESPONSES ABOVE ARE THE SOURCE OF TRUTH. If any inline snapshot conflicts, prefer tool_responses.\n"

        # Explicitly surface data freshness and microstructure detail when available
        freshness_notes = []
        for sym, data in market_data.items():
            if not data:
                continue
            parts = []
            if data.get("_latency_ms") is not None:
                parts.append(f"latency_ms={data.get('_latency_ms'):.0f}")
            if data.get("_fetched_monotonic") is not None:
                parts.append(f"fetched_at={getattr(data, 'fetched_at', '')}")
            if data.get("spread_pct") is not None:
                parts.append(f"spread_pct={data.get('spread_pct'):.3f}")
            if data.get("bid_size") and data.get("ask_size") and data.get("bid") and data.get("ask"):
                top_notional = min(data.get("bid") * data.get("bid_size"), data.get("ask") * data.get("ask_size"))
                parts.append(f"top_notional=${top_notional:,.2f}")
            if parts:
                freshness_notes.append(f"{sym}: " + ", ".join(parts))
        if freshness_notes:
            decision_prompt += "\nDATA_QUALITY:\n- " + "\n- ".join(freshness_notes) + "\n"

        # Flag plan cap/cooldown status
        cooldown_note = ""
        if not can_trade and priority_flag != "true":
            cooldown_note = f"cooldown active ({spacing_flag})"
        plan_counts = plan_counts or {}
        plan_cap_note = ""
        if any(count >= PLAN_MAX_PER_SYMBOL for count in plan_counts.values()):
            plan_cap_note = "plan cap reached on at least one symbol"
        if cooldown_note or plan_cap_note:
            decision_prompt += "\nCONSTRAINTS:\n"
            if cooldown_note:
                decision_prompt += f"- {cooldown_note}\n"
            if plan_cap_note:
                decision_prompt += f"- {plan_cap_note}\n"

        decision_prompt = self._enforce_prompt_budget(decision_prompt, budget=LLM_DECISION_BYTE_BUDGET)

        # Log full decision prompt to telemetry
        try:
            prompt_log = {
                "type": "llm_prompt",
                "role": "decision",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "prompt": decision_prompt
            }
            telemetry_logger.info(json.dumps(prompt_log, default=str))
        except Exception as e:
            logger.warning(f"Failed to log LLM prompt to telemetry: {e}")

        try:
            response = await self._invoke_llm(decision_prompt)
            self._log_llm_usage(session_id, response, response.text)

            text = self._extract_json_payload(response.text)

            try:
                market_context = {
                    "market_data": market_data,
                    "prompt_context": prompt_context,
                    "context_summary": context_summary,
                    "open_orders": open_orders,
                    "tool_requests": [tr.model_dump() for tr in tool_requests],
                    "tool_responses": [tr.model_dump() for tr in tool_responses],
                }
                trace_id = self.db.log_llm_trace(
                    session_id,
                    decision_prompt,
                    response.text,
                    decision_json=text,
                    market_context=market_context,
                )
            except Exception as e:
                logger.warning(f"Could not log LLM trace: {e}")

            return text, trace_id
        except Exception as e:
            logger.error(f"LLM Error: {e}\n{traceback.format_exc()}")
            return None, None

    def on_trade_executed(self, timestamp):
        """Callback to update internal state after a successful trade."""
        self.last_trade_ts = timestamp

    def on_trade_rejected(self, reason):
        """Callback to update internal state after a rejected trade."""
        self.last_rejection_reason = reason

    def _clamp_quantity(self, quantity: float, price: float, headroom: float) -> float:
        """Clamp quantities to fit within exposure and order caps."""
        if price <= 0 or quantity <= 0:
            return 0.0
        max_notional = min(MAX_ORDER_VALUE, headroom if headroom is not None else MAX_ORDER_VALUE)
        max_qty = max_notional / price if price else 0.0
        return min(quantity, max_qty)
