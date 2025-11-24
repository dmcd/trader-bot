import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import json
import math
from statistics import pstdev
import re
from pathlib import Path
import google.generativeai as genai
import jsonschema
from config import (
    GEMINI_API_KEY,
    MAX_ORDER_VALUE,
    MAX_DAILY_LOSS_PERCENT,
    MAX_DAILY_LOSS,
    MAX_TOTAL_EXPOSURE,
    MIN_TRADE_INTERVAL_SECONDS,
    FEE_RATIO_COOLDOWN,
    PRIORITY_MOVE_PCT,
    PRIORITY_LOOKBACK_MIN,
    BREAK_GLASS_COOLDOWN_MIN,
    LOOP_INTERVAL_SECONDS,
    ACTIVE_EXCHANGE,
    TRADING_MODE,
    MIN_TRADE_SIZE,
    MAX_POSITIONS,
)

logger = logging.getLogger(__name__)

class StrategySignal:
    def __init__(self, action: str, symbol: str, quantity: float, reason: str, order_id=None, trace_id: int = None):
        self.action = action.upper()
        self.symbol = symbol
        self.quantity = quantity
        self.reason = reason
        self.order_id = order_id
        self.trace_id = trace_id

    def __str__(self):
        return f"{self.action} {self.quantity} {self.symbol} ({self.reason})"

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
    def __init__(self, db, technical_analysis, cost_tracker, open_orders_provider=None, ohlcv_provider=None):
        super().__init__(db, technical_analysis, cost_tracker)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        else:
            logger.warning("GEMINI_API_KEY not found. LLMStrategy will not work.")
        
        self.last_trade_ts = None
        self._last_break_glass = 0.0
        self.last_rejection_reason = None
        # Optional async callable that fetches live open orders from the active exchange
        self.open_orders_provider = open_orders_provider
        # Optional async callable that fetches OHLCV data
        self.ohlcv_provider = ohlcv_provider
        self.prompt_template = self._load_prompt_template()
        self._decision_schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD", "CANCEL"]},
                "symbol": {"type": "string"},
                "quantity": {"type": "number", "minimum": 0},
                "reason": {"type": "string"},
                "order_id": {"type": ["string", "number", "null"]},
                "stop_price": {"type": ["number", "null"], "minimum": 0},
                "target_price": {"type": ["number", "null"], "minimum": 0}
            },
            "required": ["action", "symbol", "quantity", "reason"],
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
        else:
            now_ts = asyncio.get_event_loop().time()
            
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

        prompt_context = (
            f"- Cooldown: {spacing_flag}\n"
            f"- Priority signal allowed: {priority_flag}\n"
            f"- Fee regime: {fee_ratio_flag}\n"
            f"- Last trade age: {last_trade_age_str}\n"
            f"- Equity: ${equity_now:,.2f}\n"
            f"- Exposure: ${exposure_now:,.2f} of ${exposure_cap:,.2f} (room ${headroom:,.2f})\n"
            f"- Pending BUY exposure: ${pending_buy_exposure:,.2f}\n"
            f"- Max open orders per symbol: {MAX_POSITIONS}\n"
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
                quantity = decision.get('quantity', 0)
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
                    
                    signal = StrategySignal(action, symbol, quantity, reason, trace_id=trace_id)
                    # Carry stop/target forward to runner via attributes
                    signal.stop_price = stop_price
                    signal.target_price = target_price
                    return signal
                elif action == 'CANCEL' and order_id:
                    return StrategySignal('CANCEL', symbol or '', 0, reason or 'Cancel open order', order_id=order_id, trace_id=trace_id)
                elif action == 'HOLD':
                    return StrategySignal('HOLD', symbol, 0, reason, trace_id=trace_id)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {decision_json}")
        
        return None


    def _load_prompt_template(self) -> str:
        """Load template from adjacent file to make manual edits easy."""
        if hasattr(LLMStrategy, "_prompt_template_cache"):
            return LLMStrategy._prompt_template_cache

        template_path = Path(__file__).with_name("llm_prompt_template.txt")
        template_text = template_path.read_text()
        LLMStrategy._prompt_template_cache = template_text
        return template_text

    def _build_prompt(self, **kwargs) -> str:
        """Render prompt with a forgiving formatter so optional fields can be empty."""

        class _SafeDict(dict):
            def __missing__(self, key):
                return ""

        return self.prompt_template.format_map(_SafeDict(**kwargs))

    async def _get_llm_decision(self, session_id, market_data, current_equity, prompt_context=None, trading_context=None, open_orders=None, headroom: float = 0.0, pending_buy_exposure: float = 0.0):
        """Asks Gemini for a trading decision and logs full prompt/response; returns (decision_json, trace_id)."""
        if not GEMINI_API_KEY:
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
        
        symbol = available_symbols[0]
        context_summary = ""
        indicator_summary = ""
        
        timeframe_summary = ""
        regime_flags = {}
        try:
            timeframe_summary = self._build_timeframe_summary(session_id, symbol)
        except Exception as e:
            logger.debug(f"Timeframe summary error: {e}")

        if symbol and trading_context:
            try:
                context_summary = trading_context.get_context_summary(symbol, open_orders=open_orders)
                recent_data = self.db.get_recent_market_data(session_id, symbol, limit=50)
                if recent_data and len(recent_data) >= 20:
                    indicators = self.ta.calculate_indicators(recent_data)
                    if indicators:
                        current_price = market_data[symbol]['price']
                        indicator_summary = self.ta.format_indicators_for_llm(indicators, current_price)
                recent_bars = {
                    tf: self.db.get_recent_ohlcv(session_id, symbol, tf, limit=50)
                    for tf in ['1m', '5m', '1h', '1d']
                    if hasattr(self.db, "get_recent_ohlcv")
                }
                regime_flags = self._compute_regime_flags(session_id, symbol, market_data.get(symbol, {}), recent_bars)
            except Exception as e:
                logger.warning(f"Error getting context/indicators: {e}")
        
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

        prompt = self._build_prompt(
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
            indicator_block=f"{indicator_summary}\n\n" if indicator_summary else "",
            multi_tf_block=f"{timeframe_summary}\n\n" if timeframe_summary else "",
            regime_flags=(", ".join(f"{k}={v}" for k, v in regime_flags.items()) if regime_flags else "none"),
            prompt_context_block=prompt_context_block,
            rules_block=f"{rules_block}\n" if rules_block else "",
            mode_note=mode_note,
            rejection_note=rejection_note,
            exposure_headroom=f"${headroom:,.2f}",
            pending_exposure_budget=f"${pending_buy_exposure:,.2f}",
            max_open_orders_per_symbol=MAX_POSITIONS,
        )
            
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.model.generate_content, prompt),
                timeout=30
            )
            
            if hasattr(response, 'usage_metadata') and session_id:
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                    cost = self.cost_tracker.calculate_llm_cost(input_tokens, output_tokens)
                    self.db.log_llm_call(session_id, input_tokens, output_tokens, cost, response.text[:500])
                except Exception as e:
                    logger.warning(f"Error tracking LLM usage: {e}")
            
            text = self._extract_json_payload(response.text)

            trace_id = None
            try:
                market_context = {
                    "market_data": market_data,
                    "prompt_context": prompt_context,
                    "context_summary": context_summary,
                    "indicator_summary": indicator_summary,
                    "open_orders": open_orders,
                }
                trace_id = self.db.log_llm_trace(
                    session_id,
                    prompt,
                    response.text,
                    decision_json=text,
                    market_context=market_context,
                )
            except Exception as e:
                logger.warning(f"Could not log LLM trace: {e}")

            return text, trace_id
        except Exception as e:
            logger.error(f"LLM Error: {e}")
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
