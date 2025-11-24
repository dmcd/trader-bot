import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import json
from pathlib import Path
import google.generativeai as genai
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
    MIN_TRADE_SIZE
)

logger = logging.getLogger(__name__)

class StrategySignal:
    def __init__(self, action: str, symbol: str, quantity: float, reason: str, order_id=None):
        self.action = action.upper()
        self.symbol = symbol
        self.quantity = quantity
        self.reason = reason
        self.order_id = order_id

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
    def __init__(self, db, technical_analysis, cost_tracker, open_orders_provider=None):
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
        self.prompt_template = self._load_prompt_template()

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

    def _fees_too_high(self, session_id: int):
        """
        Check fee ratio vs gross PnL; returns True to pause when fees dominate.
        """
        try:
            stats = self.db.get_session_stats(session_id)
            gross = stats.get('gross_pnl', 0) or 0
            fees = stats.get('total_fees', 0) or 0
            if gross == 0:
                return False
            ratio = (fees / abs(gross)) * 100
            return ratio >= FEE_RATIO_COOLDOWN
        except Exception as e:
            logger.warning(f"Fee ratio check failed: {e}")
            return False

    async def generate_signal(self, session_id: int, market_data: Dict[str, Any], current_equity: float, current_exposure: float, context: Any = None) -> Optional[StrategySignal]:
        if not market_data:
            return None

        # Assuming single symbol focus for now as per original code
        symbol = list(market_data.keys())[0]
        data = market_data[symbol]

        # 1. Fee Check
        if self._fees_too_high(session_id):
            logger.info("Skipping trading due to high fee ratio")
            return None

        # 2. Chop Check
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
        fee_ratio_flag = "high" if self._fees_too_high(session_id) else "normal"
        priority_flag = "true" if priority and allow_break_glass else "false"
        spacing_flag = "clear" if can_trade else f"cooldown {MIN_TRADE_INTERVAL_SECONDS - (now_ts - self.last_trade_ts):.0f}s"
        
        order_cap_value = MAX_ORDER_VALUE
        exposure_cap = MAX_TOTAL_EXPOSURE
        exposure_now = current_exposure if current_exposure is not None else 0.0
        equity_now = current_equity if current_equity is not None else 0.0
        headroom = max(0.0, exposure_cap - exposure_now)
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
            f"- Order cap: ${order_cap_value:.2f}\n"
            f"- Min trade size: ${MIN_TRADE_SIZE:.2f}\n"
            f"- Open orders: {open_order_count} ({open_orders_summary})"
        )

        decision_json = await self._get_llm_decision(
            session_id,
            market_data,
            current_equity,
            prompt_context,
            context,
            open_orders=open_orders,
        )
        
        if decision_json:
            try:
                decision = json.loads(decision_json)
                action = decision.get('action')
                quantity = decision.get('quantity', 0)
                reason = decision.get('reason')
                order_id = decision.get('order_id')

                if action in ['BUY', 'SELL'] and quantity > 0:
                    # Update state if we are signaling a trade
                    if allow_break_glass and not can_trade:
                         self._last_break_glass = now_ts

                    price = data.get('price') if data else None

                    if action == 'BUY':
                        max_order_value = min(order_cap_value, headroom)
                        if not price or max_order_value <= 0:
                            return StrategySignal('HOLD', symbol, 0, 'No exposure headroom')
                        max_qty = max_order_value / price
                        if quantity > max_qty:
                            quantity = max_qty
                    
                    # We don't update last_trade_ts here, we let the runner do it upon execution success? 
                    # Actually better to update it here to prevent double signaling if execution takes time, 
                    # BUT runner might reject it. 
                    # Let's stick to the pattern: Strategy suggests, Runner executes. 
                    # Strategy state should ideally update on confirmation.
                    # For now, we'll leave state update to the caller or handle it optimistically?
                    # The original code checked `can_trade` in the loop.
                    
                    return StrategySignal(action, symbol, quantity, reason)
                elif action == 'CANCEL' and order_id:
                    return StrategySignal('CANCEL', symbol or '', 0, reason or 'Cancel open order', order_id=order_id)
                elif action == 'HOLD':
                    return StrategySignal('HOLD', symbol, 0, reason)
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

    async def _get_llm_decision(self, session_id, market_data, current_equity, prompt_context=None, trading_context=None, open_orders=None):
        """Asks Gemini for a trading decision."""
        if not GEMINI_API_KEY:
            return None

        open_orders = open_orders or []

        equity_now = current_equity if current_equity is not None else 0.0

        available_symbols = list(market_data.keys())
        if not available_symbols:
            return None
        
        # Build market data summary
        market_summary = ""
        for symbol, data in market_data.items():
            if data:
                market_summary += f"\n  - {symbol}: Price ${data.get('price', 'N/A')}, Bid ${data.get('bid', 'N/A')}, Ask ${data.get('ask', 'N/A')}"
        
        symbol = available_symbols[0]
        context_summary = ""
        indicator_summary = ""
        
        if symbol and trading_context:
            try:
                context_summary = trading_context.get_context_summary(symbol, open_orders=open_orders)
                recent_data = self.db.get_recent_market_data(session_id, symbol, limit=50)
                if recent_data and len(recent_data) >= 20:
                    indicators = self.ta.calculate_indicators(recent_data)
                    if indicators:
                        current_price = market_data[symbol]['price']
                        indicator_summary = self.ta.format_indicators_for_llm(indicators, current_price)
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
            prompt_context_block=prompt_context_block,
            rules_block=f"{rules_block}\n" if rules_block else "",
            mode_note=mode_note,
            rejection_note=rejection_note,
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
            
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3]
            elif text.startswith('```'):
                text = text[3:-3]
            return text
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return None

    def on_trade_executed(self, timestamp):
        """Callback to update internal state after a successful trade."""
        self.last_trade_ts = timestamp

    def on_trade_rejected(self, reason):
        """Callback to update internal state after a rejected trade."""
        self.last_rejection_reason = reason
