import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import json
import google.generativeai as genai
from config import (
    GEMINI_API_KEY,
    MAX_ORDER_VALUE,
    MAX_DAILY_LOSS_PERCENT,
    MAX_DAILY_LOSS,
    ORDER_SIZE_BY_TIER,
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
    def __init__(self, action: str, symbol: str, quantity: float, reason: str):
        self.action = action.upper()
        self.symbol = symbol
        self.quantity = quantity
        self.reason = reason

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
    def __init__(self, db, technical_analysis, cost_tracker, size_tier='MODERATE'):
        super().__init__(db, technical_analysis, cost_tracker)
        self.size_tier = size_tier
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
        else:
            logger.warning("GEMINI_API_KEY not found. LLMStrategy will not work.")
        
        self.last_trade_ts = None
        self._last_break_glass = 0.0
        self.last_rejection_reason = None

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
        # Use simulated time for DB lookups if in backtest
        before_ts = None
        if context and hasattr(context, 'current_iso_time'):
            before_ts = context.current_iso_time
            
        recent_data = self.db.get_recent_market_data(session_id, symbol, limit=50, before_timestamp=before_ts)
        if self._is_choppy(symbol, data, recent_data):
            logger.info("Skipping trade: market in chop")
            return None

        # 3. Timing / Priority Check
        # Use simulated time from context if available (for backtesting)
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
        last_trade_age = (now_ts - self.last_trade_ts) if self.last_trade_ts else None
        last_trade_age_str = f"{last_trade_age:.0f}s" if last_trade_age is not None else "n/a"
        fee_ratio_flag = "high" if self._fees_too_high(session_id) else "normal"
        priority_flag = "true" if priority and allow_break_glass else "false"
        spacing_flag = "clear" if can_trade else f"cooldown {MIN_TRADE_INTERVAL_SECONDS - (now_ts - self.last_trade_ts):.0f}s"
        
        order_cap_value = min(MAX_ORDER_VALUE, ORDER_SIZE_BY_TIER.get(self.size_tier, MAX_ORDER_VALUE))
        exposure_cap = MAX_TOTAL_EXPOSURE
        exposure_now = current_exposure if current_exposure is not None else 0.0
        equity_now = current_equity if current_equity is not None else 0.0
        headroom = max(0.0, exposure_cap - exposure_now)

        open_orders = self.db.get_open_orders(session_id) if session_id else []
        open_order_count = len(open_orders)
        open_order_snippets = []
        for order in open_orders[:5]:
            side = (order.get('side') or '').upper() or 'N/A'
            sym = order.get('symbol') or 'Unknown'
            qty = order.get('amount') or 0
            remaining = order.get('remaining') if order.get('remaining') is not None else qty
            price = order.get('price')
            price_str = f"${price:,.2f}" if price else "mkt"
            open_order_snippets.append(f"{side} {qty:.4f} {sym} @ {price_str} (rem {remaining:.4f})")
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

        decision_json = await self._get_llm_decision(session_id, market_data, current_equity, prompt_context, context)
        
        if decision_json:
            try:
                decision = json.loads(decision_json)
                action = decision.get('action')
                quantity = decision.get('quantity', 0)
                reason = decision.get('reason')
                
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
                elif action == 'HOLD':
                    return StrategySignal('HOLD', symbol, 0, reason)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {decision_json}")
        
        return None

    async def _get_llm_decision(self, session_id, market_data, current_equity, prompt_context=None, trading_context=None):
        """Asks Gemini for a trading decision."""
        if not GEMINI_API_KEY:
            return None

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
                context_summary = trading_context.get_context_summary(symbol)
                recent_data = self.db.get_recent_market_data(session_id, symbol, limit=50)
                if recent_data and len(recent_data) >= 20:
                    indicators = self.ta.calculate_indicators(recent_data)
                    if indicators:
                        current_price = market_data[symbol]['price']
                        indicator_summary = self.ta.format_indicators_for_llm(indicators, current_price)
            except Exception as e:
                logger.warning(f"Error getting context/indicators: {e}")
        
        is_crypto = ACTIVE_EXCHANGE == 'GEMINI' and any('/' in symbol for symbol in available_symbols)
        
        if is_crypto:
             prompt = f"""
You are an autonomous crypto trading bot. Your goal is to make small, profitable trades.

Current Status:
- Equity: ${equity_now:,.2f} USD
- Market Data:{market_summary}

Risk Constraints:
- Max Order Value: ${MAX_ORDER_VALUE:.2f} USD
- Min Trade Size: ${MIN_TRADE_SIZE:.2f} USD
- Max Daily Loss: {MAX_DAILY_LOSS_PERCENT}% of portfolio

Available Symbols: {', '.join(available_symbols)}

For crypto trading, you can use FRACTIONAL quantities (e.g., 0.001 BTC).
Calculate the appropriate fractional amount to stay within the ${MAX_ORDER_VALUE:.2f} max order value.
"""
        else:
            prompt = f"""
You are an autonomous stock trading bot. Your goal is to make small, profitable trades.

Current Status:
- Equity: ${equity_now:,.2f} AUD
- Market Data:{market_summary}

Risk Constraints:
- Max Order Value: ${MAX_ORDER_VALUE:.2f} AUD
- Min Trade Size: ${MIN_TRADE_SIZE:.2f} AUD
- Max Daily Loss: ${MAX_DAILY_LOSS:.2f} AUD

Available Symbols: {', '.join(available_symbols)}

For stock trading, quantities must be WHOLE NUMBERS (integers).
"""

        if context_summary:
            prompt += f"\n{context_summary}\n"
        if indicator_summary:
            prompt += f"\n{indicator_summary}\n"
        
        prompt += """
Decide on an action for one of the available symbols.
Return ONLY a JSON object with the following format:
{
    "action": "BUY" | "SELL" | "HOLD",
    "symbol": "<symbol from available symbols>",
    "quantity": <number>,
    "reason": "<short explanation>"
}
"""
        
        if prompt_context:
            prompt += f"\nCONTEXT:\n{prompt_context}\n"
            prompt += "\nRULES:\n"
            prompt += "- If cooldown is active and priority signal is NOT allowed, you MUST return HOLD.\n"
            prompt += "- If priority signal allowed = true, you may trade despite cooldown but size must be reduced and within caps.\n"
            prompt += "- Always obey order cap and exposure cap; quantities must fit caps.\n"
            prompt += f"- Ensure trade value is at least ${MIN_TRADE_SIZE:.2f}.\n"
            prompt += "- If fee regime is high, avoid churn: prefer HOLD or maker-first trades.\n"
            prompt += "- Always use symbols from the Available Symbols list.\n"
            prompt += "- Factor existing open orders into sizing/direction to avoid over-allocation or duplicate legs.\n"

        if TRADING_MODE == 'PAPER':
             prompt += "\nNOTE: You are running in SANDBOX/PAPER mode. The 'Portfolio Value' shown above represents the Profit/Loss (PnL) relative to the starting balance, NOT the total account value. It may be negative. This is expected. You still have sufficient capital to trade. Do NOT stop trading because of a negative Portfolio Value.\n"

        if self.last_rejection_reason:
            prompt += f"\nIMPORTANT: Your previous order was REJECTED by the Risk Manager for the following reason:\n"
            prompt += f"'{self.last_rejection_reason}'\n"
            prompt += "You MUST adjust your strategy to avoid this rejection (e.g., lower quantity, check limits).\n"
            self.last_rejection_reason = None
            
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
