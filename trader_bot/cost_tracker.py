import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from trader_bot.config import (
    GEMINI_INPUT_COST_PER_TOKEN,
    GEMINI_MAKER_FEE,
    GEMINI_OUTPUT_COST_PER_TOKEN,
    GEMINI_TAKER_FEE,
    IB_BASE_CURRENCY,
    IB_FX_COMMISSION_PCT,
    IB_STOCK_COMMISSION_PER_SHARE,
    IB_STOCK_MIN_COMMISSION,
    OPENAI_INPUT_COST_PER_TOKEN,
    OPENAI_OUTPUT_COST_PER_TOKEN,
)
from trader_bot.symbols import infer_instrument_type, normalize_symbol

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks trading fees and LLM costs."""
    
    def __init__(self, exchange: str, llm_provider: str = "GEMINI"):
        self.exchange = exchange
        self.llm_provider = (llm_provider or "GEMINI").upper()
        
        # Fee rates per exchange
        self.fee_rates = {
            'GEMINI': {
                'maker': GEMINI_MAKER_FEE,
                'taker': GEMINI_TAKER_FEE,
            }
        }
        
        # LLM costs by provider
        self.llm_costs_by_provider = {
            'GEMINI': {
                'input_per_token': GEMINI_INPUT_COST_PER_TOKEN,
                'output_per_token': GEMINI_OUTPUT_COST_PER_TOKEN,
            },
            'OPENAI': {
                'input_per_token': OPENAI_INPUT_COST_PER_TOKEN,
                'output_per_token': OPENAI_OUTPUT_COST_PER_TOKEN,
            },
        }

        self.llm_costs = self.llm_costs_by_provider.get(
            self.llm_provider, self.llm_costs_by_provider['GEMINI']
        )
        if self.llm_provider not in self.llm_costs_by_provider:
            logger.warning(
                "Unknown LLM provider %s; defaulting cost tracking to GEMINI rates",
                self.llm_provider,
            )
    
    def calculate_trade_fee(self, symbol: str, quantity: float, price: float, 
                           action: str = 'BUY', liquidity: str = 'taker') -> float:
        """Calculate trading fee based on exchange, trade details, and maker/taker liquidity."""
        
        if self.exchange == 'GEMINI':
            # Crypto: percentage-based fee
            trade_value = quantity * price
            liq = (liquidity or 'taker').lower()
            fee_rate = self.fee_rates['GEMINI'].get('maker' if liq == 'maker' else 'taker')
            fee = trade_value * fee_rate
            logger.debug(f"Gemini fee: ${fee:.4f} ({fee_rate*100}% of ${trade_value:.2f}) [{liq}]")
            return fee
        if self.exchange == 'IB':
            symbol_norm = normalize_symbol(symbol) if symbol else symbol
            instrument = infer_instrument_type(
                *(symbol_norm.split("/", 1)),
                allowed_instrument_types=None,
                base_currency=IB_BASE_CURRENCY,
            )
            if instrument == "FX":
                trade_value = abs(quantity) * price
                fee = trade_value * IB_FX_COMMISSION_PCT
                logger.debug(
                    f"IB FX fee: ${fee:.4f} ({IB_FX_COMMISSION_PCT*100:.4f}% of ${trade_value:.2f})"
                )
                return fee
            # Default to stock/ETF per-share commission with min
            shares = abs(quantity)
            fee = max(shares * IB_STOCK_COMMISSION_PER_SHARE, IB_STOCK_MIN_COMMISSION)
            logger.debug(
                f"IB stock fee: ${fee:.4f} (per-share ${IB_STOCK_COMMISSION_PER_SHARE:.4f}, min ${IB_STOCK_MIN_COMMISSION:.2f})"
            )
            return fee

        logger.warning(f"Unknown exchange: {self.exchange}, assuming no fees")
        return 0.0
    
    def calculate_llm_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost of LLM API call."""
        input_cost = input_tokens * self.llm_costs['input_per_token']
        output_cost = output_tokens * self.llm_costs['output_per_token']
        total_cost = input_cost + output_cost
        
        logger.debug(f"LLM cost: ${total_cost:.6f} ({input_tokens} in + {output_tokens} out tokens)")
        return total_cost
    
    def calculate_net_pnl(self, gross_pnl: float, total_fees: float, total_llm_cost: float) -> float:
        """Calculate net PnL after all costs."""
        net_pnl = gross_pnl - total_fees - total_llm_cost
        
        logger.info(f"PnL Breakdown: Gross ${gross_pnl:.2f} - Fees ${total_fees:.2f} - LLM ${total_llm_cost:.2f} = Net ${net_pnl:.2f}")
        return net_pnl
    
    def get_cost_summary(self, total_fees: float, total_llm_cost: float, gross_pnl: float) -> Dict[str, Any]:
        """Get summary of all costs."""
        net_pnl = self.calculate_net_pnl(gross_pnl, total_fees, total_llm_cost)
        
        return {
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'total_llm_cost': total_llm_cost,
            'total_costs': total_fees + total_llm_cost,
            'net_pnl': net_pnl,
            'cost_ratio': (total_fees + total_llm_cost) / abs(gross_pnl) if gross_pnl != 0 else 0,
            'profitable': net_pnl > 0
        }

    def calculate_llm_burn(
        self,
        total_llm_cost: float,
        session_started: Optional[Union[str, datetime]],
        budget: float,
        now: Optional[datetime] = None,
        min_window_minutes: float = 5.0,
    ) -> Dict[str, Any]:
        """Compute burn rate vs budget using session start time.

        Returns elapsed hours, burn rate per hour, percent of budget used,
        remaining budget, and projected hours to cap (None when idle).
        """

        now_dt = now or datetime.now(timezone.utc)

        start_dt: Optional[datetime] = None
        if isinstance(session_started, datetime):
            start_dt = session_started
        elif isinstance(session_started, str) and session_started:
            try:
                start_dt = datetime.fromisoformat(session_started)
            except ValueError:
                try:
                    start_dt = datetime.fromisoformat(session_started.replace("Z", "+00:00"))
                except Exception:
                    start_dt = None

        if start_dt is None:
            start_dt = now_dt

        # Normalize naive datetimes to UTC to avoid negative deltas
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        elapsed_seconds = max((now_dt - start_dt).total_seconds(), 0)
        min_window_seconds = max(min_window_minutes * 60.0, 1.0)
        normalized_seconds = max(elapsed_seconds, min_window_seconds)
        elapsed_hours = normalized_seconds / 3600.0

        burn_rate_per_hour = (total_llm_cost or 0.0) / elapsed_hours
        pct_of_budget = (total_llm_cost / budget) if budget else 0.0
        remaining = max((budget or 0.0) - (total_llm_cost or 0.0), 0.0)
        hours_to_cap = (remaining / burn_rate_per_hour) if burn_rate_per_hour > 0 else None

        return {
            "elapsed_hours": elapsed_hours,
            "burn_rate_per_hour": burn_rate_per_hour,
            "pct_of_budget": pct_of_budget,
            "remaining_budget": remaining,
            "budget": budget,
            "hours_to_cap": hours_to_cap,
            "total_llm_cost": total_llm_cost or 0.0,
        }
