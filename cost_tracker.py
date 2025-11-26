import logging
from typing import Dict, Any
from config import (
    GEMINI_MAKER_FEE,
    GEMINI_TAKER_FEE,
    GEMINI_INPUT_COST_PER_TOKEN,
    GEMINI_OUTPUT_COST_PER_TOKEN,
)

logger = logging.getLogger(__name__)

class CostTracker:
    """Tracks trading fees and LLM costs."""
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        
        # Fee rates per exchange
        self.fee_rates = {
            'GEMINI': {
                'maker': GEMINI_MAKER_FEE,
                'taker': GEMINI_TAKER_FEE,
            }
        }
        
        # LLM costs (Gemini 2.5 Flash pricing)
        self.llm_costs = {
            'input_per_token': GEMINI_INPUT_COST_PER_TOKEN,
            'output_per_token': GEMINI_OUTPUT_COST_PER_TOKEN,
        }
    
    def calculate_trade_fee(self, symbol: str, quantity: float, price: float, 
                           action: str = 'BUY') -> float:
        """Calculate trading fee based on exchange and trade details."""
        
        if self.exchange == 'GEMINI':
            # Crypto: percentage-based fee
            trade_value = quantity * price
            # Assume taker fee (immediate execution)
            fee = trade_value * self.fee_rates['GEMINI']['taker']
            logger.debug(f"Gemini fee: ${fee:.4f} ({self.fee_rates['GEMINI']['taker']*100}% of ${trade_value:.2f})")
            return fee
        
        elif self.exchange == 'IB':
            # Stocks: per-share fee
            fee = quantity * self.fee_rates['IB']['stock_per_share']
            # Apply minimum fee
            fee = max(fee, self.fee_rates['IB']['min_fee'])
            logger.debug(f"IB fee: ${fee:.4f} (${self.fee_rates['IB']['stock_per_share']}/share * {quantity} shares)")
            return fee
        
        else:
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
