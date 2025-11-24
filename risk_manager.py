import logging
from config import (
    MAX_DAILY_LOSS,
    MAX_DAILY_LOSS_PERCENT,
    MAX_ORDER_VALUE,
    MAX_POSITIONS,
    MAX_TOTAL_EXPOSURE,
    ORDER_VALUE_BUFFER,
    MIN_TRADE_SIZE,
)

logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""

class RiskManager:
    def __init__(self, bot=None):
        self.bot = bot # Optional, mostly for position checks if needed
        self.daily_loss = 0.0
        self.start_of_day_equity = None
        self.positions = {} # Symbol -> Quantity

    def seed_start_of_day(self, start_equity: float):
        """Persist start-of-day equity so restarts keep loss limits consistent."""
        if start_equity is not None:
            self.start_of_day_equity = start_equity

    def update_equity(self, current_equity: float):
        """Track drawdown off start-of-day equity (keeps loss limits consistent)."""
        if self.start_of_day_equity is None:
            self.start_of_day_equity = current_equity

        drawdown = (self.start_of_day_equity or 0) - (current_equity or 0)
        self.daily_loss = max(0.0, drawdown)

    # Backward compatibility
    update_pnl = update_equity

    def update_positions(self, positions: dict):
        """Updates the current positions for exposure calculation.
        positions: dict of {symbol: {'quantity': float, 'current_price': float}}
        """
        self.positions = positions

    def apply_order_value_buffer(self, quantity: float, price: float):
        """Trim quantity so notional stays under the order cap minus buffer."""
        if price <= 0 or quantity <= 0:
            return quantity, 0.0

        order_value = quantity * price
        if order_value <= MAX_ORDER_VALUE:
            return quantity, 0.0

        capped_value = max(0.0, MAX_ORDER_VALUE - ORDER_VALUE_BUFFER)
        capped_qty = capped_value / price if price else 0.0
        capped_qty = max(0.0, capped_qty)
        overage = order_value - MAX_ORDER_VALUE
        return capped_qty, overage

    def check_trade_allowed(self, symbol, action, quantity, price) -> RiskCheckResult:
        """Checks if a trade is allowed based on risk limits."""
        if price <= 0 or quantity <= 0:
            return RiskCheckResult(False, "Invalid price or quantity")
        
        # 1. Check Max Order Value
        order_value = quantity * price
        if order_value > MAX_ORDER_VALUE:
            msg = f"Order value ${order_value:.2f} exceeds limit of ${MAX_ORDER_VALUE:.2f}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)

        # 1.5 Check Min Trade Size
        if order_value < MIN_TRADE_SIZE:
            msg = f"Order value ${order_value:.2f} is below minimum of ${MIN_TRADE_SIZE:.2f}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)

        # 2. Check Daily Loss (both absolute and percentage)
        # We need to fetch current equity to update PnL first, but assuming it's updated periodically
        if self.start_of_day_equity and self.start_of_day_equity > 0:
            loss_percent = (self.daily_loss / self.start_of_day_equity) * 100
            if loss_percent > MAX_DAILY_LOSS_PERCENT:
                msg = f"Daily loss {loss_percent:.2f}% exceeds limit of {MAX_DAILY_LOSS_PERCENT}%"
                logger.warning(f"Risk Reject: {msg}")
                return RiskCheckResult(False, msg)
        
        # Also check absolute loss for small accounts
        if self.daily_loss > MAX_DAILY_LOSS:
            msg = f"Daily loss ${self.daily_loss:.2f} exceeds limit of ${MAX_DAILY_LOSS:.2f}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)

        # 3. Check Max Positions (only for BUY orders)
        if action == 'BUY':
            # Check Total Exposure
            price_overrides = {symbol: price} if price else None
            current_exposure = self.get_total_exposure(price_overrides=price_overrides)
            new_exposure = current_exposure + order_value
            
            if new_exposure > MAX_TOTAL_EXPOSURE:
                msg = f"Total exposure ${new_exposure:.2f} would exceed limit of ${MAX_TOTAL_EXPOSURE:.2f}"
                logger.warning(f"Risk Reject: {msg}")
                return RiskCheckResult(False, msg)
                
            # Safe Buffer Warning (90%)
            if new_exposure > (MAX_TOTAL_EXPOSURE * 0.9):
                logger.warning(f"Risk Warning: Total exposure ${new_exposure:.2f} is close to limit of ${MAX_TOTAL_EXPOSURE:.2f}")

            # This requires the bot to have a way to count positions
            # For now, we'll skip or implement a simple check if we can access positions
            pass

        return RiskCheckResult(True, "Trade allowed")

    def get_total_exposure(self, price_overrides: dict = None) -> float:
        """Return total notional exposure using marked prices."""
        exposure = 0.0
        for sym, data in self.positions.items():
            qty = data.get('quantity', 0) or 0.0
            curr_price = 0.0

            if price_overrides and sym in price_overrides and price_overrides[sym]:
                curr_price = price_overrides[sym]
            else:
                curr_price = data.get('current_price', 0) or 0.0

            exposure += qty * curr_price

        return exposure
