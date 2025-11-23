import logging
from config import MAX_DAILY_LOSS, MAX_DAILY_LOSS_PERCENT, MAX_ORDER_VALUE, MAX_POSITIONS

logger = logging.getLogger(__name__)

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

    def update_pnl(self, current_equity):
        """Updates the daily PnL based on equity change."""
        if self.start_of_day_equity is None:
            self.start_of_day_equity = current_equity
        
        # Calculate change from start of day (simplified PnL tracking)
        # In a real system, we'd track realized vs unrealized more carefully
        self.daily_loss = self.start_of_day_equity - current_equity
        
        # If profit, daily_loss is negative
        if self.daily_loss < 0:
            self.daily_loss = 0 # We only care about loss for the limit

    def check_trade_allowed(self, symbol, action, quantity, price):
        """Checks if a trade is allowed based on risk limits."""
        
        # 1. Check Max Order Value
        order_value = quantity * price
        if order_value > MAX_ORDER_VALUE:
            logger.warning(f"Risk Reject: Order value {order_value} > Limit {MAX_ORDER_VALUE}")
            return False

        # 2. Check Daily Loss (both absolute and percentage)
        # We need to fetch current equity to update PnL first, but assuming it's updated periodically
        if self.start_of_day_equity and self.start_of_day_equity > 0:
            loss_percent = (self.daily_loss / self.start_of_day_equity) * 100
            if loss_percent > MAX_DAILY_LOSS_PERCENT:
                logger.warning(f"Risk Reject: Daily loss {loss_percent:.2f}% > Limit {MAX_DAILY_LOSS_PERCENT}%")
                return False
        
        # Also check absolute loss for small accounts
        if self.daily_loss > MAX_DAILY_LOSS:
            logger.warning(f"Risk Reject: Daily loss ${self.daily_loss:.2f} > Limit ${MAX_DAILY_LOSS}")
            return False

        # 3. Check Max Positions (only for BUY orders)
        if action == 'BUY':
            # This requires the bot to have a way to count positions
            # For now, we'll skip or implement a simple check if we can access positions
            pass

        return True
