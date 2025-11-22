import logging
from config import MAX_DAILY_LOSS, MAX_ORDER_VALUE, MAX_POSITIONS

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, bot=None):
        self.bot = bot # Optional, mostly for position checks if needed
        self.daily_loss = 0.0
        self.positions = {} # Symbol -> Quantity

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

        # 2. Check Daily Loss
        # We need to fetch current equity to update PnL first, but assuming it's updated periodically
        if self.daily_loss > MAX_DAILY_LOSS:
            logger.warning(f"Risk Reject: Daily loss {self.daily_loss} > Limit {MAX_DAILY_LOSS}")
            return False

        # 3. Check Max Positions (only for BUY orders)
        if action == 'BUY':
            # This requires the bot to have a way to count positions
            # For now, we'll skip or implement a simple check if we can access positions
            pass

        return True
