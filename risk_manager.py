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
        self.pending_buy_exposure = 0.0  # Notional of outstanding buy orders
        self.pending_sell_exposure = 0.0  # Notional of outstanding sell orders (short intent)
        self.pending_orders_by_symbol = {}  # symbol -> {'buy': notional, 'sell': notional, 'count_buy': int, 'count_sell': int}

    def seed_start_of_day(self, start_equity: float):
        """Persist start-of-day equity so restarts keep loss limits consistent."""
        if start_equity is not None:
            self.start_of_day_equity = start_equity

    def update_equity(self, current_equity: float):
        """Track drawdown off start-of-day equity (keeps loss limits consistent)."""
        if current_equity is None:
            return

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

    def update_pending_orders(self, pending_orders: list, price_lookup: dict = None):
        """
        Track notional exposure from outstanding orders to avoid over-allocation.
        Both BUY and SELL sides are tracked so shorts/hedges consume gross exposure.
        """
        buy_total = 0.0
        sell_total = 0.0
        counts = {}
        price_lookup = price_lookup or {}
        for order in pending_orders or []:
            symbol = order.get('symbol')
            side = (order.get('side') or '').upper()
            price = order.get('price') or price_lookup.get(symbol) or 0.0
            qty = order.get('remaining')
            if qty is None:
                qty = order.get('amount', 0.0)
            if not price or not qty:
                continue
            notional = price * qty
            sym_entry = counts.setdefault(symbol, {'buy': 0.0, 'sell': 0.0, 'count_buy': 0, 'count_sell': 0})
            if side == 'BUY':
                buy_total += notional
                sym_entry['buy'] += notional
                sym_entry['count_buy'] += 1
            elif side == 'SELL':
                sell_total += notional
                sym_entry['sell'] += notional
                sym_entry['count_sell'] += 1
        self.pending_buy_exposure = buy_total
        self.pending_sell_exposure = sell_total
        self.pending_orders_by_symbol = counts

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

        # 3. Exposure and position caps
        price_overrides = {symbol: price} if price else None
        current_exposure = self.get_total_exposure(price_overrides=price_overrides)

        def projected_exposure_for_sell(sym_qty, qty, px):
            # If selling more than current long, the overage is new short exposure
            if qty > sym_qty:
                return current_exposure + (qty - sym_qty) * px
            # Otherwise exposure shrinks or stays; allow
            return current_exposure

        projected_exposure = current_exposure
        if action == 'BUY':
            projected_exposure = current_exposure + order_value
        else:  # SELL
            sym_qty = (self.positions or {}).get(symbol, {}).get('quantity', 0.0) or 0.0
            projected_exposure = projected_exposure_for_sell(sym_qty, quantity, price)

        if projected_exposure > MAX_TOTAL_EXPOSURE:
            msg = f"Total exposure ${projected_exposure:.2f} would exceed limit of ${MAX_TOTAL_EXPOSURE:.2f}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)
        if projected_exposure > (MAX_TOTAL_EXPOSURE * 0.9):
            logger.warning(f"Risk Warning: Total exposure ${projected_exposure:.2f} is close to limit of ${MAX_TOTAL_EXPOSURE:.2f}")

        # Enforce max distinct positions (including pending buys on new symbols)
        if action == 'BUY':
            active_positions = {
                sym for sym, data in (self.positions or {}).items()
                if abs(data.get('quantity', 0) or 0.0) > 1e-9
            }
            pending_symbols = {
                sym for sym, data in (self.pending_orders_by_symbol or {}).items()
                if (data.get('count_buy', 0) + data.get('count_sell', 0)) > 0
            }
            distinct_symbols = active_positions.union(pending_symbols)
            has_position = symbol in active_positions
            if not has_position and symbol not in pending_symbols and len(distinct_symbols) >= MAX_POSITIONS:
                msg = f"Max positions limit reached ({len(distinct_symbols)}/{MAX_POSITIONS})"
                logger.warning(f"Risk Reject: {msg}")
                return RiskCheckResult(False, msg)

            # Cap stacking multiple pending buys on the same symbol
            pending_for_symbol = self.pending_orders_by_symbol.get(symbol, {}) if self.pending_orders_by_symbol else {}
            if pending_for_symbol.get('count_buy', 0) >= MAX_POSITIONS:
                msg = f"Open order cap reached for {symbol} ({pending_for_symbol.get('count_buy')}/{MAX_POSITIONS})"
                logger.warning(f"Risk Reject: {msg}")
                return RiskCheckResult(False, msg)

        return RiskCheckResult(True, "Trade allowed")

    def get_total_exposure(self, price_overrides: dict = None) -> float:
        """Return total notional exposure using marked prices."""
        exposure = 0.0
        per_symbol_notional = {}
        for sym, data in self.positions.items():
            qty = data.get('quantity', 0) or 0.0
            curr_price = 0.0

            if price_overrides and sym in price_overrides and price_overrides[sym]:
                curr_price = price_overrides[sym]
            else:
                curr_price = data.get('current_price', 0) or 0.0

            notional = abs(qty) * curr_price
            per_symbol_notional[sym] = notional
            exposure += notional

        # Pending buys always consume headroom
        exposure += self.pending_buy_exposure

        # Pending sells can offset existing longs; only remainder adds exposure (short intent)
        for sym, counts in (self.pending_orders_by_symbol or {}).items():
            sell_notional = counts.get('sell', 0.0) or 0.0
            if sell_notional <= 0:
                continue
            long_notional = per_symbol_notional.get(sym, 0.0)
            offset = min(long_notional, sell_notional)
            exposure -= offset
            remainder = sell_notional - offset
            if remainder > 0:
                exposure += remainder

        return max(0.0, exposure)
