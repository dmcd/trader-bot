import logging
import math
from dataclasses import dataclass

from trader_bot.config import (
    MAX_ORDER_VALUE,
    MAX_POSITIONS,
    MAX_TOTAL_EXPOSURE,
    MIN_TRADE_SIZE,
    ORDER_VALUE_BUFFER,
    CORRELATION_BUCKETS,
    BUCKET_MAX_POSITIONS,
    PORTFOLIO_BASE_CURRENCY,
)
from trader_bot.symbols import normalize_symbol

logger = logging.getLogger(__name__)


class QuoteToBaseConverter:
    """
    Lightweight currency converter for risk checks.

    Expects `fx_rate_provider` to return a rate that converts FROM the given
    currency INTO `base_currency`. Falls back to 1.0 when the quote already
    matches the base.
    """

    def __init__(self, base_currency: str | None, fx_rate_provider=None):
        self.base_currency = base_currency.upper() if base_currency else None
        self.fx_rate_provider = fx_rate_provider

    def convert_notional(self, symbol: str | None, notional_quote: float, price: float | None = None) -> tuple[float, float | None]:
        """
        Convert notional from quote currency to base currency.

        Returns (converted_notional, fx_rate_used or None).
        """
        try:
            notional_val = float(notional_quote)
        except (TypeError, ValueError):
            return 0.0, None
        if math.isnan(notional_val) or math.isinf(notional_val):
            return 0.0, None
        if self.base_currency is None:
            return notional_val, 1.0

        base, quote = self._split_symbol(symbol)
        if quote is None or quote == self.base_currency:
            return notional_val, 1.0

        rate = self._lookup_fx_rate(quote, symbol=symbol, price=price)
        if rate is None and base == self.base_currency and price not in (0, None):
            try:
                if not math.isnan(price) and not math.isinf(price):
                    rate = 1 / price
            except Exception:
                rate = None

        if rate is None:
            return notional_val, None
        return notional_val * rate, rate

    def _lookup_fx_rate(self, currency: str, symbol: str | None = None, price: float | None = None) -> float | None:
        if currency is None or currency == self.base_currency:
            return 1.0
        provider = self.fx_rate_provider
        if provider:
            try:
                return provider(currency, symbol=symbol, price=price)
            except TypeError:
                # Support providers without symbol/price kwargs
                try:
                    return provider(currency)
                except Exception:
                    return None
            except Exception:
                return None
        return None

    @staticmethod
    def _split_symbol(symbol: str | None) -> tuple[str | None, str | None]:
        if not symbol:
            return None, None
        try:
            normalized = normalize_symbol(symbol)
            base, quote = normalized.split("/", 1)
            return base, quote
        except Exception:
            return None, None


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str = ""

class RiskManager:
    def __init__(
        self,
        bot=None,
        ignore_baseline_positions: bool = False,
        base_currency: str | None = None,
        fx_rate_provider=None,
        portfolio_id: int | None = None,
    ):
        self.bot = bot # Optional, mostly for position checks if needed
        self.current_equity: float | None = None
        self.positions = {} # Symbol -> Quantity
        self.pending_buy_exposure = 0.0  # Notional of outstanding buy orders
        self.pending_sell_exposure = 0.0  # Notional of outstanding sell orders (short intent)
        self.pending_orders_by_symbol = {}  # symbol -> {'buy': notional, 'sell': notional, 'count_buy': int, 'count_sell': int}
        self.correlation_buckets = CORRELATION_BUCKETS
        self.bucket_max_positions = BUCKET_MAX_POSITIONS
        self.ignore_baseline_positions = ignore_baseline_positions
        # Baseline positions seen at startup (used to ignore sandbox airdrops)
        self.position_baseline: dict[str, float] = {}
        self.fx_rate_provider = fx_rate_provider
        self.base_currency = None
        self._converter = QuoteToBaseConverter(None, fx_rate_provider)
        resolved_base_currency = base_currency or PORTFOLIO_BASE_CURRENCY
        self.set_base_currency(resolved_base_currency)
        self.portfolio_id = portfolio_id
        self.baseline_equity: float | None = None
        self.baseline_timestamp: str | None = None

    def seed_start_of_day(self, start_equity: float):
        """Backward-compatible alias for seeding current equity on startup."""
        self.update_equity(start_equity)

    def set_baseline(self, equity: float | None, timestamp: str | None = None):
        """
        Store portfolio-level baseline metadata for telemetry/context.
        This is optional and does not gate risk checks.
        """
        try:
            self.baseline_equity = float(equity) if equity is not None else None
        except (TypeError, ValueError):
            self.baseline_equity = None
        self.baseline_timestamp = timestamp

    def update_equity(self, current_equity: float):
        """Track latest portfolio equity for telemetry."""
        if current_equity is None:
            return
        try:
            self.current_equity = float(current_equity)
        except (TypeError, ValueError):
            return

    # Backward compatibility
    update_pnl = update_equity

    def update_positions(self, positions: dict):
        """Updates the current positions for exposure calculation.
        positions: dict of {symbol: {'quantity': float, 'current_price': float}}
        """
        self.positions = positions
        if self.ignore_baseline_positions and positions:
            for sym, data in positions.items():
                if sym not in self.position_baseline:
                    qty = data.get('quantity', 0.0) or 0.0
                    self.position_baseline[sym] = qty

    def set_position_baseline(self, baseline: dict):
        """Set or extend the baseline positions that should be ignored for exposure."""
        if not baseline:
            return
        # Do not overwrite existing baselines so restarts retain the initial snapshot
        for sym, qty in baseline.items():
            self.position_baseline.setdefault(sym, qty or 0.0)

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
            notional_quote = price * qty
            notional, _ = self._convert_notional_to_base(symbol, notional_quote, price)
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

    def _convert_notional_to_base(self, symbol: str | None, notional: float, price: float | None = None) -> tuple[float, float | None]:
        return self._converter.convert_notional(symbol, notional, price)

    def set_portfolio(self, portfolio_id: int | None):
        """Update the active portfolio scope for downstream services."""
        self.portfolio_id = portfolio_id

    def set_base_currency(self, base_currency: str | None, fx_rate_provider=None):
        """Update the portfolio base currency and reset the converter."""
        if fx_rate_provider is not None:
            self.fx_rate_provider = fx_rate_provider
        self.base_currency = base_currency.upper() if base_currency else None
        self._converter = QuoteToBaseConverter(self.base_currency, self.fx_rate_provider)

    def apply_order_value_buffer(self, quantity: float, price: float, symbol: str | None = None):
        """Trim quantity so notional stays under the order cap minus buffer."""
        def _invalid_number(val: float | int | None) -> bool:
            try:
                return val is None or math.isnan(float(val)) or math.isinf(float(val))
            except Exception:
                return True

        if _invalid_number(price) or _invalid_number(quantity):
            return 0.0, 0.0
        if price <= 0 or quantity <= 0:
            return quantity, 0.0

        notional = quantity * price
        order_value, rate_used = self._convert_notional_to_base(symbol, notional, price)
        rate_used = rate_used or 1.0

        capped_value = max(0.0, MAX_ORDER_VALUE - ORDER_VALUE_BUFFER)
        if order_value <= capped_value:
            return quantity, 0.0

        capped_qty = capped_value / (price * rate_used) if price else 0.0
        capped_qty = max(0.0, capped_qty)
        overage = order_value - capped_value
        return capped_qty, overage

    def check_trade_allowed(self, symbol, action, quantity, price) -> RiskCheckResult:
        """Checks if a trade is allowed based on risk limits."""
        def _invalid_number(val: float | int | None) -> bool:
            try:
                return val is None or math.isnan(float(val)) or math.isinf(float(val))
            except Exception:
                return True

        if _invalid_number(price) or _invalid_number(quantity):
            return RiskCheckResult(False, "Invalid price or quantity")
        if price <= 0 or quantity <= 0:
            return RiskCheckResult(False, "Invalid price or quantity")
        
        # 1. Check Max Order Value
        order_value_quote = quantity * price
        order_value, _ = self._convert_notional_to_base(symbol, order_value_quote, price)
        currency = self.base_currency or "base"
        if order_value > MAX_ORDER_VALUE:
            msg = f"Order value {order_value:.2f} {currency} exceeds limit of {MAX_ORDER_VALUE:.2f} {currency}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)

        # 1.5 Check Min Trade Size
        if order_value < MIN_TRADE_SIZE:
            msg = f"Order value {order_value:.2f} {currency} is below minimum of {MIN_TRADE_SIZE:.2f} {currency}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)

        # 2. Exposure and position caps
        price_overrides = {symbol: price} if price else None
        current_exposure = self.get_total_exposure(price_overrides=price_overrides)

        def projected_exposure_for_sell(sym_qty, qty, px):
            # Existing short: any additional sell adds to exposure in base currency
            if sym_qty < 0:
                return current_exposure + order_value

            # If selling more than current long, the overage is new short exposure
            if qty > sym_qty:
                overage = (qty - sym_qty) * px
                overage_base, _ = self._convert_notional_to_base(symbol, overage, px)
                return current_exposure + overage_base

            # Otherwise exposure shrinks or stays; allow
            return max(0.0, current_exposure - min(order_value, current_exposure))

        projected_exposure = current_exposure
        if action == 'BUY':
            projected_exposure = current_exposure + order_value
        else:  # SELL
            sym_qty = (self.positions or {}).get(symbol, {}).get('quantity', 0.0) or 0.0
            projected_exposure = projected_exposure_for_sell(sym_qty, quantity, price)

        if projected_exposure > MAX_TOTAL_EXPOSURE:
            currency_label = self.base_currency or "base"
            msg = f"Total exposure {projected_exposure:.2f} {currency_label} would exceed limit of {MAX_TOTAL_EXPOSURE:.2f} {currency_label}"
            logger.warning(f"Risk Reject: {msg}")
            return RiskCheckResult(False, msg)
        if projected_exposure > (MAX_TOTAL_EXPOSURE * 0.9):
            currency_label = self.base_currency or "base"
            logger.warning(f"Risk Warning: Total exposure {projected_exposure:.2f} {currency_label} is close to limit of {MAX_TOTAL_EXPOSURE:.2f} {currency_label}")

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

            # Correlation bucket guard: block same-direction adds when bucket already loaded
            bucket = self._get_bucket(symbol)
            if bucket:
                bucket_syms = set(bucket)
                # Count active positions and pending buys in the same bucket
                active_bucket = {
                    sym for sym, data in (self.positions or {}).items()
                    if sym in bucket_syms and abs(data.get('quantity', 0) or 0.0) > 1e-9
                }
                pending_bucket = {
                    sym for sym, data in (self.pending_orders_by_symbol or {}).items()
                    if sym in bucket_syms and (data.get('count_buy', 0) > 0)
                }
                # Exclude current symbol if we already have a position; focus on stacking new symbols in bucket
                bucket_count = len(active_bucket.union(pending_bucket))
                already_active = symbol in active_bucket
                if not already_active and bucket_count >= self.bucket_max_positions:
                    msg = f"Correlation bucket limit reached for {symbol} ({bucket_count}/{self.bucket_max_positions})"
                    logger.warning(f"Risk Reject: {msg}")
                    return RiskCheckResult(False, msg)

        return RiskCheckResult(True, "Trade allowed")

    def _get_bucket(self, symbol: str):
        if not symbol or not self.correlation_buckets:
            return None
        sym_up = symbol.upper()
        for _, members in self.correlation_buckets.items():
            if sym_up in members:
                return members
        return None

    def _net_quantity_for_exposure(self, quantity: float, baseline_qty: float) -> float:
        """Apply sandbox baseline so initial airdrop inventory doesn't consume exposure."""
        if not self.ignore_baseline_positions:
            return quantity

        quantity = quantity or 0.0
        baseline_qty = baseline_qty or 0.0

        # Baseline long: ignore exposure until we exceed baseline; shorts count fully
        if baseline_qty >= 0:
            if quantity >= 0:
                return max(0.0, quantity - baseline_qty)
            return quantity

        # Baseline short: ignore exposure within [baseline, 0]; long flips count fully
        if quantity <= 0:
            return min(0.0, quantity - baseline_qty)
        return quantity

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

            baseline_qty = self.position_baseline.get(sym, 0.0) if self.position_baseline else 0.0
            qty_for_exposure = self._net_quantity_for_exposure(qty, baseline_qty)
            notional_quote = qty_for_exposure * curr_price
            notional_base, _ = self._convert_notional_to_base(sym, notional_quote, curr_price)
            notional_abs = abs(notional_base)
            per_symbol_notional[sym] = notional_abs
            exposure += notional_abs

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
