from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from trader_bot.cost_tracker import CostTracker


def _normalize_tag(raw_tag: Any) -> str:
    if raw_tag is None:
        return ""
    return str(raw_tag).strip().upper().replace(" ", "")


def _parse_numeric(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class AccountSnapshot:
    """
    Normalized view of broker account summary data.

    Keeps UI/service layers decoupled from ib_insync objects or exchange-specific shapes.
    """

    base_currency: str
    net_liquidation: float | None = None
    available_funds: float | None = None
    excess_liquidity: float | None = None
    buying_power: float | None = None
    cash_balances: Dict[str, float] = field(default_factory=dict)
    source: str = "IB"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[dict[str, Any]] | None,
        base_currency: str,
        source: str | None = None,
    ) -> Optional["AccountSnapshot"]:
        if not entries:
            return None

        base_ccy = (base_currency or "USD").upper()
        cash_balances: Dict[str, float] = {}
        mapped: Dict[str, float | None] = {
            "net_liquidation": None,
            "available_funds": None,
            "excess_liquidity": None,
            "buying_power": None,
        }

        tag_map = {
            "NETLIQUIDATION": "net_liquidation",
            "AVAILABLEFUNDS": "available_funds",
            "EXCESSLIQUIDITY": "excess_liquidity",
            "BUYINGPOWER": "buying_power",
            "TOTALCASHVALUE": "cash",
            "CASHBALANCE": "cash",
        }

        for entry in entries:
            tag_norm = _normalize_tag(entry.get("tag"))
            value = _parse_numeric(entry.get("value"))
            currency = (entry.get("currency") or base_ccy).upper()

            if value is None:
                continue

            mapped_field = tag_map.get(tag_norm)
            if mapped_field is None:
                continue

            if mapped_field == "cash":
                cash_balances[currency] = cash_balances.get(currency, 0.0) + value
                continue

            # Prefer base currency entries; fall back to first seen
            if currency == base_ccy or mapped[mapped_field] is None:
                mapped[mapped_field] = value

        return cls(
            base_currency=base_ccy,
            net_liquidation=mapped["net_liquidation"],
            available_funds=mapped["available_funds"],
            excess_liquidity=mapped["excess_liquidity"],
            buying_power=mapped["buying_power"],
            cash_balances=cash_balances,
            source=source or "IB",
        )

    def to_record(self) -> Dict[str, Any]:
        """Dict representation safe for DB persistence."""
        return {
            "base_currency": self.base_currency,
            "net_liquidation": self.net_liquidation,
            "available_funds": self.available_funds,
            "excess_liquidity": self.excess_liquidity,
            "buying_power": self.buying_power,
            "cash_balances": dict(self.cash_balances),
            "timestamp": self.timestamp,
            "source": self.source,
        }


def estimate_commissions_for_orders(
    open_orders: List[dict],
    price_lookup: Optional[Dict[str, float]],
    cost_tracker: CostTracker,
) -> List[dict]:
    """
    Build estimated commission rows for open orders using configured CostTracker.
    Skips rows without enough data to price the estimate.
    """
    estimates = []
    price_lookup = price_lookup or {}
    for order in open_orders or []:
        symbol = order.get("symbol")
        side = (order.get("side") or "").upper()
        qty = order.get("remaining")
        if qty is None:
            qty = order.get("amount")
        price = order.get("price") or price_lookup.get(symbol)
        if not symbol or qty is None or price is None:
            continue
        try:
            qty_val = float(qty)
            price_val = float(price)
        except (TypeError, ValueError):
            continue

        try:
            est_fee = cost_tracker.calculate_trade_fee(symbol, qty_val, price_val, action=side)
        except Exception:
            continue

        estimates.append(
            {
                "symbol": symbol,
                "side": side or "BUY",
                "quantity": qty_val,
                "price": price_val,
                "estimated_fee": est_fee,
                "fee_currency": getattr(cost_tracker, "exchange", "IB"),
            }
        )
    return estimates
