import logging
from typing import Any, Optional


class PortfolioTracker:
    """
    Maintains holdings and portfolio-level accounting (trades, fees, PnL).
    Extracted from StrategyRunner so PnL math and cache updates can be unit tested in isolation.
    """

    def __init__(self, db: Any, session_id: Optional[int] = None, portfolio_id: Optional[int] = None, logger: Optional[logging.Logger] = None):
        self.db = db
        self.session_id = session_id
        self.portfolio_id = portfolio_id
        self.logger = logger or logging.getLogger(__name__)
        self.holdings: dict[str, dict[str, float]] = {}
        self.session_stats = {
            "total_trades": 0,
            "gross_pnl": 0.0,
            "total_fees": 0.0,
            "total_llm_cost": 0.0,
        }

    def set_session(self, session_id: int, portfolio_id: Optional[int] = None) -> None:
        self.session_id = session_id
        if portfolio_id is not None:
            self.portfolio_id = portfolio_id
        elif hasattr(self.db, "get_session_portfolio_id") and self.session_id is not None:
            try:
                self.portfolio_id = self.db.get_session_portfolio_id(self.session_id)
            except Exception:
                self.portfolio_id = None

    def reset_holdings(self) -> None:
        self.holdings.clear()

    def reset_stats(self) -> None:
        self.session_stats.clear()
        self.session_stats.update(
            {
                "total_trades": 0,
                "gross_pnl": 0.0,
                "total_fees": 0.0,
                "total_llm_cost": 0.0,
            }
        )

    def update_holdings_and_realized(self, symbol: str, action: str, quantity: float, price: float, fee: float) -> float:
        """
        Maintain holdings to compute realized PnL per trade.
        Realized PnL is fee-exclusive so costs are handled exactly once in aggregates.
        """
        pos = self.holdings.get(symbol, {"qty": 0.0, "avg_cost": 0.0})
        qty = pos["qty"]
        avg_cost = pos["avg_cost"]
        realized = 0.0

        if action == "BUY":
            new_qty = qty + quantity
            new_avg = ((qty * avg_cost) + (quantity * price)) / new_qty if new_qty > 0 else 0.0
            self.holdings[symbol] = {"qty": new_qty, "avg_cost": new_avg}
            realized = 0.0
        else:  # SELL
            realized = (price - avg_cost) * quantity
            new_qty = max(0.0, qty - quantity)
            self.holdings[symbol] = {"qty": new_qty, "avg_cost": avg_cost if new_qty > 0 else 0.0}

        return realized

    def apply_trade_to_holdings(self, symbol: str, action: str, quantity: float, price: float) -> None:
        """Update holdings without computing realized PnL (used for replays)."""
        pos = self.holdings.get(symbol, {"qty": 0.0, "avg_cost": 0.0})
        qty = pos["qty"]
        avg_cost = pos["avg_cost"]

        if action == "BUY":
            new_qty = qty + quantity
            new_avg = ((qty * avg_cost) + (quantity * price)) / new_qty if new_qty > 0 else 0.0
            self.holdings[symbol] = {"qty": new_qty, "avg_cost": new_avg}
        else:
            new_qty = max(0.0, qty - quantity)
            self.holdings[symbol] = {"qty": new_qty, "avg_cost": avg_cost if new_qty > 0 else 0.0}

    def load_holdings_from_db(self) -> None:
        """Rebuild holdings from historical trades for this session."""
        trades = self.db.get_trades_for_session(self.session_id, portfolio_id=self.portfolio_id)
        self.holdings = {}
        for trade in trades:
            self.apply_trade_to_holdings(
                trade["symbol"],
                trade["action"],
                trade["quantity"],
                trade["price"],
            )

    def apply_fill_to_session_stats(
        self,
        order_id: Optional[str],
        actual_fee: float,
        realized_pnl: float,
        estimated_fee_map: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Reconcile session stats with an executed trade.
        If we estimated a fee earlier, we still book the actual fee but drop the estimate marker.
        """
        if not self.session_stats:
            self.reset_stats()

        if order_id:
            order_key = str(order_id)
            if estimated_fee_map is not None and order_key in estimated_fee_map:
                estimated_fee_map.pop(order_key, None)
        fee_delta = actual_fee
        self.session_stats["total_trades"] += 1
        self.session_stats["total_fees"] += fee_delta
        self.session_stats["gross_pnl"] += realized_pnl
        self._persist_stats_cache()

    @staticmethod
    def extract_fee_cost(fee_field: Any) -> float:
        """Normalize fee representations (dict, list, scalar) to a float cost."""
        if fee_field is None:
            return 0.0
        if isinstance(fee_field, dict):
            try:
                return float(fee_field.get("cost") or 0.0)
            except (TypeError, ValueError):
                return 0.0
        if isinstance(fee_field, (list, tuple)):
            total = 0.0
            for entry in fee_field:
                total += PortfolioTracker.extract_fee_cost(entry)
            return total
        try:
            return float(fee_field)
        except (TypeError, ValueError):
            return 0.0

    def _normalize_exchange_trade(self, trade: dict) -> tuple[str, str, float, float, float] | None:
        """Ensure required fields exist and types are sane for rebuild steps."""
        if not isinstance(trade, dict):
            self.logger.warning("Skipping malformed trade during rebuild: not a dict")
            return None
        try:
            symbol = trade.get("symbol")
            side_raw = trade.get("side")
            quantity = trade.get("amount")
            price = trade.get("price")
            if not symbol or side_raw is None or quantity is None or price is None:
                self.logger.warning("Skipping malformed trade during rebuild: missing required fields")
                return None
            try:
                side = str(side_raw).upper()
                quantity_val = float(quantity)
                price_val = float(price)
            except (TypeError, ValueError) as exc:
                self.logger.warning(f"Skipping malformed trade during rebuild: {exc}")
                return None
            if quantity_val <= 0 or price_val <= 0:
                self.logger.warning("Skipping malformed trade during rebuild: non-positive quantity or price")
                return None
            fee_cost = self.extract_fee_cost(trade.get("fee"))
            return symbol, side, quantity_val, price_val, fee_cost
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"Skipping malformed trade during rebuild: {exc}")
            return None

    def apply_exchange_trades_for_rebuild(self, trades: list) -> dict:
        """
        Rebuild holdings and session stats from a list of exchange trades.
        Malformed entries are skipped with warnings instead of breaking the rebuild.
        """
        self.reset_holdings()
        self.reset_stats()
        skipped = 0
        for trade in trades or []:
            normalized = self._normalize_exchange_trade(trade)
            if not normalized:
                skipped += 1
                continue
            symbol, side, quantity, price, fee_cost = normalized
            try:
                realized = self.update_holdings_and_realized(symbol, side, quantity, price, fee_cost)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(f"Could not apply trade during rebuild for {symbol}: {exc}")
                skipped += 1
                continue

            self.session_stats["total_trades"] += 1
            self.session_stats["total_fees"] += fee_cost
            self.session_stats["gross_pnl"] += realized

        if skipped:
            self.logger.warning(f"Skipped {skipped} malformed trades while rebuilding stats")
        self._persist_stats_cache()
        return self.session_stats

    def rebuild_session_stats_from_trades(self, current_equity: float | None = None) -> dict:
        """Recompute session_stats from recorded trades and update cache."""
        trades = self.db.get_trades_for_session(self.session_id, portfolio_id=self.portfolio_id)
        self.reset_holdings()
        self.reset_stats()
        for trade in trades:
            try:
                fee_cost = self.extract_fee_cost(trade.get("fee"))
                realized = self.update_holdings_and_realized(
                    trade["symbol"],
                    trade["action"],
                    trade["quantity"],
                    trade["price"],
                    fee_cost,
                )
                self.session_stats["total_trades"] += 1
                self.session_stats["total_fees"] += fee_cost
                self.session_stats["gross_pnl"] += realized
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(f"Skipping trade during stats rebuild due to error: {exc}")

        # Pull LLM costs from session row
        db_stats = {}
        if self.portfolio_id is not None and hasattr(self.db, "get_portfolio_stats_cache"):
            try:
                db_stats = self.db.get_portfolio_stats_cache(self.portfolio_id) or {}
            except Exception:
                db_stats = {}
        if not db_stats and self.session_id is not None:
            db_stats = self.db.get_session_stats(self.session_id)
        self.session_stats["total_llm_cost"] = db_stats.get("total_llm_cost", 0.0)
        self._persist_stats_cache()
        try:
            if hasattr(self.db, "update_session_totals") and self.session_id is not None:
                self.db.update_session_totals(
                    self.session_id,
                    total_trades=self.session_stats["total_trades"],
                    total_fees=self.session_stats["total_fees"],
                    total_llm_cost=self.session_stats["total_llm_cost"],
                    net_pnl=self.session_stats["gross_pnl"]
                    - self.session_stats["total_fees"]
                    - self.session_stats["total_llm_cost"],
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"Could not update session totals: {exc}")
        return self.session_stats

    def _persist_stats_cache(self) -> None:
        """Write stats aggregates to the appropriate cache scope."""
        try:
            if self.portfolio_id is not None and hasattr(self.db, "set_portfolio_stats_cache"):
                self.db.set_portfolio_stats_cache(self.portfolio_id, self.session_stats)
            elif self.session_id is not None and hasattr(self.db, "set_session_stats_cache"):
                self.db.set_session_stats_cache(self.session_id, self.session_stats)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"Failed to persist session stats cache: {exc}")
