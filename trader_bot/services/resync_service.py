import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Optional

from trader_bot.config import (
    CLIENT_ORDER_PREFIX,
    COMMAND_RETENTION_DAYS,
    TRADE_SYNC_CUTOFF_MINUTES,
)
from trader_bot.utils import get_client_order_id


class ResyncService:
    """
    Handles reconciliation of positions/open orders and syncing fills from the exchange.
    Extracted from StrategyRunner to isolate DB/venue reconciliation logic.
    """

    def __init__(
        self,
        db: Any,
        bot: Any,
        risk_manager: Any,
        holdings_updater: Callable[[str, str, float, float, float], float],
        session_stats_applier: Callable[[str, float, float], None],
        record_health_state: Optional[Callable[[str, str, Optional[dict]], None]] = None,
        logger: Optional[logging.Logger] = None,
        trade_sync_cutoff_minutes: int = TRADE_SYNC_CUTOFF_MINUTES,
        portfolio_id: Optional[int] = None,
    ):
        self.db = db
        self.bot = bot
        self.risk_manager = risk_manager
        self.holdings_updater = holdings_updater
        self.session_stats_applier = session_stats_applier
        self.record_health_state = record_health_state or (lambda *_: None)
        self.logger = logger or logging.getLogger(__name__)
        self.session_id: Optional[int] = None
        self.portfolio_id: Optional[int] = portfolio_id
        self.trade_sync_cutoff_minutes = trade_sync_cutoff_minutes

    def set_session(self, session_id: int, portfolio_id: Optional[int] = None) -> None:
        self.session_id = session_id
        if portfolio_id is not None:
            self.portfolio_id = portfolio_id

    def filter_our_orders(self, orders: list) -> list:
        """Only keep open orders with our client id prefix."""
        filtered = []
        for order in orders or []:
            client_oid = get_client_order_id(order)
            if client_oid and client_oid.startswith(CLIENT_ORDER_PREFIX):
                filtered.append(order)
        return filtered

    async def reconcile_open_orders(self):
        """
        Refresh open order snapshot using live exchange data and drop any DB orders
        that no longer exist on the venue.
        """
        if not self.session_id:
            return
        try:
            exchange_orders = await self.bot.get_open_orders_async()
            exchange_orders = self.filter_our_orders(exchange_orders)
        except Exception as exc:
            self.logger.warning(f"Could not fetch open orders for reconciliation: {exc}")
            return

        try:
            db_orders = self.db.get_open_orders(self.session_id, portfolio_id=self.portfolio_id)
        except Exception as exc:
            self.logger.warning(f"Could not load open orders from DB for reconciliation: {exc}")
            db_orders = []

        db_ids = {str(o.get('order_id')) for o in db_orders if o.get('order_id')}
        exch_ids = {str(o.get('order_id') or o.get('id')) for o in exchange_orders if o.get('order_id') or o.get('id')}
        stale = db_ids - exch_ids
        if stale:
            self.logger.info(f"🧹 Removed {len(stale)} stale open orders not on exchange")

        try:
            self.db.replace_open_orders(self.session_id, exchange_orders, portfolio_id=self.portfolio_id)
        except Exception as exc:
            self.logger.warning(f"Could not refresh open orders snapshot: {exc}")

    async def reconcile_exchange_state(self):
        """
        Reconcile positions and open orders against the live exchange at startup.
        Ensures DB snapshots and risk manager state reflect actual venue state.
        """
        if not self.session_id:
            return
        try:
            live_positions = await self.bot.get_positions_async()
        except Exception as exc:
            self.record_health_state("restart_recovery", "error", {"stage": "positions", "error": str(exc)})
            self.logger.warning(f"Could not fetch live positions during recovery: {exc}")
            return

        try:
            live_orders = await self.bot.get_open_orders_async()
            live_orders = self.filter_our_orders(live_orders)
        except Exception as exc:
            self.record_health_state("restart_recovery", "error", {"stage": "open_orders", "error": str(exc)})
            self.logger.warning(f"Could not fetch live open orders during recovery: {exc}")
            return

        # Load existing snapshots
        try:
            db_positions = self.db.get_positions(self.session_id, portfolio_id=self.portfolio_id) or []
            db_orders = self.db.get_open_orders(self.session_id, portfolio_id=self.portfolio_id) or []
        except Exception as exc:
            self.record_health_state("restart_recovery", "error", {"stage": "db_read", "error": str(exc)})
            self.logger.warning(f"Could not load DB snapshots during recovery: {exc}")
            return

        # Replace snapshots with live state
        try:
            self.db.replace_positions(self.session_id, live_positions, portfolio_id=self.portfolio_id)
            self.db.replace_open_orders(self.session_id, live_orders, portfolio_id=self.portfolio_id)
        except Exception as exc:
            self.record_health_state("restart_recovery", "error", {"stage": "db_write", "error": str(exc)})
            self.logger.warning(f"Could not persist reconciled snapshots: {exc}")
            return

        # Update risk manager with live state
        try:
            positions_dict = {}
            for pos in live_positions or []:
                symbol = pos.get("symbol")
                if not symbol:
                    continue
                mark = pos.get("current_price") or pos.get("avg_price") or 0.0
                positions_dict[symbol] = {"quantity": pos.get("quantity", 0.0), "current_price": mark}
            self.risk_manager.update_positions(positions_dict)
            self.risk_manager.update_pending_orders(live_orders, price_lookup=None)
        except Exception as exc:
            self.logger.debug(f"Risk manager update after recovery failed: {exc}")

        detail = {
            "positions_before": len(db_positions),
            "positions_after": len(live_positions or []),
            "orders_before": len(db_orders),
            "orders_after": len(live_orders or []),
        }
        self.record_health_state("restart_recovery", "ok", detail)
        self.logger.info(
            f"🧹 Startup reconciliation applied: positions {detail['positions_before']}→{detail['positions_after']}, "
            f"open orders {detail['orders_before']}→{detail['orders_after']}"
        )

    async def sync_trades_from_exchange(
        self,
        session_id: int,
        processed_trade_ids: set[tuple[str, str | None]],
        order_reasons: dict,
        plan_reason_lookup: Callable[[int, str, str], Optional[str]],
        get_symbols: Callable[[], set],
    ):
        """Sync recent trades from exchange to DB."""
        if not session_id:
            return

        new_processed: set[tuple[str, str | None]] = set()
        if not processed_trade_ids:
            try:
                persisted_ids = self.db.get_processed_trade_ids(session_id, portfolio_id=self.portfolio_id)
                processed_trade_ids.update(persisted_ids or set())
            except Exception as exc:
                self.logger.debug(f"Could not load processed trade ids: {exc}")
        try:
            if not processed_trade_ids:
                persisted_ids = self.db.get_processed_trade_ids(session_id, portfolio_id=self.portfolio_id)
                processed_trade_ids.update(persisted_ids or set())
        except Exception as exc:
            self.logger.debug(f"Could not load processed trade ids: {exc}")

        try:
            symbols = get_symbols()
            if not symbols:
                symbols = {"BTC/USD"}

            since_iso = self.db.get_latest_trade_timestamp(session_id, portfolio_id=self.portfolio_id)
            since_ms = None
            if since_iso:
                try:
                    since_dt = datetime.fromisoformat(since_iso)
                    since_ms = int(since_dt.timestamp() * 1000) - 5000
                except Exception:
                    since_ms = None
            cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=self.trade_sync_cutoff_minutes)

            for symbol in symbols:
                cursor_since = since_ms
                while True:
                    trades = await self.bot.get_my_trades_async(symbol, since=cursor_since, limit=100)
                    filtered_trades = []
                    for trade in trades:
                        client_oid = get_client_order_id(trade)
                        if not client_oid:
                            continue
                        if not client_oid.startswith(CLIENT_ORDER_PREFIX):
                            continue
                        trade["_client_oid"] = client_oid
                        filtered_trades.append(trade)
                    trades = filtered_trades

                    if not trades:
                        break

                    latest_ts = None
                    for trade in trades:
                        client_oid = trade.get("_client_oid") or get_client_order_id(trade)
                        trade_id = str(trade["id"])
                        if trade_id in processed_trade_ids:
                            continue

                        if self.portfolio_id is not None:
                            existing = self.db.conn.execute(
                                "SELECT id FROM trades WHERE trade_id = ? AND portfolio_id = ?",
                                (trade_id, self.portfolio_id),
                            ).fetchone()
                        else:
                            existing = self.db.conn.execute(
                                "SELECT id FROM trades WHERE trade_id = ? AND session_id = ?",
                                (trade_id, session_id),
                            ).fetchone()
                        if existing:
                            processed_trade_ids.add(trade_id)
                            new_processed.add((trade_id, client_oid))
                            continue

                        ts_ms = trade.get("timestamp")
                        if ts_ms:
                            trade_dt = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
                            if trade_dt < cutoff_dt:
                                processed_trade_ids.add(trade_id)
                                new_processed.add((trade_id, client_oid))
                                continue

                        order_id = trade.get("order")
                        side = trade["side"].upper()
                        price = trade["price"]
                        quantity = trade["amount"]
                        fee = trade.get("fee", {}).get("cost", 0.0)

                        info = trade.get("info") or {}
                        liquidity = (
                            trade.get("liquidity")
                            or info.get("liquidity")
                            or info.get("fillLiquidity")
                            or info.get("liquidityIndicator")
                            or "unknown"
                        )
                        if liquidity:
                            liquidity = str(liquidity).lower()

                        try:
                            plan_reason = plan_reason_lookup(session_id, order_id, client_oid)
                        except Exception:
                            plan_reason = None
                        reason = order_reasons.get(str(order_id)) or plan_reason
                        if not reason:
                            processed_trade_ids.add(trade_id)
                            new_processed.add((trade_id, client_oid))
                            continue

                        realized_pnl = self.holdings_updater(symbol, side, quantity, price, fee)

                        self.db.log_trade(
                            session_id,
                            symbol,
                            side,
                            quantity,
                            price,
                            fee,
                            reason,
                            liquidity=liquidity,
                            realized_pnl=realized_pnl,
                            trade_id=trade_id,
                            timestamp=trade.get("datetime"),
                            portfolio_id=self.portfolio_id,
                        )
                        self.session_stats_applier(order_id, fee, realized_pnl)
                        processed_trade_ids.add(trade_id)
                        new_processed.add((trade_id, client_oid))
                        self.logger.info(f"✅ Synced trade: {side} {quantity} {symbol} @ ${price:,.2f} (Fee: ${fee:.4f})")

                        ts = trade.get("timestamp")
                        if ts is not None:
                            latest_ts = max(latest_ts or ts, ts)

                    if latest_ts is None or len(trades) < 100:
                        break
                    cursor_since = latest_ts + 1

        except Exception as exc:  # pragma: no cover - defensive
            self.logger.exception(f"Error syncing trades: {exc}")
        finally:
            if new_processed:
                try:
                    self.db.record_processed_trade_ids(session_id, new_processed, portfolio_id=self.portfolio_id)
                except Exception as exc:
                    self.logger.debug(f"Could not persist processed trade ids: {exc}")
