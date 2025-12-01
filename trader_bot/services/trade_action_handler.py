import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional


class TradeActionHandler:
    """
    Handles plan updates and flattening actions plus sizing/microstructure helpers.
    Extracted from StrategyRunner for isolation and testability.
    """

    def __init__(
        self,
        db: Any,
        bot: Any,
        risk_manager: Any,
        cost_tracker: Any,
        portfolio_tracker: Any,
        prefer_maker: Callable[[str], bool],
        health_manager: Any,
        emit_telemetry: Callable[[dict], None],
        log_execution_trace: Callable[[Any, dict], None],
        on_trade_rejected: Optional[Callable[[str], None]] = None,
        actions_logger: Optional[logging.Logger] = None,
        logger: Optional[logging.Logger] = None,
        portfolio_id: Optional[int] = None,
    ):
        self.db = db
        self.bot = bot
        self.risk_manager = risk_manager
        self.cost_tracker = cost_tracker
        self.portfolio_tracker = portfolio_tracker
        self.prefer_maker = prefer_maker
        self.health_manager = health_manager
        self.emit_telemetry = emit_telemetry
        self.log_execution_trace = log_execution_trace
        self.on_trade_rejected = on_trade_rejected
        self.actions_logger = actions_logger or logging.getLogger(__name__)
        self.logger = logger or logging.getLogger(__name__)
        self.portfolio_id = portfolio_id

    @staticmethod
    def _merge_ib_order_metadata(telemetry_record: dict, order_result: dict | None):
        if not isinstance(order_result, dict):
            return
        mapping = {
            "contract_id": "ib_contract_id",
            "exchange": "ib_exchange",
            "primary_exchange": "ib_primary_exchange",
            "commission_source": "ib_commission_source",
            "instrument_type": "instrument_type",
        }
        for source, target in mapping.items():
            val = order_result.get(source)
            if val is not None:
                telemetry_record[target] = val

    # --- Helper sizing/microstructure utilities ---
    def apply_order_value_buffer(self, quantity: float, price: float, symbol: str | None = None):
        """Trim quantity so the notional sits under the order cap minus buffer."""
        adjusted_qty, overage = self.risk_manager.apply_order_value_buffer(quantity, price, symbol=symbol)
        if adjusted_qty < quantity:
            original_value = quantity * price
            adjusted_value = adjusted_qty * price
            self.actions_logger.info(
                f"✂️ Trimmed order from ${original_value:.2f} to ${adjusted_value:.2f} "
                f"to stay under risk cap"
            )
        return adjusted_qty

    def passes_rr_filter(self, action: str, price: float, stop_price: float, target_price: float, min_rr: float) -> bool:
        """Require minimum risk/reward when both stop and target are provided."""
        if not price or stop_price is None or target_price is None:
            return True
        if action == "BUY":
            risk = price - stop_price
            reward = target_price - price
        else:
            risk = stop_price - price
            reward = price - target_price
        if risk <= 0 or reward <= 0:
            return False
        rr = reward / risk
        return rr >= min_rr

    def compute_slippage_cap(
        self,
        market_data_point: dict,
        max_slippage_pct: float,
        max_spread_pct: float,
        min_top_of_book_notional: float,
    ) -> float:
        """Derive a slippage cap that tightens on thin books or wide spreads."""
        cap = max_slippage_pct
        if not market_data_point:
            return cap

        spread_pct = market_data_point.get("spread_pct")
        bid = market_data_point.get("bid")
        ask = market_data_point.get("ask")
        bid_size = market_data_point.get("bid_size")
        ask_size = market_data_point.get("ask_size")

        top_notional = None
        if bid and ask and bid_size and ask_size:
            top_notional = min(bid * bid_size, ask * ask_size)

        if top_notional is not None:
            if top_notional < min_top_of_book_notional:
                cap *= 0.25
            elif top_notional < min_top_of_book_notional * 2:
                cap *= 0.5

        if spread_pct is not None and spread_pct > 0:
            factor = max(0.3, min(1.0, max_spread_pct / max(spread_pct, 1e-9)))
            cap *= factor

        return max(cap, max_slippage_pct * 0.1)

    def slippage_within_limit(
        self,
        decision_price: float,
        latest_price: float,
        market_data_point: dict,
        max_slippage_pct: float,
        max_spread_pct: float,
        min_top_of_book_notional: float,
    ):
        """Return (allowed, move_pct) based on dynamic slippage cap."""
        if not decision_price or not latest_price:
            return True, 0.0
        move_pct = abs(latest_price - decision_price) / decision_price * 100
        cap = self.compute_slippage_cap(market_data_point or {}, max_slippage_pct, max_spread_pct, min_top_of_book_notional)
        return move_pct <= cap, move_pct

    def liquidity_ok(self, market_data_point: dict, max_spread_pct: float, min_top_of_book_notional: float, min_quote_size: float | None = None) -> bool:
        """Simple microstructure filters using spread, top-of-book depth, and quote size."""
        if not market_data_point:
            return True

        instrument_type = market_data_point.get("instrument_type")
        spread_pct = market_data_point.get("spread_pct")
        bid = market_data_point.get("bid")
        ask = market_data_point.get("ask")
        bid_size = market_data_point.get("bid_size")
        ask_size = market_data_point.get("ask_size")

        if spread_pct is None and bid and ask:
            mid = (bid + ask) / 2
            if mid:
                spread_pct = ((ask - bid) / mid) * 100

        if spread_pct is not None and spread_pct > max_spread_pct:
            self.actions_logger.info(f"⏸️ Skipping trade: spread {spread_pct:.3f}% > cap {max_spread_pct:.3f}%")
            return False

        if min_quote_size and instrument_type == "STK":
            sizes = [s for s in (bid_size, ask_size) if s is not None]
            if sizes:
                min_size = min(sizes)
                if min_size < min_quote_size:
                    self.actions_logger.info(
                        f"⏸️ Skipping trade: quote size {min_size:.2f} < min shares {min_quote_size:.2f}"
                    )
                    return False

        if bid and ask and bid_size and ask_size:
            # Use the weaker side as liquidity floor
            min_notional = min(bid * bid_size, ask * ask_size)
            if min_notional < min_top_of_book_notional:
                self.actions_logger.info(
                    f"⏸️ Skipping trade: top-of-book notional ${min_notional:.2f} < ${min_top_of_book_notional:.2f} floor"
                )
                return False

        return True

    def stacking_block(self, action: str, open_plan_count: int, pending_data: dict, position_qty: float) -> bool:
        """
        Prevent stacking same-direction risk when we already have a position and
        pending orders/plans on the symbol.
        """
        if action != "BUY":
            return False
        if position_qty <= 0:
            return False
        pending_same_side = (pending_data.get("count_buy", 0) or 0) > 0
        has_plans = open_plan_count > 0
        return pending_same_side or has_plans

    # --- Action handlers ---
    async def handle_update_plan(self, plan_id: Optional[int], stop_price: float, target_price: float, reason: str, trace_id: Any):
        telemetry_record = {"status": "update_plan_missing_id"}
        if not plan_id:
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        try:
            self.db.update_trade_plan_prices(plan_id, stop_price=stop_price, target_price=target_price, reason=reason)
            telemetry_record["status"] = "plan_updated"
            self.log_execution_trace(trace_id, telemetry_record)
            self.emit_telemetry(telemetry_record)
            self.actions_logger.info(f"✏️ Plan {plan_id} updated: stop={stop_price}, target={target_price}")
        except Exception as exc:  # pragma: no cover - defensive
            telemetry_record["status"] = "plan_update_error"
            telemetry_record["error"] = str(exc)
            self.log_execution_trace(trace_id, telemetry_record)
            self.emit_telemetry(telemetry_record)
            self.logger.error(f"Plan update failed: {exc}")
        return telemetry_record

    async def handle_partial_close(
        self,
        plan_id: Optional[int],
        close_fraction: float,
        symbol: str,
        price: float | None,
        current_exposure: float,
        trace_id: Any,
    ):
        telemetry_record = {
            "status": "partial_close_invalid",
            "symbol": symbol,
            "plan_id": plan_id,
            "close_fraction": close_fraction,
        }
        if not plan_id or close_fraction <= 0 or close_fraction > 1:
            telemetry_record["error"] = "invalid plan_id or fraction"
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        try:
            open_plans = self.db.get_open_trade_plans_for_portfolio(self.portfolio_id)
            plan = next((p for p in open_plans if p.get("id") == plan_id), None)
        except Exception:
            plan = None
        if not plan:
            telemetry_record["status"] = "partial_close_missing_plan"
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        plan_side = plan.get("side", "BUY").upper()
        plan_size = plan.get("size", 0.0) or 0.0
        close_qty = max(0.0, plan_size * close_fraction)
        if close_qty <= 0:
            telemetry_record["status"] = "partial_close_zero_qty"
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        flatten_action = "SELL" if plan_side == "BUY" else "BUY"
        qty_for_risk = self.apply_order_value_buffer(close_qty, price or 0, symbol)
        risk_result = self.risk_manager.check_trade_allowed(symbol, flatten_action, qty_for_risk, price or 0)
        if not getattr(risk_result, "allowed", False):
            telemetry_record["status"] = "partial_close_blocked"
            telemetry_record["risk_reason"] = getattr(risk_result, "reason", None)
            if self.on_trade_rejected and telemetry_record.get("risk_reason"):
                self.on_trade_rejected(telemetry_record["risk_reason"])
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        self.actions_logger.info(f"🔻 Partial close {close_fraction*100:.0f}% of plan {plan_id}: {flatten_action} {qty_for_risk} {symbol}")
        try:
            prefer_maker = self.prefer_maker(symbol)
            order_result = await self.bot.place_order_async(symbol, flatten_action, qty_for_risk, prefer_maker=prefer_maker)
            liquidity_tag = order_result.get("liquidity", "taker") if order_result else "taker"
            fee = self.cost_tracker.calculate_trade_fee(symbol, qty_for_risk, price or 0, flatten_action, liquidity=liquidity_tag)
            realized = self.portfolio_tracker.update_holdings_and_realized(symbol, flatten_action, qty_for_risk, price or 0, fee)
            self.db.log_trade_for_portfolio(
                self.portfolio_id,
                symbol,
                flatten_action,
                qty_for_risk,
                price or 0,
                fee,
                f"Partial close plan {plan_id} ({close_fraction*100:.0f}%)",
                liquidity=order_result.get("liquidity") if order_result else "taker",
                realized_pnl=realized,
            )
            self.portfolio_tracker.apply_fill_to_portfolio_stats(order_result.get("order_id") if order_result else None, fee, realized)
            remaining_size = max(plan_size - close_qty, 0.0)
            try:
                partial_reason = f"Partial close {close_fraction*100:.0f}%"
                if remaining_size <= 1e-9:
                    self.db.update_trade_plan_status(
                        plan_id,
                        status="closed",
                        closed_at=datetime.now(timezone.utc).isoformat(),
                        reason=partial_reason,
                    )
                else:
                    self.db.update_trade_plan_size(plan_id, size=remaining_size, reason=partial_reason)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug(f"Could not update plan size after partial close: {exc}")
            telemetry_record["status"] = "partial_close_executed"
            telemetry_record["order_result"] = order_result
            self._merge_ib_order_metadata(telemetry_record, order_result)
        except Exception as exc:  # pragma: no cover - defensive
            telemetry_record["status"] = "partial_close_error"
            telemetry_record["error"] = str(exc)
            self.logger.error(f"Partial close failed: {exc}")
        self.log_execution_trace(trace_id, telemetry_record)
        self.emit_telemetry(telemetry_record)
        return telemetry_record

    async def handle_close_position(
        self,
        symbol: str,
        price: float | None,
        trace_id: Any,
    ):
        telemetry_record = {"status": "close_position_none", "symbol": symbol}
        qty = 0.0
        try:
            positions = self.db.get_positions_for_portfolio(self.portfolio_id)
            for pos in positions:
                if pos.get("symbol") == symbol:
                    qty = pos.get("quantity", 0.0) or 0.0
                    break
        except Exception:
            qty = 0.0
        if qty <= 0:
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        action = "SELL" if qty > 0 else "BUY"
        qty_abs = abs(qty)
        qty_buffered = self.apply_order_value_buffer(qty_abs, price or 0, symbol)
        if qty_buffered <= 0:
            telemetry_record["status"] = "close_position_zero_after_buffer"
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        risk_result = self.risk_manager.check_trade_allowed(symbol, action, qty_buffered, price or 0)
        if not getattr(risk_result, "allowed", False):
            telemetry_record["status"] = "close_position_blocked"
            telemetry_record["risk_reason"] = getattr(risk_result, "reason", None)
            self.emit_telemetry(telemetry_record)
            return telemetry_record
        try:
            prefer_maker = self.prefer_maker(symbol)
            order_result = await self.bot.place_order_async(symbol, action, qty_buffered, prefer_maker=prefer_maker)
            liquidity_tag = order_result.get("liquidity", "taker") if order_result else "taker"
            fee = self.cost_tracker.calculate_trade_fee(symbol, qty_buffered, price or 0, action, liquidity=liquidity_tag)
            realized = self.portfolio_tracker.update_holdings_and_realized(symbol, action, qty_buffered, price or 0, fee)
            self.db.log_trade_for_portfolio(
                self.portfolio_id,
                symbol,
                action,
                qty_buffered,
                price or 0,
                fee,
                f"Close position request ({qty})",
                liquidity=order_result.get("liquidity") if order_result else "taker",
                realized_pnl=realized,
            )
            self.portfolio_tracker.apply_fill_to_portfolio_stats(order_result.get("order_id") if order_result else None, fee, realized)
            telemetry_record["status"] = "close_position_executed"
            telemetry_record["order_result"] = order_result
            self._merge_ib_order_metadata(telemetry_record, order_result)
        except Exception as exc:  # pragma: no cover - defensive
            telemetry_record["status"] = "close_position_error"
            telemetry_record["error"] = str(exc)
            self.logger.error(f"Close position failed: {exc}")
        self.log_execution_trace(trace_id, telemetry_record)
        self.emit_telemetry(telemetry_record)
        return telemetry_record

    async def handle_pause_trading(self, duration_minutes: float):
        """Request a pause via health manager and emit telemetry record."""
        duration = duration_minutes or 5
        pause_seconds = max(0, duration * 60)
        pause_until = self.health_manager.request_pause(pause_seconds)
        telemetry_record = {"status": "paused", "pause_seconds": pause_seconds, "pause_until": pause_until}
        self.emit_telemetry(telemetry_record)
        self.actions_logger.info(f"⏸️ Trading paused for {pause_seconds/60:.1f} minutes by request")
        return telemetry_record
