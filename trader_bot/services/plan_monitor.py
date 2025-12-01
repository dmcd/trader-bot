import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from trader_bot.config import MAX_TOTAL_EXPOSURE

bot_actions_logger = logging.getLogger("bot_actions")
logger = logging.getLogger(__name__)


@dataclass
class PlanMonitorConfig:
    max_plan_age_minutes: int | None
    day_end_flatten_hour_utc: int | None
    trail_to_breakeven_pct: float


class PlanMonitor:
    """Encapsulates trade plan monitoring and closing logic."""

    def __init__(
        self,
        db,
        bot,
        cost_tracker,
        risk_manager,
        prefer_maker: Callable[[str], bool],
        holdings_updater: Callable[[str, str, float, float, float], float],
        portfolio_stats_applier: Callable[[str | None, float, float], None],
        max_total_exposure: float = MAX_TOTAL_EXPOSURE,
        portfolio_id: int | None = None,
    ):
        self.db = db
        self.bot = bot
        self.cost_tracker = cost_tracker
        self.risk_manager = risk_manager
        self.prefer_maker = prefer_maker
        self.holdings_updater = holdings_updater
        self.portfolio_stats_applier = portfolio_stats_applier
        self.max_total_exposure = max_total_exposure
        self.portfolio_id = portfolio_id

    def refresh_bindings(
        self,
        *,
        bot=None,
        db=None,
        cost_tracker=None,
        risk_manager=None,
        prefer_maker: Callable[[str], bool] | None = None,
        holdings_updater: Callable[[str, str, float, float, float], float] | None = None,
        portfolio_stats_applier: Callable[[str | None, float, float], None] | None = None,
        portfolio_id: int | None = None,
    ) -> None:
        """Allow tests/runners to swap dependencies without rebuilding the service."""
        if bot is not None:
            self.bot = bot
        if db is not None:
            self.db = db
        if cost_tracker is not None:
            self.cost_tracker = cost_tracker
        if risk_manager is not None:
            self.risk_manager = risk_manager
        if prefer_maker is not None:
            self.prefer_maker = prefer_maker
        if holdings_updater is not None:
            self.holdings_updater = holdings_updater
        if portfolio_stats_applier is not None:
            self.portfolio_stats_applier = portfolio_stats_applier
        if portfolio_id is not None:
            self.portfolio_id = portfolio_id

    async def monitor(
        self,
        price_lookup: dict,
        open_orders: Iterable[dict],
        config: PlanMonitorConfig,
        *,
        now: datetime | None = None,
        portfolio_id: int | None = None,
    ) -> None:
        """
        Monitor open trade plans for stop/target hits and enforce max age/day-end flattening per symbol.
        price_lookup: symbol -> latest price
        open_orders: list of open orders from exchange
        """
        if portfolio_id is not None:
            self.portfolio_id = portfolio_id
        if not self.portfolio_id:
            return
        try:
            open_plans = self.db.get_open_trade_plans_for_portfolio(self.portfolio_id)
            now = now or datetime.now(timezone.utc)
            now_iso = now.isoformat()
            day_end_cutoff = None
            if config.day_end_flatten_hour_utc is not None:
                day_end_cutoff = now.replace(
                    hour=config.day_end_flatten_hour_utc,
                    minute=0,
                    second=0,
                    microsecond=0,
                )

            open_orders_by_symbol = {}
            for o in open_orders or []:
                sym = o.get("symbol")
                if not sym:
                    continue
                open_orders_by_symbol.setdefault(sym, []).append(o)

            for plan in open_plans:
                plan_id = plan["id"]
                side = plan["side"].upper()
                regime_flags = plan.get("regime_flags") or {}
                stop = plan.get("stop_price")
                target = plan.get("target_price")
                size = plan.get("size") or 0.0
                symbol = plan.get("symbol")
                price_now = (price_lookup or {}).get(symbol)
                entry = plan.get("entry_price") or price_now
                version = plan.get("version") or 1
                if not price_now or size <= 0:
                    continue

                should_close = False
                reason = None

                # Close stale plans with no position and no open orders for the symbol
                pos_qty = (self.risk_manager.positions or {}).get(symbol, {}).get("quantity", 0.0) or 0.0
                has_open_orders = bool(open_orders_by_symbol.get(symbol))
                if abs(pos_qty) < 1e-9 and not has_open_orders:
                    should_close = True
                    reason = "Plan closed: position flat and no open orders"

                # Cancel if exposure headroom exhausted
                try:
                    exposure_now = self.risk_manager.get_total_exposure()
                    if exposure_now >= self.max_total_exposure * 0.98:
                        should_close = True
                        reason = "Cancelled plan: exposure headroom exhausted"
                except Exception:
                    pass

                opened_at = plan.get("opened_at")
                if opened_at and config.max_plan_age_minutes:
                    try:
                        opened_dt = datetime.fromisoformat(opened_at)
                        age_min = (now - opened_dt).total_seconds() / 60.0
                        if age_min >= config.max_plan_age_minutes:
                            should_close = True
                            reason = f"Plan age exceeded {config.max_plan_age_minutes} min"
                    except Exception:
                        pass

                if not should_close and day_end_cutoff and opened_at:
                    try:
                        opened_dt = datetime.fromisoformat(opened_at)
                        if opened_dt < day_end_cutoff:
                            should_close = True
                            reason = "Day-end flatten"
                    except Exception:
                        pass

                try:
                    vol_flag = (plan.get("volatility") or regime_flags.get("volatility") or "").lower()
                except Exception:
                    vol_flag = ""
                trail_pct = config.trail_to_breakeven_pct
                if vol_flag:
                    if "low" in vol_flag:
                        trail_pct *= 1.5
                    elif "high" in vol_flag:
                        trail_pct *= 0.7

                if side == "BUY":
                    # Trail stop to entry after move in favor
                    if stop and price_now >= entry * (1 + trail_pct) and stop < entry:
                        try:
                            self.db.update_trade_plan_prices(
                                plan_id, stop_price=entry, reason="Trailed stop to breakeven"
                            )
                            bot_actions_logger.info(
                                f"↩️ Trailed stop to breakeven for plan {plan_id} (v{version}→v{version+1})"
                            )
                            stop = entry
                        except Exception as e:
                            logger.debug(f"Could not trail stop for plan {plan_id}: {e}")
                    # Apply volatility-aware trailing: widen on low vol, tighten on high vol
                    if stop and price_now <= stop:
                        should_close = True
                        reason = f"Stop hit at ${price_now:,.2f}"
                    elif target and price_now >= target:
                        should_close = True
                        reason = f"Target hit at ${price_now:,.2f}"
                else:  # SELL plan (short)
                    if stop and price_now <= entry * (1 - trail_pct) and stop > entry:
                        try:
                            self.db.update_trade_plan_prices(
                                plan_id, stop_price=entry, reason="Trailed stop to breakeven"
                            )
                            bot_actions_logger.info(
                                f"↩️ Trailed stop to breakeven for plan {plan_id} (v{version}→v{version+1})"
                            )
                            stop = entry
                        except Exception as e:
                            logger.debug(f"Could not trail stop for plan {plan_id}: {e}")
                    if stop and price_now >= stop:
                        should_close = True
                        reason = f"Stop hit at ${price_now:,.2f}"
                    elif target and price_now <= target:
                        should_close = True
                        reason = f"Target hit at ${price_now:,.2f}"

                if should_close:
                    try:
                        action = "SELL" if side == "BUY" else "BUY"
                        bot_actions_logger.info(f"🏁 Closing plan {plan_id}: {reason}")
                        order_result = await self.bot.place_order_async(
                            plan["symbol"], action, size, prefer_maker=False, force_market=True
                        )
                        liquidity_tag = order_result.get("liquidity", "taker") if order_result else "taker"
                        fee = self.cost_tracker.calculate_trade_fee(
                            plan["symbol"], size, price_now, action, liquidity=liquidity_tag
                        )
                        realized = self.holdings_updater(plan["symbol"], action, size, price_now, fee)
                        self.db.log_trade_for_portfolio(
                            self.portfolio_id,
                            plan["symbol"],
                            action,
                            size,
                            price_now,
                            fee,
                            reason,
                            liquidity=order_result.get("liquidity") if order_result else "taker",
                            realized_pnl=realized,
                        )
                        self.portfolio_stats_applier(
                            order_result.get("order_id") if order_result else None,
                            fee,
                            realized,
                        )
                        self.db.update_trade_plan_status(plan_id, status="closed", closed_at=now_iso, reason=reason)
                    except Exception as e:
                        logger.error(f"Failed to close plan {plan_id}: {e}")
        except Exception as e:
            logger.warning(f"Monitor trade plans failed: {e}")
