import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from trader_bot.config import ACTIVE_EXCHANGE, CLIENT_ORDER_PREFIX, IB_BASE_CURRENCY
from trader_bot.database import TradingDatabase
from trader_bot.llm_tools import estimate_json_bytes
from trader_bot.utils import get_client_order_id

logger = logging.getLogger(__name__)

class TradingContext:
    """Provides rich trading context for LLM decision-making."""

    def __init__(self, db: TradingDatabase, portfolio_id: int, run_id: str | None = None):
        self.db = db
        self.portfolio_id = portfolio_id
        self.run_id = run_id
        self.position_baseline: Dict[str, float] = {}

    @staticmethod
    def _filter_our_orders(open_orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Only include orders with our client order prefix."""
        filtered = []
        for order in open_orders or []:
            client_oid = get_client_order_id(order)
            if client_oid and client_oid.startswith(CLIENT_ORDER_PREFIX):
                filtered.append(order)
        return filtered

    def set_position_baseline(self, baseline: Dict[str, float]):
        """Record baseline positions to hide sandbox airdrops from LLM context."""
        if not baseline:
            return
        for sym, qty in baseline.items():
            self.position_baseline[sym] = qty or 0.0

    @staticmethod
    def _parse_iso(ts_str: str) -> datetime | None:
        """Parse ISO timestamps while tolerating Z suffix."""
        if not ts_str:
            return None
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    def _get_portfolio_baseline(self) -> tuple[float | None, str | None]:
        """Return earliest equity snapshot (value, timestamp) for portfolio lifetime context."""
        baseline_equity = None
        baseline_ts = None
        try:
            snapshot = self.db.get_first_equity_snapshot_for_portfolio(self.portfolio_id)
            if snapshot:
                baseline_equity = snapshot.get("equity")
                baseline_ts = snapshot.get("timestamp")
        except Exception as exc:
            logger.debug(f"Could not load portfolio baseline snapshot: {exc}")
        if baseline_ts is None:
            try:
                portfolio_meta = self.db.get_portfolio(self.portfolio_id) or {}
                baseline_ts = portfolio_meta.get("created_at")
            except Exception as exc:
                logger.debug(f"Could not load portfolio metadata for baseline: {exc}")
        return baseline_equity, baseline_ts

    @staticmethod
    def _net_quantity_with_baseline(quantity: float, baseline_qty: float) -> float:
        """
        Remove baseline inventory so exposure reflects only positions opened/closed
        during the current portfolio run. Baseline defines a neutral band between 0 and baseline.
        """
        quantity = quantity or 0.0
        baseline_qty = baseline_qty or 0.0

        if baseline_qty >= 0:
            if quantity >= 0:
                return max(0.0, quantity - baseline_qty)
            return quantity

        if quantity <= 0:
            return min(0.0, quantity - baseline_qty)
        return quantity
    
    def get_context_summary(self, symbol: str, open_orders: List[Dict[str, Any]] = None) -> str:
        """
        Generate a compact, size-capped context summary for the LLM.
        Returns JSON (no pretty-print) to minimize prompt bytes.
        """
        max_positions = 5
        max_orders = 5
        max_trades = 5

        portfolio_stats = self.db.get_portfolio_stats(self.portfolio_id) if self.portfolio_id is not None else {}
        portfolio_meta = self.db.get_portfolio(self.portfolio_id) if self.portfolio_id is not None else {}
        baseline_equity, baseline_ts = self._get_portfolio_baseline()
        recent_trades = list(
            reversed(self.db.get_recent_trades_for_portfolio(self.portfolio_id, limit=50))
        ) if self.portfolio_id is not None else []

        # Calculate portfolio duration (hours, rounded) and win/loss counts
        duration_hours = 0.0
        created_at = portfolio_meta.get("created_at")
        start_ts = baseline_ts or created_at
        start_dt = self._parse_iso(start_ts) if start_ts else None
        if start_dt:
            duration_hours = (datetime.now(timezone.utc) - start_dt).total_seconds() / 3600

        wins = losses = 0
        open_position = None
        for trade in recent_trades:
            action = trade['action'].upper()
            qty = trade['quantity']
            price = trade['price']
            fee = trade.get('fee', 0.0) or 0.0
            if action == 'BUY':
                if open_position is None:
                    open_position = {'qty': qty, 'cost': qty * price, 'fees': fee}
                else:
                    open_position['qty'] += qty
                    open_position['cost'] += qty * price
                    open_position['fees'] += fee
            elif action == 'SELL' and open_position:
                closed_qty = min(open_position['qty'], qty)
                avg_cost = open_position['cost'] / open_position['qty'] if open_position['qty'] else 0.0
                pnl = (closed_qty * price) - (closed_qty * avg_cost) - fee - open_position.get('fees', 0.0)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                open_position['qty'] -= closed_qty
                open_position['cost'] -= closed_qty * avg_cost
                if open_position['qty'] <= 1e-9:
                    open_position = None

        total_closed = wins + losses
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

        # Recent market trend (quick delta)
        market_data = (
            self.db.get_recent_market_data_for_portfolio(self.portfolio_id, symbol, limit=20)
            if self.portfolio_id is not None
            else []
        )
        price_trend_pct = None
        if len(market_data) >= 2 and market_data[-1].get('price'):
            recent_price = market_data[0]['price']
            older_price = market_data[-1]['price']
            if older_price:
                price_trend_pct = ((recent_price - older_price) / older_price) * 100

        positions = self.db.get_positions_for_portfolio(self.portfolio_id) if self.portfolio_id is not None else []
        if open_orders is None:
            open_orders = self.db.get_open_orders_for_portfolio(self.portfolio_id) if self.portfolio_id is not None else []
        open_orders = self._filter_our_orders(open_orders)

        positions_summary = []
        total_exposure = 0.0
        baseline_map = self.position_baseline or {}
        for pos in positions[:max_positions]:
            sym = pos['symbol']
            qty = pos['quantity']
            avg_price = pos.get('avg_price') or 0

            current_price = avg_price
            recent_data = (
                self.db.get_recent_market_data_for_portfolio(self.portfolio_id, sym, limit=1)
                if self.portfolio_id is not None
                else []
            )
            if recent_data and recent_data[0].get('price'):
                current_price = recent_data[0]['price']

            if not avg_price or not current_price:
                continue

            baseline_qty = baseline_map.get(sym, 0.0)
            portfolio_qty = self._net_quantity_with_baseline(qty, baseline_qty)
            if abs(portfolio_qty) < 1e-9:
                continue

            cost_basis = portfolio_qty * avg_price
            current_value = portfolio_qty * current_price
            unrealized_pnl = current_value - cost_basis
            total_exposure += abs(current_value)

            positions_summary.append(
                {
                    "symbol": sym,
                    "qty": round(portfolio_qty, 8),
                    "avg": avg_price,
                    "px": current_price,
                    "unrealized": unrealized_pnl,
                }
            )

        orders_summary = []
        for order in (open_orders or [])[:max_orders]:
            orders_summary.append(
                {
                    "id": order.get('order_id'),
                    "side": (order.get('side') or '').upper(),
                    "symbol": order.get('symbol'),
                    "qty": order.get('amount'),
                    "remaining": order.get('remaining'),
                    "px": order.get('price'),
                    "status": order.get('status') or 'open',
                }
            )

        trades_summary = []
        if recent_trades:
            for trade in list(recent_trades)[-max_trades:]:
                trades_summary.append(
                    {
                        "ts": trade['timestamp'],
                        "action": trade['action'],
                        "symbol": trade['symbol'],
                        "qty": trade['quantity'],
                        "px": trade['price'],
                    }
                )

        venue_base_currency = portfolio_meta.get("base_currency") or (IB_BASE_CURRENCY if ACTIVE_EXCHANGE == "IB" else "USD")
        summary = {
            "portfolio": {
                "portfolio_id": self.portfolio_id,
                "created_at": created_at,
                "baseline_timestamp": baseline_ts,
                "hours": round(duration_hours, 1),
                "starting_balance": baseline_equity,
                "net_pnl": portfolio_stats.get('net_pnl'),
                "fees": portfolio_stats.get('total_fees'),
                "llm_cost": portfolio_stats.get('total_llm_cost'),
                "total_trades": portfolio_stats.get('total_trades'),
                "win_rate_pct": round(win_rate, 1),
                "wins": wins,
                "losses": losses,
                "exposure_notional": portfolio_stats.get("exposure_notional"),
            },
            "trend_pct": round(price_trend_pct, 2) if price_trend_pct is not None else None,
            "positions": positions_summary,
            "total_exposure": total_exposure,
            "open_orders": orders_summary,
            "recent_trades": trades_summary,
            "venue": {
                "exchange": ACTIVE_EXCHANGE,
                "base_currency": venue_base_currency,
                "instruments": "ASX equities/ETFs and FX (no crypto)" if ACTIVE_EXCHANGE == "IB" else "Crypto",
                "market_hours_note": "ASX ~10:00-16:00 AEST; FX ~24/5" if ACTIVE_EXCHANGE == "IB" else "24/7 crypto",
            },
            "run": {"id": self.run_id},
        }

        try:
            return json.dumps(summary, separators=(",", ":"))
        except Exception as exc:
            logger.debug(f"Failed to serialize context summary: {exc}")
            return ""

    def get_memory_snapshot(
        self,
        max_bytes: int = 1200,
        max_plans: int = 5,
        max_traces: int = 5,
    ) -> str:
        """
        Provide a small memory block with open plans and recent decisions/execution outcomes.
        Trims entries to stay within max_bytes; returns empty string on failure.
        """
        memory = {"open_plans": [], "recent_decisions": []}

        try:
            plans = self.db.get_open_trade_plans_for_portfolio(self.portfolio_id)[:max_plans]
            for plan in plans:
                memory["open_plans"].append(
                    {
                        "id": plan.get("id"),
                        "symbol": plan.get("symbol"),
                        "side": plan.get("side"),
                        "size": plan.get("size"),
                        "entry": plan.get("entry_price"),
                        "stop": plan.get("stop_price"),
                        "target": plan.get("target_price"),
                        "version": plan.get("version"),
                        "reason": plan.get("reason"),
                    }
                )
        except Exception as exc:
            logger.debug(f"Failed to fetch open plans for memory: {exc}")

        try:
            traces = self.db.get_recent_llm_traces_for_portfolio(self.portfolio_id, limit=max_traces)
            for trace in traces:
                decision = trace.get("decision_json")
                execution = trace.get("execution_result")
                try:
                    if decision:
                        decision = json.loads(decision)
                except Exception:
                    pass
                try:
                    if execution:
                        execution = json.loads(execution)
                except Exception:
                    pass
                memory["recent_decisions"].append(
                    {
                        "ts": trace.get("timestamp"),
                        "decision": decision,
                        "execution": execution,
                    }
                )
        except Exception as exc:
            logger.debug(f"Failed to fetch LLM traces for memory: {exc}")

        def _encode(mem: Dict[str, Any]) -> str:
            try:
                return json.dumps(mem, separators=(",", ":"))
            except Exception:
                return ""

        encoded = _encode(memory)
        if encoded and estimate_json_bytes(memory) <= max_bytes:
            return encoded

        # Trim decisions first, then plans until it fits or is empty
        while memory["recent_decisions"] and estimate_json_bytes(memory) > max_bytes:
            memory["recent_decisions"].pop()
        while memory["open_plans"] and estimate_json_bytes(memory) > max_bytes:
            memory["open_plans"].pop()

        encoded = _encode(memory)
        if encoded and estimate_json_bytes(memory) <= max_bytes:
            return encoded

        return ""
    
    def get_recent_performance(self) -> Dict[str, Any]:
        """Get recent performance metrics."""
        recent_trades = list(
            reversed(self.db.get_recent_trades_for_portfolio(self.portfolio_id, limit=50))
        )  # chronological
        
        if not recent_trades:
            return {
                'total_trades': 0,
                'avg_profit': 0,
                'win_rate': 0,
                'last_trade_profitable': None
            }
        
        profits = []
        open_position = None
        for trade in recent_trades:
            action = trade['action'].upper()
            qty = trade['quantity']
            price = trade['price']
            fee = trade.get('fee', 0.0) or 0.0
            if action == 'BUY':
                if open_position is None:
                    open_position = {'qty': qty, 'cost': qty * price, 'fees': fee}
                else:
                    open_position['qty'] += qty
                    open_position['cost'] += qty * price
                    open_position['fees'] += fee
            elif action == 'SELL' and open_position:
                closed_qty = min(open_position['qty'], qty)
                avg_cost = open_position['cost'] / open_position['qty'] if open_position['qty'] else 0.0
                pnl = (closed_qty * price) - (closed_qty * avg_cost) - fee - open_position.get('fees', 0.0)
                profits.append(pnl)
                open_position['qty'] -= closed_qty
                open_position['cost'] -= closed_qty * avg_cost
                if open_position['qty'] <= 1e-9:
                    open_position = None

        wins = sum(1 for p in profits if p > 0)
        return {
            'total_trades': len(recent_trades),
            'avg_profit': sum(profits) / len(profits) if profits else 0,
            'win_rate': (wins / len(profits) * 100) if profits else 0,
            'last_trade_profitable': profits[-1] > 0 if profits else None
        }
