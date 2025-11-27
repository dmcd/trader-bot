import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from database import TradingDatabase
from llm_tools import estimate_json_bytes

logger = logging.getLogger(__name__)

class TradingContext:
    """Provides rich trading context for LLM decision-making."""

    def __init__(self, db: TradingDatabase, session_id: int):
        self.db = db
        self.session_id = session_id
    
    def get_context_summary(self, symbol: str, open_orders: List[Dict[str, Any]] = None) -> str:
        """
        Generate a compact, size-capped context summary for the LLM.
        Returns JSON (no pretty-print) to minimize prompt bytes.
        """
        max_positions = 5
        max_orders = 5
        max_trades = 5

        session = self.db.get_session_stats(self.session_id)
        recent_trades = list(reversed(self.db.get_recent_trades(self.session_id, limit=50)))  # chronological

        # Calculate session duration (hours, rounded) and win/loss counts
        session_start = datetime.fromisoformat(session['created_at'])
        duration_hours = (datetime.now() - session_start).total_seconds() / 3600

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
        market_data = self.db.get_recent_market_data(self.session_id, symbol, limit=20)
        price_trend_pct = None
        if len(market_data) >= 2 and market_data[-1].get('price'):
            recent_price = market_data[0]['price']
            older_price = market_data[-1]['price']
            if older_price:
                price_trend_pct = ((recent_price - older_price) / older_price) * 100

        positions = self.db.get_positions(self.session_id)
        if open_orders is None:
            open_orders = self.db.get_open_orders(self.session_id)

        positions_summary = []
        total_exposure = 0.0
        for pos in positions[:max_positions]:
            sym = pos['symbol']
            qty = pos['quantity']
            avg_price = pos.get('avg_price') or 0

            current_price = avg_price
            recent_data = self.db.get_recent_market_data(self.session_id, sym, limit=1)
            if recent_data and recent_data[0].get('price'):
                current_price = recent_data[0]['price']

            if not avg_price or not current_price:
                continue

            cost_basis = qty * avg_price
            current_value = qty * current_price
            unrealized_pnl = current_value - cost_basis
            total_exposure += current_value

            positions_summary.append(
                {
                    "symbol": sym,
                    "qty": round(qty, 8),
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

        summary = {
            "session": {
                "date": session.get('date'),
                "hours": round(duration_hours, 1),
                "starting_balance": session.get('starting_balance'),
                "net_pnl": session.get('net_pnl'),
                "fees": session.get('total_fees'),
                "llm_cost": session.get('total_llm_cost'),
                "total_trades": session.get('total_trades'),
                "win_rate_pct": round(win_rate, 1),
                "wins": wins,
                "losses": losses,
            },
            "trend_pct": round(price_trend_pct, 2) if price_trend_pct is not None else None,
            "positions": positions_summary,
            "total_exposure": total_exposure,
            "open_orders": orders_summary,
            "recent_trades": trades_summary,
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
            plans = self.db.get_open_trade_plans(self.session_id)[:max_plans]
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
            traces = self.db.get_recent_llm_traces(self.session_id, limit=max_traces)
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
        recent_trades = list(reversed(self.db.get_recent_trades(self.session_id, limit=50)))  # chronological
        
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
