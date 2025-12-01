import logging
from typing import Dict, Optional

from trader_bot.database import TradingDatabase

logger = logging.getLogger(__name__)


class MetricsDrift:
    def __init__(self, portfolio_id: int, db: Optional[TradingDatabase] = None):
        self.portfolio_id = portfolio_id
        self.db = db or TradingDatabase()

    def check_drift(self, threshold_pct: float = 1.0) -> Dict[str, float | bool]:
        """Compare persisted portfolio stats vs latest equity snapshot."""
        portfolio = self.db.get_portfolio(self.portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio {self.portfolio_id} not found")

        stats = self.db.get_portfolio_stats(self.portfolio_id) or {}
        baseline = self.db.get_first_equity_snapshot_for_portfolio(self.portfolio_id)
        if baseline is None:
            raise ValueError("No equity snapshots available")

        starting_balance = baseline.get("equity") or 0.0
        baseline_ts = baseline.get("timestamp")
        net_pnl = stats.get("net_pnl") or 0.0
        reference = starting_balance + net_pnl
        latest_equity = self.db.get_latest_equity_for_portfolio(self.portfolio_id)

        drift = (latest_equity or 0.0) - reference
        ref_denom = max(1.0, abs(reference))
        drift_pct = (drift / ref_denom) * 100
        exceeded = abs(drift_pct) >= threshold_pct

        result = {
            "starting_balance": starting_balance,
            "net_pnl": net_pnl,
            "reference_equity": reference,
            "latest_equity": latest_equity,
            "drift": drift,
            "drift_pct": drift_pct,
            "exceeded": exceeded,
            "threshold_pct": threshold_pct,
            "baseline_timestamp": baseline_ts,
        }

        try:
            self.db.log_llm_call_for_portfolio(self.portfolio_id, 0, 0, 0.0, f"metrics_drift:{result}")
        except Exception:
            logger.debug("Could not log metrics drift")
        return result
