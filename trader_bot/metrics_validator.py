import logging
from typing import Dict, Optional

from trader_bot.database import TradingDatabase

logger = logging.getLogger(__name__)


class MetricsDrift:
    def __init__(self, session_id: int, db: Optional[TradingDatabase] = None):
        self.session_id = session_id
        self.db = db or TradingDatabase()

    def check_drift(self, threshold_pct: float = 1.0) -> Dict[str, float | bool]:
        """Compare persisted session stats vs latest equity snapshot.

        Returns a dict with computed drift percent and a flag indicating if drift exceeds threshold.
        Drift is calculated as (equity - (starting_balance + net_pnl)) / max(1, abs(reference)).
        """
        session = self.db.get_session(self.session_id)
        if not session:
            raise ValueError(f"Session {self.session_id} not found")

        starting_balance = session.get("starting_balance") or 0.0
        net_pnl = session.get("net_pnl") or 0.0
        reference = starting_balance + net_pnl
        latest_equity = self.db.get_latest_equity(self.session_id)
        if latest_equity is None:
            raise ValueError("No equity snapshots available")

        drift = latest_equity - reference
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
        }

        try:
            self.db.log_llm_call(self.session_id, 0, 0, 0.0, f"metrics_drift:{result}")
        except Exception:
            logger.debug("Could not log metrics drift")
        return result
