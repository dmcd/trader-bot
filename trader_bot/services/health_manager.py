import asyncio
import logging
import time
from typing import Any, Callable, Optional


class HealthCircuitManager:
    """
    Tracks exchange/tool health, pause windows, and market data freshness.
    Extracted from StrategyRunner to make circuit logic testable in isolation.
    """

    def __init__(
        self,
        record_health_state: Optional[Callable[[str, str, Optional[dict]], None]] = None,
        exchange_error_threshold: int = 3,
        exchange_pause_seconds: int = 30,
        tool_error_threshold: int = 3,
        tool_pause_seconds: int = 30,
        ticker_max_age_ms: float = 0,
        ticker_max_latency_ms: float = 0,
        monotonic: Optional[Callable[[], float]] = None,
        actions_logger: Optional[logging.Logger] = None,
        logger: Optional[logging.Logger] = None,
        reconnect_cooldown_seconds: int = 30,
    ):
        self.record_health_state = record_health_state or (lambda *_: None)
        self.exchange_error_threshold = exchange_error_threshold
        self.exchange_pause_seconds = exchange_pause_seconds
        self.tool_error_threshold = tool_error_threshold
        self.tool_pause_seconds = tool_pause_seconds
        self.ticker_max_age_ms = ticker_max_age_ms
        self.ticker_max_latency_ms = ticker_max_latency_ms
        self.monotonic = monotonic or time.monotonic
        self.actions_logger = actions_logger or logging.getLogger(__name__)
        self.logger = logger or logging.getLogger(__name__)
        self.reconnect_cooldown_seconds = reconnect_cooldown_seconds

        self.exchange_error_streak = 0
        self.tool_error_streak = 0
        self.exchange_health = "ok"
        self.tool_health = "ok"
        self.pause_until: Optional[float] = None  # monotonic seconds
        self._last_reconnect: float = -float(self.reconnect_cooldown_seconds)

    def request_pause(self, seconds: float) -> float:
        """Request a trading pause for the given duration in seconds."""
        now = self.monotonic()
        pause_ts = now + max(0.0, seconds)
        self.pause_until = max(self.pause_until or 0.0, pause_ts)
        return self.pause_until

    def should_pause(self, now: Optional[float] = None) -> bool:
        """Return True when still inside a pause window."""
        if not self.pause_until:
            return False
        now = self.monotonic() if now is None else now
        return now < self.pause_until

    def pause_remaining(self, now: Optional[float] = None) -> float:
        """Return seconds remaining in current pause window."""
        if not self.should_pause(now):
            return 0.0
        now = self.monotonic() if now is None else now
        return max(0.0, (self.pause_until or 0.0) - now)

    def record_exchange_failure(self, context: str, error: Exception | str | None = None) -> None:
        """Track consecutive exchange failures and trigger auto-pause."""
        self.exchange_error_streak += 1
        detail = {
            "context": context,
            "error": str(error) if error is not None else None,
            "streak": self.exchange_error_streak,
        }
        if self.exchange_health != "degraded":
            self.exchange_health = "degraded"
            self.record_health_state("exchange_circuit", "degraded", detail)
        if self.exchange_error_streak >= self.exchange_error_threshold:
            pause_until = self.request_pause(self.exchange_pause_seconds)
            detail.update(
                {
                    "pause_seconds": self.exchange_pause_seconds,
                    "tripped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "pause_until": pause_until,
                }
            )
            self.record_health_state("exchange_circuit", "tripped", detail)
            self.actions_logger.info(
                f"🛑 Exchange circuit breaker tripped after {self.exchange_error_threshold} failures; "
                f"pausing for {self.exchange_pause_seconds}s"
            )
            self.exchange_error_streak = 0
            self.exchange_health = "tripped"

    def reset_exchange_errors(self) -> None:
        """Mark exchange channel healthy and clear streak."""
        if self.exchange_error_streak > 0 or self.exchange_health != "ok":
            self.exchange_error_streak = 0
            self.exchange_health = "ok"
            self.record_health_state("exchange_circuit", "ok", {"note": "recovered"})

    def record_tool_failure(
        self, request: Any = None, error: Exception | str | None = None, context: str | None = None
    ) -> None:
        """Track consecutive tool failures and trigger auto-pause."""
        self.tool_error_streak += 1
        detail = {
            "request": getattr(request, "id", None) if request is not None else None,
            "tool": getattr(request, "tool", None).value if getattr(request, "tool", None) else None,
            "context": context,
            "error": str(error) if error is not None else None,
            "streak": self.tool_error_streak,
        }
        if self.tool_health != "degraded":
            self.tool_health = "degraded"
            self.record_health_state("tool_circuit", "degraded", detail)
        if self.tool_error_streak >= self.tool_error_threshold:
            pause_until = self.request_pause(self.tool_pause_seconds)
            detail.update(
                {
                    "pause_seconds": self.tool_pause_seconds,
                    "tripped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "pause_until": pause_until,
                }
            )
            self.record_health_state("tool_circuit", "tripped", detail)
            self.actions_logger.info(
                f"🛑 Tool circuit breaker tripped after {self.tool_error_threshold} failures; "
                f"pausing for {self.tool_pause_seconds}s"
            )
            self.tool_error_streak = 0
            self.tool_health = "tripped"

    def record_tool_success(self) -> None:
        """Mark tool path healthy and clear streak."""
        if self.tool_error_streak > 0 or self.tool_health != "ok":
            self.tool_error_streak = 0
            self.tool_health = "ok"
            self.record_health_state("tool_circuit", "ok", {"note": "recovered"})

    def is_stale_market_data(self, data: dict) -> tuple[bool, dict]:
        """Return (stale, detail) using latency and age thresholds."""
        if not data:
            return True, {"reason": "empty"}
        detail: dict[str, Any] = {}
        now_mono = self.monotonic()
        latency_ms = data.get("_latency_ms")
        if latency_ms is not None:
            detail["latency_ms"] = latency_ms
        fetched_mono = data.get("_fetched_monotonic")
        age_ms = None
        if fetched_mono is not None:
            age_ms = max(0.0, (now_mono - fetched_mono) * 1000)
        ts_field = data.get("timestamp") or data.get("ts")
        if ts_field is not None:
            ts_ms = ts_field if ts_field > 1e12 else ts_field * 1000
            wall_age = max(0.0, (time.time() * 1000) - ts_ms)
            age_ms = age_ms if age_ms is not None else wall_age
            detail["data_age_ms"] = wall_age
        if age_ms is None and fetched_mono is not None:
            age_ms = max(0.0, (now_mono - fetched_mono) * 1000)
        if age_ms is not None:
            detail["age_ms"] = age_ms
        if latency_ms is not None and latency_ms > self.ticker_max_latency_ms:
            detail["reason"] = "latency"
            return True, detail
        if age_ms is not None and age_ms > self.ticker_max_age_ms:
            detail["reason"] = "age"
            return True, detail
        return False, detail

    async def maybe_reconnect(self, bot: Any) -> bool:
        """Reconnect the broker client with a cooldown to avoid thrash."""
        now = self.monotonic()
        if now - self._last_reconnect < self.reconnect_cooldown_seconds:
            return False
        try:
            await bot.connect_async()
            self._last_reconnect = now
            self.logger.info("Reconnected to broker")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(f"Reconnect failed: {exc}")
            return False
