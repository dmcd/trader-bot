import logging
from typing import Any, Callable, Optional


class MarketDataService:
    """
    Handles OHLCV capture with spacing and retention rules, plus timeframe parsing.
    Extracted from StrategyRunner for isolated testing.
    """

    def __init__(
        self,
        db: Any,
        bot: Any,
        session_id: Optional[int] = None,
        portfolio_id: Optional[int] = None,
        monotonic: Optional[Callable[[], float]] = None,
        logger: Optional[logging.Logger] = None,
        ohlcv_min_capture_spacing_seconds: int = 0,
        ohlcv_retention_limit: Optional[int] = None,
    ):
        self.db = db
        self.bot = bot
        self.session_id = session_id
        self.portfolio_id = portfolio_id
        self.monotonic = monotonic
        self.logger = logger or logging.getLogger(__name__)
        self.ohlcv_min_capture_spacing_seconds = ohlcv_min_capture_spacing_seconds
        self.ohlcv_retention_limit = ohlcv_retention_limit
        self._last_ohlcv_capture: dict[tuple[str, str], float] = {}

    def set_session(self, session_id: int, portfolio_id: Optional[int] = None) -> None:
        self.session_id = session_id
        if portfolio_id is not None:
            self.portfolio_id = portfolio_id

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """Convert simple timeframe strings like '1m' or '1h' to seconds."""
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            if unit == 'm':
                return value * 60
            if unit == 'h':
                return value * 3600
            if unit == 'd':
                return value * 86400
        except Exception:
            return 0
        return 0

    async def capture_ohlcv(self, symbol: str, timeframes: list[str] | None = None):
        """Fetch multi-timeframe OHLCV for the active symbol and persist."""
        if not hasattr(self.bot, "fetch_ohlcv"):
            return
        if timeframes is None:
            timeframes = ['1m', '5m', '1h', '1d']
        now = self.monotonic() if self.monotonic else 0.0
        for tf in timeframes:
            try:
                tf_seconds = self.timeframe_to_seconds(tf)
                min_spacing = max(self.ohlcv_min_capture_spacing_seconds, tf_seconds or 0)
                last_key = (symbol, tf)
                last_capture = self._last_ohlcv_capture.get(last_key)
                if last_capture is not None and (now - last_capture) < min_spacing:
                    continue

                bars = await self.bot.fetch_ohlcv(symbol, timeframe=tf, limit=50)
                if self.session_id is not None:
                    self.db.log_ohlcv_batch(self.session_id, symbol, tf, bars, portfolio_id=self.portfolio_id)
                    if self.ohlcv_retention_limit:
                        try:
                            self.db.prune_ohlcv(self.session_id, symbol, tf, self.ohlcv_retention_limit)
                        except Exception as exc:
                            self.logger.debug(f"OHLCV prune failed for {symbol} {tf}: {exc}")
                self._last_ohlcv_capture[last_key] = now
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug(f"OHLCV fetch failed for {symbol} {tf}: {exc}")
