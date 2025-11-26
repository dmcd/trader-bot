import asyncio
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from config import (
    TOOL_CACHE_TTL_SECONDS,
    TOOL_MAX_BARS,
    TOOL_MAX_DEPTH,
    TOOL_MAX_JSON_BYTES,
    TOOL_MAX_TRADES,
)
from llm_tools import (
    MarketDataParams,
    OrderBookParams,
    RecentTradesParams,
    ToolName,
    ToolRequest,
    ToolResponse,
    clamp_payload_size,
    normalize_candles,
    normalize_order_book,
    normalize_trades,
)

logger = logging.getLogger(__name__)


class DataFetchCoordinator:
    """
    Coordinates LLM tool requests for market data, order books, and trades.

    An exchange-like object with async methods fetch_ohlcv, fetch_order_book,
    and fetch_trades should be supplied (ccxt async works).
    """

    def __init__(
        self,
        exchange: Any,
        max_bars: int = TOOL_MAX_BARS,
        max_depth: int = TOOL_MAX_DEPTH,
        max_trades: int = TOOL_MAX_TRADES,
        cache_ttl_seconds: int = TOOL_CACHE_TTL_SECONDS,
        max_json_bytes: int = TOOL_MAX_JSON_BYTES,
    ):
        self.exchange = exchange
        self.max_bars = max_bars
        self.max_depth = max_depth
        self.max_trades = max_trades
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_json_bytes = max_json_bytes
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self.cache_ttl_seconds:
            return None
        return entry["data"]

    def _cache_set(self, key: str, data: Dict[str, Any]) -> None:
        self._cache[key] = {"ts": time.time(), "data": data}

    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        if not hasattr(self.exchange, "fetch_ohlcv"):
            raise RuntimeError("Exchange does not support fetch_ohlcv")
        return await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    async def _fetch_order_book(self, symbol: str, depth: int) -> Dict[str, Any]:
        if not hasattr(self.exchange, "fetch_order_book"):
            raise RuntimeError("Exchange does not support fetch_order_book")
        return await self.exchange.fetch_order_book(symbol, limit=depth)

    async def _fetch_recent_trades(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        if not hasattr(self.exchange, "fetch_trades"):
            raise RuntimeError("Exchange does not support fetch_trades")
        return await self.exchange.fetch_trades(symbol, limit=limit)

    @staticmethod
    def _timeframe_ms(timeframe: str) -> int:
        """Convert ccxt-style timeframe to milliseconds."""
        units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        if not timeframe or timeframe[-1] not in units:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        try:
            value = int(timeframe[:-1])
        except ValueError as exc:
            raise ValueError(f"Unsupported timeframe: {timeframe}") from exc
        return value * units[timeframe[-1]]

    def _build_candles_from_trades(
        self, trades: List[Dict[str, Any]], timeframe: str, limit: int
    ) -> List[List[Any]]:
        """Fallback candle builder from trades (IB-style)."""
        if not trades:
            return []
        frame_ms = self._timeframe_ms(timeframe)
        buckets: Dict[int, List[Tuple[int, float, float]]] = {}
        for trade in trades:
            ts = trade.get("timestamp") or trade.get("ts")
            price = trade.get("price")
            size = trade.get("amount") or trade.get("size")
            if ts is None or price is None or size is None:
                continue
            bucket = int(math.floor(ts / frame_ms) * frame_ms)
            buckets.setdefault(bucket, []).append((int(ts), float(price), float(size)))

        candles: List[List[Any]] = []
        for bucket_ts, points in buckets.items():
            points_sorted = sorted(points, key=lambda p: p[0])
            prices = [p[1] for p in points_sorted]
            sizes = [p[2] for p in points_sorted]
            o = prices[0]
            h = max(prices)
            l = min(prices)
            c = prices[-1]
            v = sum(sizes)
            candles.append([bucket_ts, o, h, l, c, v])

        candles_sorted = sorted(candles, key=lambda c: c[0], reverse=False)
        return candles_sorted[-limit:]

    async def fetch_market_data(self, params: MarketDataParams) -> Dict[str, Any]:
        limit = min(params.limit, self.max_bars)
        results: Dict[str, Any] = {"symbol": params.symbol, "timeframes": {}}
        pending: List[tuple[str, asyncio.Task]] = []
        for tf in params.timeframes:
            cache_key = f"ohlcv:{params.symbol}:{tf}:{limit}"
            cached = self._cache_get(cache_key)
            if cached:
                results["timeframes"][tf] = cached
                continue
            pending.append(
                (
                    tf,
                    asyncio.create_task(
                        self._fetch_ohlcv(params.symbol, tf, limit)
                    ),
                )
            )

        for tf, task in pending:
            try:
                data = await task
                if not data:
                    raise ValueError("Empty OHLCV response")
                shaped = normalize_candles(data, tf, params.limit, self.max_bars)
                self._cache_set(f"ohlcv:{params.symbol}:{tf}:{limit}", shaped)
                results["timeframes"][tf] = shaped
            except Exception as exc:
                logger.warning(f"OHLCV fetch failed for {params.symbol} {tf}: {exc}")
                # Fallback: build candles from trades when available
                try:
                    trades_raw = await self._fetch_recent_trades(params.symbol, limit)
                    fallback = self._build_candles_from_trades(trades_raw, tf, limit)
                    shaped = normalize_candles(fallback, tf, params.limit, self.max_bars)
                    self._cache_set(f"ohlcv:{params.symbol}:{tf}:{limit}", shaped)
                    results["timeframes"][tf] = shaped
                except Exception as inner_exc:
                    logger.warning(f"Fallback trades->candles failed for {params.symbol} {tf}: {inner_exc}")
                    results["timeframes"][tf] = {"error": str(exc), "timeframe": tf, "candles": []}
        return results

    async def fetch_order_book(self, params: OrderBookParams) -> Dict[str, Any]:
        depth = min(params.depth, self.max_depth)
        cache_key = f"order_book:{params.symbol}:{depth}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        raw = await self._fetch_order_book(params.symbol, depth)
        shaped = normalize_order_book(raw, params.depth, self.max_depth)
        self._cache_set(cache_key, shaped)
        return shaped

    async def fetch_recent_trades(self, params: RecentTradesParams) -> Dict[str, Any]:
        limit = min(params.limit, self.max_trades)
        cache_key = f"trades:{params.symbol}:{limit}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        raw = await self._fetch_recent_trades(params.symbol, limit)
        shaped = normalize_trades(raw, params.limit, self.max_trades)
        self._cache_set(cache_key, shaped)
        return shaped

    async def handle_requests(self, requests: List[ToolRequest]) -> List[ToolResponse]:
        responses: List[ToolResponse] = []
        for req in requests:
            try:
                if req.tool == ToolName.GET_MARKET_DATA:
                    data = await self.fetch_market_data(req.params)  # type: ignore[arg-type]
                elif req.tool == ToolName.GET_ORDER_BOOK:
                    data = await self.fetch_order_book(req.params)  # type: ignore[arg-type]
                elif req.tool == ToolName.GET_RECENT_TRADES:
                    data = await self.fetch_recent_trades(req.params)  # type: ignore[arg-type]
                else:
                    raise ValueError(f"Unsupported tool: {req.tool}")
                data = clamp_payload_size(data, self.max_json_bytes)
                responses.append(ToolResponse(id=req.id, tool=req.tool, data=data))
            except Exception as exc:
                logger.error(f"Tool request {req.id} failed: {exc}")
                responses.append(ToolResponse(id=req.id, tool=req.tool, data={}, error=str(exc)))
        return responses
