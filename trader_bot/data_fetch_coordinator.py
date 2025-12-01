import asyncio
import copy
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trader_bot.config import (
    ALLOWED_SYMBOLS,
    TOOL_CACHE_TTL_SECONDS,
    TOOL_MAX_BARS,
    TOOL_MAX_DEPTH,
    TOOL_MAX_JSON_BYTES,
    TOOL_MAX_TRADES,
    TOOL_RATE_LIMIT_MARKET_DATA,
    TOOL_RATE_LIMIT_ORDER_BOOK,
    TOOL_RATE_LIMIT_RECENT_TRADES,
    TOOL_RATE_LIMIT_WINDOW_SECONDS,
)
from trader_bot.llm_tools import (
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
from trader_bot.symbols import normalize_symbol, normalize_symbols

logger = logging.getLogger(__name__)
telemetry_logger = logging.getLogger("telemetry")


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
        allowed_symbols: Optional[List[str]] = None,
        rate_limits: Optional[Dict[ToolName, int]] = None,
        rate_limit_window_seconds: int = TOOL_RATE_LIMIT_WINDOW_SECONDS,
        dedup_window_seconds: Optional[int] = None,
        error_callback: Any = None,
        success_callback: Any = None,
        portfolio_id: Optional[int] = None,
        run_id: Optional[str] = None,
    ):
        self.exchange = exchange
        self.max_bars = max_bars
        self.max_depth = max_depth
        self.max_trades = max_trades
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_json_bytes = max_json_bytes
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.allowed_symbols = normalize_symbols(allowed_symbols or ALLOWED_SYMBOLS)
        self.rate_limits = rate_limits or {
            ToolName.GET_MARKET_DATA: TOOL_RATE_LIMIT_MARKET_DATA,
            ToolName.GET_ORDER_BOOK: TOOL_RATE_LIMIT_ORDER_BOOK,
            ToolName.GET_RECENT_TRADES: TOOL_RATE_LIMIT_RECENT_TRADES,
        }
        self.rate_limit_window_seconds = rate_limit_window_seconds
        self.dedup_window_seconds = dedup_window_seconds or rate_limit_window_seconds
        self._tool_counts: Dict[str, int] = {}
        self._tool_window_start = time.time()
        self._recent_responses: Dict[str, Tuple[float, ToolResponse]] = {}
        self.error_callback = error_callback
        self.success_callback = success_callback
        self.portfolio_id = portfolio_id
        self.run_id = run_id

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self.cache_ttl_seconds:
            return None
        return entry["data"]

    def _cache_set(self, key: str, data: Dict[str, Any]) -> None:
        self._cache[key] = {"ts": time.time(), "data": data}

    def _is_symbol_allowed(self, symbol: str) -> bool:
        if not symbol:
            return False
        if "*" in self.allowed_symbols:
            return True
        try:
            canonical = normalize_symbol(symbol)
        except ValueError:
            return False
        return canonical in self.allowed_symbols

    def _prune_response_cache(self, now: float) -> None:
        if not self._recent_responses:
            return
        horizon = self.dedup_window_seconds
        if horizon is None:
            return
        stale_keys = [k for k, (ts, _) in self._recent_responses.items() if now - ts > horizon]
        for key in stale_keys:
            self._recent_responses.pop(key, None)

    def _reset_rate_window(self, now: float) -> None:
        if now - self._tool_window_start >= self.rate_limit_window_seconds:
            self._tool_window_start = now
            self._tool_counts = {}

    def _request_keys(self, req: ToolRequest) -> List[str]:
        symbol = getattr(req.params, "symbol", "") if hasattr(req, "params") else ""
        try:
            symbol = normalize_symbol(symbol) if symbol else ""
        except ValueError:
            symbol = symbol.upper() if symbol else ""
        if req.tool == ToolName.GET_MARKET_DATA:
            timeframes = getattr(req.params, "timeframes", []) if hasattr(req, "params") else []
            limit = getattr(req.params, "limit", None)
            keys = [f"{req.tool.value}:{symbol}:{tf}:limit{limit}" for tf in timeframes or ["none"]]
            return keys
        if req.tool == ToolName.GET_ORDER_BOOK:
            depth = getattr(req.params, "depth", None)
            return [f"{req.tool.value}:{symbol}:depth{depth}"]
        if req.tool == ToolName.GET_RECENT_TRADES:
            limit = getattr(req.params, "limit", None)
            return [f"{req.tool.value}:{symbol}:limit{limit}"]
        return [req.tool.value]

    def _check_rate_limit(self, tool: ToolName, keys: List[str]) -> bool:
        now = time.time()
        self._reset_rate_window(now)
        limit = self.rate_limits.get(tool)
        if limit is None:
            return True

        for key in keys:
            if self._tool_counts.get(key, 0) >= limit:
                return False

        for key in keys:
            self._tool_counts[key] = self._tool_counts.get(key, 0) + 1
        return True

    def _get_cached_response(self, req: ToolRequest, keys: List[str]) -> Tuple[Optional[ToolResponse], Optional[float]]:
        now = time.time()
        horizon = self.dedup_window_seconds
        if horizon is None:
            return None, None
        cached_entries: List[Tuple[float, ToolResponse]] = []
        for key in keys:
            ts_resp = self._recent_responses.get(key)
            if not ts_resp:
                return None, None
            ts, cached = ts_resp
            age = now - ts
            if age > horizon:
                return None, None
            cached_entries.append((age, cached))

        if not cached_entries:
            return None, None

        newest_age, cached_response = min(cached_entries, key=lambda item: item[0])
        data_copy = copy.deepcopy(cached_response.data)
        meta = data_copy.get("meta", {}) if isinstance(data_copy, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        meta.update(
            {
                "deduped": True,
                "dedup_age_ms": newest_age * 1000,
            }
        )
        if isinstance(data_copy, dict):
            data_copy["meta"] = meta
        response = ToolResponse(id=req.id, tool=req.tool, data=data_copy, error=cached_response.error)
        return response, newest_age * 1000

    def _record_response_cache(self, keys: List[str], response: ToolResponse) -> None:
        if response.error:
            return
        ts = time.time()
        for key in keys:
            self._recent_responses[key] = (ts, response)

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
            pending.append((tf, asyncio.create_task(self._fetch_ohlcv(params.symbol, tf, limit))))

        for tf, task in pending:
            started = time.time()
            try:
                data = await task
                if not data:
                    raise ValueError("Empty OHLCV response")
                shaped = normalize_candles(data, tf, params.limit, self.max_bars)
                ended = time.time()
                shaped["meta"] = {
                    "fetched_at": ended,
                    "latency_ms": (ended - started) * 1000,
                    "data_age_ms": None,
                }
                self._cache_set(f"ohlcv:{params.symbol}:{tf}:{limit}", shaped)
                results["timeframes"][tf] = shaped
            except Exception as exc:
                logger.warning(f"OHLCV fetch failed for {params.symbol} {tf}: {exc}")
                # Fallback: build candles from trades when available
                try:
                    trades_raw = await self._fetch_recent_trades(params.symbol, limit)
                    fallback = self._build_candles_from_trades(trades_raw, tf, limit)
                    shaped = normalize_candles(fallback, tf, params.limit, self.max_bars)
                    ended = time.time()
                    shaped["meta"] = {
                        "fetched_at": ended,
                        "latency_ms": (ended - started) * 1000,
                        "data_age_ms": None,
                        "fallback": True,
                    }
                    self._cache_set(f"ohlcv:{params.symbol}:{tf}:{limit}", shaped)
                    results["timeframes"][tf] = shaped
                except Exception as inner_exc:
                    logger.warning(f"Fallback trades->candles failed for {params.symbol} {tf}: {inner_exc}")
                    results["timeframes"][tf] = {"error": str(exc), "timeframe": tf, "candles": []}
        return results

    async def fetch_order_book(self, params: OrderBookParams) -> Dict[str, Any]:
        started = time.time()
        depth = min(params.depth, self.max_depth)
        cache_key = f"order_book:{params.symbol}:{depth}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        raw = await self._fetch_order_book(params.symbol, depth)
        shaped = normalize_order_book(raw, params.depth, self.max_depth)
        ended = time.time()
        meta = shaped.get("meta", {})
        meta["fetched_at"] = time.time()
        meta["latency_ms"] = (ended - started) * 1000
        ts = raw.get("timestamp") or raw.get("ts")
        if ts is not None:
            now_ms = time.time() * 1000
            meta["data_age_ms"] = max(0, now_ms - (ts if ts > 1e12 else ts * 1000))
        else:
            meta["data_age_ms"] = None
        shaped["meta"] = meta
        self._cache_set(cache_key, shaped)
        return shaped

    async def fetch_recent_trades(self, params: RecentTradesParams) -> Dict[str, Any]:
        started = time.time()
        limit = min(params.limit, self.max_trades)
        cache_key = f"trades:{params.symbol}:{limit}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached
        raw = await self._fetch_recent_trades(params.symbol, limit)
        shaped = normalize_trades(raw, params.limit, self.max_trades)
        ended = time.time()
        shaped["meta"] = {
            "fetched_at": time.time(),
            "latency_ms": (ended - started) * 1000,
        }
        self._cache_set(cache_key, shaped)
        return shaped

    async def handle_requests(self, requests: List[ToolRequest]) -> List[ToolResponse]:
        responses: List[ToolResponse] = []
        had_error = False
        thrash_events: Dict[str, List[Dict[str, Any]]] = {"rate_limited": [], "deduped": []}
        now = time.time()
        self._prune_response_cache(now)
        for req in requests:
            try:
                symbol = getattr(req.params, "symbol", "") if hasattr(req, "params") else ""
                keys = self._request_keys(req)
                if symbol and not self._is_symbol_allowed(symbol):
                    logger.info(f"Tool request {req.id} rejected: symbol {symbol} not in allowlist")
                    had_error = True
                    if self.error_callback:
                        try:
                            self.error_callback(req, "symbol_not_allowed")
                        except Exception:
                            logger.debug("Tool error callback failed")
                    responses.append(
                        ToolResponse(
                            id=req.id,
                            tool=req.tool,
                            data={},
                            error=f"symbol_not_allowed:{symbol}",
                        )
                    )
                    continue

                if not self._check_rate_limit(req.tool, keys):
                    logger.info(f"Tool request {req.id} dropped: rate limit exceeded for {req.tool.value}")
                    had_error = True
                    if self.error_callback:
                        try:
                            self.error_callback(req, "rate_limited")
                        except Exception:
                            logger.debug("Tool error callback failed")
                    responses.append(
                        ToolResponse(
                            id=req.id,
                            tool=req.tool,
                            data={},
                            error="rate_limited",
                        )
                    )
                    thrash_events["rate_limited"].append({"tool": req.tool.value, "keys": keys})
                    continue

                cached_response, dedup_age_ms = self._get_cached_response(req, keys)
                if cached_response:
                    responses.append(cached_response)
                    thrash_events["deduped"].append({
                        "tool": req.tool.value,
                        "keys": keys,
                        "dedup_age_ms": dedup_age_ms,
                    })
                    continue

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
                self._record_response_cache(keys, responses[-1])
            except Exception as exc:
                logger.error(f"Tool request {req.id} failed: {exc}")
                had_error = True
                if self.error_callback:
                    try:
                        self.error_callback(req, exc)
                    except Exception:
                        logger.debug("Tool error callback failed")
                responses.append(ToolResponse(id=req.id, tool=req.tool, data={}, error=str(exc)))
                continue

            # Track per-response errors from the tool layer
            if responses and responses[-1].error:
                had_error = True
                if self.error_callback:
                    try:
                        self.error_callback(req, responses[-1].error)
                    except Exception:
                        logger.debug("Tool error callback failed")
            else:
                # No error for this request; defer success notification until batch end
                pass

        if self.success_callback and not had_error and requests:
            try:
                self.success_callback()
            except Exception:
                logger.debug("Tool success callback failed")

        if (thrash_events["rate_limited"] or thrash_events["deduped"]) and telemetry_logger:
            try:
                payload_dict = {
                    "type": "tool_thrash",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "rate_limited": thrash_events["rate_limited"],
                    "deduped": thrash_events["deduped"],
                    "window_seconds": self.rate_limit_window_seconds,
                    "portfolio_id": self.portfolio_id,
                    "run_id": self.run_id,
                }
                payload = json.dumps(payload_dict)
                telemetry_logger.info(payload)
                logger.info(payload)
            except Exception:
                logger.debug("Failed to log tool thrash telemetry")
        return responses
