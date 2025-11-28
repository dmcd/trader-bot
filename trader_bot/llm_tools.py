import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trader_bot.config import (
    TOOL_ALLOWED_TIMEFRAMES,
    TOOL_DEFAULT_TIMEFRAMES,
    TOOL_MAX_BARS,
    TOOL_MAX_DEPTH,
    TOOL_MAX_JSON_BYTES,
    TOOL_MAX_TRADES,
)

logger = logging.getLogger(__name__)


def _clean_timeframes(raw: List[str]) -> List[str]:
    """
    Normalize and filter timeframes to a whitelisted set.

    - Trims whitespace
    - Maps common aliases (1hr -> 1h, 1day -> 1d, 6hr -> 6h)
    - Drops unsupported entries (e.g., 4h on Gemini)
    """
    alias_map = {
        "1hr": "1h",
        "1hour": "1h",
        "1day": "1d",
        "1d": "1d",
        "6hr": "6h",
        "6hour": "6h",
        "30min": "30m",
        "30m": "30m",
        "15min": "15m",
        "15m": "15m",
        "5min": "5m",
        "5m": "5m",
        "1min": "1m",
        "1m": "1m",
        "4hr": "6h",
        "4hour": "6h",
        "4h": "6h",
    }

    allowed_raw = [tf.strip() for tf in TOOL_ALLOWED_TIMEFRAMES if tf and tf.strip()]
    allowed_map = {tf.lower(): tf for tf in allowed_raw}

    normalized: List[str] = []
    for tf in raw:
        if not tf:
            continue
        trimmed = tf.strip().lower()
        if not trimmed:
            continue
        canonical = alias_map.get(trimmed, trimmed)
        allowed_value = allowed_map.get(canonical)
        if allowed_value:
            normalized.append(allowed_value)

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for tf in normalized:
        if tf not in seen:
            seen.add(tf)
            ordered.append(tf)
    return ordered


class ToolBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketDataParams(ToolBaseModel):
    symbol: str
    timeframes: List[str] = Field(
        default_factory=lambda: _clean_timeframes(TOOL_DEFAULT_TIMEFRAMES)
    )
    limit: int = Field(default=500, gt=0, le=TOOL_MAX_BARS)
    include_volume: bool = True

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, value: List[str]) -> List[str]:
        cleaned = _clean_timeframes(value)
        if not cleaned:
            raise ValueError("At least one timeframe is required")
        return cleaned

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Symbol is required")
        return value.strip().upper()


class OrderBookParams(ToolBaseModel):
    symbol: str
    depth: int = Field(default=50, gt=0, le=TOOL_MAX_DEPTH)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Symbol is required")
        return value.strip().upper()


class RecentTradesParams(ToolBaseModel):
    symbol: str
    limit: int = Field(default=100, gt=0, le=TOOL_MAX_TRADES)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Symbol is required")
        return value.strip().upper()


class ToolName(str, Enum):
    GET_MARKET_DATA = "get_market_data"
    GET_ORDER_BOOK = "get_order_book"
    GET_RECENT_TRADES = "get_recent_trades"


TOOL_PARAM_MODELS = {
    ToolName.GET_MARKET_DATA: MarketDataParams,
    ToolName.GET_ORDER_BOOK: OrderBookParams,
    ToolName.GET_RECENT_TRADES: RecentTradesParams,
}


class ToolRequest(ToolBaseModel):
    id: str = Field(min_length=1)
    tool: ToolName
    params: Union[MarketDataParams, OrderBookParams, RecentTradesParams]

    @model_validator(mode="before")
    @classmethod
    def validate_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
            
        tool_name = data.get("tool")
        params = data.get("params")
        
        if not tool_name or params is None:
            return data

        # Look up the target model class
        try:
            # Handle both string and Enum input for tool
            tool_enum = ToolName(tool_name)
            target_model = TOOL_PARAM_MODELS.get(tool_enum)
        except ValueError:
            # Invalid tool name, let standard validation handle it
            return data

        if target_model and isinstance(params, dict):
            try:
                # Explicitly validate against the target model
                validated_params = target_model.model_validate(params)
                data["params"] = validated_params
            except Exception as exc:
                raise ValueError(f"Invalid params for {tool_name}: {exc}") from exc
                
        return data


class ToolResponse(ToolBaseModel):
    id: str
    tool: ToolName
    data: Dict[str, Any]
    error: Optional[str] = None


def normalize_candles(
    raw: List[List[Any]],
    timeframe: str,
    requested_limit: int,
    max_bars: int = TOOL_MAX_BARS,
) -> Dict[str, Any]:
    safe_limit = min(requested_limit or max_bars, max_bars)
    trimmed = (raw or [])[:safe_limit]
    candles: List[Dict[str, Any]] = []
    for entry in trimmed:
        if entry is None or len(entry) < 5:
            continue
        ts, o, h, l, c = entry[0], entry[1], entry[2], entry[3], entry[4]
        v = entry[5] if len(entry) > 5 else None
        if ts is None or o is None or h is None or l is None or c is None:
            continue
        candles.append(
            {
                "ts": int(ts),
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "v": float(v) if v is not None else None,
            }
        )

    sorted_candles = sorted(candles, key=lambda c: c["ts"])
    summary: Dict[str, Any] = {}
    if len(sorted_candles) >= 2:
        first_close = sorted_candles[0]["c"]
        last_close = sorted_candles[-1]["c"]
        if first_close:
            summary["change_pct"] = ((last_close - first_close) / first_close) * 100
        summary["last"] = last_close

    truncated = len(raw or []) > safe_limit
    return {
        "timeframe": timeframe,
        "requested": requested_limit,
        "returned": len(candles),
        "truncated": truncated,
        "candles": candles,
        "summary": summary,
    }


def normalize_order_book(
    raw: Dict[str, Any],
    requested_depth: int,
    max_depth: int = TOOL_MAX_DEPTH,
) -> Dict[str, Any]:
    depth = min(requested_depth or max_depth, max_depth)
    bids = (raw.get("bids") or [])[:depth]
    asks = (raw.get("asks") or [])[:depth]
    ts = raw.get("timestamp") or raw.get("ts")
    mid = None
    spread_bps = None
    if bids and asks:
        bid = bids[0][0]
        ask = asks[0][0]
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            if mid:
                spread_bps = ((ask - bid) / mid) * 10000

    def _depth_notional(levels: List[List[Any]]) -> float:
        total = 0.0
        for level in levels:
            if len(level) < 2:
                continue
            price, size = level[0], level[1]
            if price is None or size is None:
                continue
            total += float(price) * float(size)
        return total

    meta = {
        "requested": requested_depth,
        "returned": max(len(bids), len(asks)),
        "depth": depth,
        "mid": mid,
        "spread_bps": spread_bps,
        "bid_notional": _depth_notional(bids),
        "ask_notional": _depth_notional(asks),
    }
    return {
        "bids": bids,
        "asks": asks,
        "ts": ts,
        "truncated": (len(raw.get("bids") or []) > depth) or (len(raw.get("asks") or []) > depth),
        "meta": meta,
    }


def normalize_trades(
    raw: List[Dict[str, Any]],
    requested_limit: int,
    max_trades: int = TOOL_MAX_TRADES,
) -> Dict[str, Any]:
    limit = min(requested_limit or max_trades, max_trades)
    trimmed = (raw or [])[:limit]
    trades: List[Dict[str, Any]] = []
    for trade in trimmed:
        price = trade.get("price")
        size = trade.get("amount") or trade.get("size")
        ts = trade.get("timestamp") or trade.get("ts")
        side = trade.get("side")
        if price is None or size is None or ts is None:
            continue
        trades.append(
            {
                "ts": int(ts),
                "side": (side or "").upper(),
                "price": float(price),
                "size": float(size),
            }
        )
    truncated = len(raw or []) > limit
    return {
        "requested": requested_limit,
        "returned": len(trades),
        "truncated": truncated,
        "trades": trades,
    }


def estimate_json_bytes(payload: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    except Exception as exc:
        logger.debug(f"Failed to estimate payload size: {exc}")
        return 0


def clamp_payload_size(payload: Dict[str, Any], max_bytes: int = TOOL_MAX_JSON_BYTES) -> Dict[str, Any]:
    """
    Ensure tool payload fits within max_bytes by progressively trimming lists (candles,
    order book levels, trades). Marks payload as truncated when modifications occur.
    """
    try:
        # Deep copy via JSON for predictable sizing
        result: Dict[str, Any] = json.loads(json.dumps(payload, default=str))
    except Exception:
        # Fall back to shallow copy
        result = dict(payload)

    def _size(obj: Dict[str, Any]) -> int:
        return estimate_json_bytes(obj)

    def _mark_truncated(note: str = "") -> None:
        result["truncated"] = True
        if note:
            existing = result.get("note", "")
            if existing:
                result["note"] = f"{existing}; {note}"
            else:
                result["note"] = note

    if _size(result) <= max_bytes:
        return result

    # Step 1: shrink candles per timeframe
    timeframes = result.get("timeframes")
    if isinstance(timeframes, dict):
        for tf_data in timeframes.values():
            if isinstance(tf_data, dict) and "candles" in tf_data:
                candles = tf_data.get("candles") or []
                if len(candles) > 10:
                    new_len = max(5, len(candles) // 2)
                    tf_data["candles"] = candles[:new_len]
                    tf_data["truncated"] = True
                    _mark_truncated("candles trimmed")
        if _size(result) <= max_bytes:
            return result

    # Step 2: shrink order book depth
    if "bids" in result or "asks" in result:
        bids = result.get("bids") or []
        asks = result.get("asks") or []
        max_depth = max(1, min(len(bids), len(asks), TOOL_MAX_DEPTH))
        if max_depth > 5:
            max_depth = max(5, max_depth // 2)
        result["bids"] = bids[:max_depth]
        result["asks"] = asks[:max_depth]
        _mark_truncated("order book depth trimmed")
        if _size(result) <= max_bytes:
            return result

    # Step 3: shrink trades list
    if "trades" in result and isinstance(result.get("trades"), list):
        trades = result.get("trades") or []
        if len(trades) > 10:
            result["trades"] = trades[: max(5, len(trades) // 2)]
            _mark_truncated("trades trimmed")
            if _size(result) <= max_bytes:
                return result

    # Step 4: aggressively drop heavy fields if still too large
    heavy_fields = ("candles", "bids", "asks", "trades")
    # Drop nested candles first
    if isinstance(timeframes, dict):
        for tf_data in timeframes.values():
            if isinstance(tf_data, dict) and "candles" in tf_data:
                tf_data["candles"] = []
                tf_data["truncated"] = True
                _mark_truncated("candles dropped")
    for field in heavy_fields:
        if field in result:
            result[field] = []
            _mark_truncated(f"{field} dropped")
    return result
