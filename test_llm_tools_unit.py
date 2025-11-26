import pytest

from llm_tools import (
    MarketDataParams,
    OrderBookParams,
    ToolName,
    ToolRequest,
    clamp_payload_size,
    normalize_candles,
    normalize_order_book,
)


def test_tool_request_validates_and_normalizes_symbol():
    req = ToolRequest(
        id="r1",
        tool=ToolName.GET_MARKET_DATA,
        params={
            "symbol": "btc/usd",
            "timeframes": ["1m", "5m", "1m"],  # duplicate to ensure de-duplication
            "limit": 10,
        },
    )
    assert isinstance(req.params, MarketDataParams)
    assert req.params.symbol == "BTC/USD"
    assert req.params.timeframes == ["1m", "5m"]


def test_normalize_candles_truncates_and_summarizes():
    raw = [
        [3, 3, 4, 2, 3, 30],
        [2, 2, 3, 1, 2, 20],
        [1, 1, 2, 0.5, 1, 10],
    ]
    shaped = normalize_candles(raw, "1m", requested_limit=2, max_bars=2)
    assert shaped["returned"] == 2
    assert shaped["truncated"] is True
    assert shaped["summary"]["last"] == 3
    assert shaped["summary"]["change_pct"] == pytest.approx(50.0)


def test_normalize_order_book_mid_and_spread():
    raw = {
        "bids": [[99, 1], [98, 2]],
        "asks": [[101, 1.5], [102, 3]],
        "timestamp": 123,
    }
    shaped = normalize_order_book(raw, requested_depth=1, max_depth=2)
    assert shaped["meta"]["mid"] == 100
    assert shaped["meta"]["spread_bps"] == pytest.approx(200.0)
    assert shaped["bids"] == [[99, 1]]
    assert shaped["asks"] == [[101, 1.5]]


def test_clamp_payload_size_marks_truncation_when_exceeding_max_bytes():
    payload = {"big": "x" * 50}
    clamped = clamp_payload_size(payload, max_bytes=10)
    assert clamped.get("truncated") is True
    assert "note" in clamped
