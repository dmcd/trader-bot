import pytest

from trader_bot.llm_tools import (
    MarketDataParams,
    OrderBookParams,
    ToolName,
    ToolRequest,
    _clean_timeframes,
    estimate_json_bytes,
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
    payload = {
        "timeframes": {
            "1m": {"candles": [[i, 1, 1, 1, 1, 1] for i in range(100)], "summary": {}}
        },
        "bids": [[100 - i, 1] for i in range(50)],
        "asks": [[101 + i, 1] for i in range(50)],
        "trades": [{"ts": i, "price": 100 + i, "amount": 1} for i in range(50)],
    }
    max_bytes = 500
    clamped = clamp_payload_size(payload, max_bytes=max_bytes)
    assert clamped.get("truncated") is True
    assert "note" in clamped
    assert estimate_json_bytes(clamped) <= max_bytes
    # Ensure large lists were reduced
    tf = clamped["timeframes"]["1m"]
    assert len(tf.get("candles", [])) <= 50
    assert len(clamped.get("bids", [])) <= 50
    assert len(clamped.get("trades", [])) <= 50


def test_tool_request_disambiguates_union_types():
    # Regression test for "get_recent_trades" being confused with "get_market_data"
    # because MarketDataParams has defaults that match RecentTradesParams structure.
    req = ToolRequest(
        id="r2",
        tool=ToolName.GET_RECENT_TRADES,
        params={
            "symbol": "BTC/USD",
            "limit": 50,
        },
    )
    # Should be RecentTradesParams, NOT MarketDataParams
    assert req.tool == ToolName.GET_RECENT_TRADES
    # RecentTradesParams does NOT have 'timeframes' or 'include_volume'
    assert not hasattr(req.params, "timeframes")
    assert not hasattr(req.params, "include_volume")
    assert req.params.limit == 50


@pytest.mark.parametrize(
    "raw,expected",
    [
        (["1m", "1m", "5m", "4h", " 1d "], ["1m", "5m", "6h", "1d"]),
        (["1hr", "1hour", "30min", "bad", ""], ["1h", "30m"]),
        (["1", "15", "60", "D"], ["1m", "15m", "1h", "1d"]),
        (["   ", None, ""], []),
    ],
)
def test_clean_timeframes_normalizes_and_dedupes(raw, expected):
    cleaned = _clean_timeframes(raw)
    assert cleaned == expected
    # All results should be in allowed map (no rogue entries)
    assert all(isinstance(tf, str) and tf for tf in cleaned)


def test_clamp_payload_size_idempotent_and_respects_budget():
    payload = {
        "timeframes": {
            "1m": {"candles": [[i, 1, 1, 1, 1, 1] for i in range(200)], "summary": {}},
        },
        "bids": [[100 - i, 1] for i in range(200)],
        "asks": [[101 + i, 1] for i in range(200)],
        "trades": [{"ts": i, "price": 100 + i, "amount": 1} for i in range(200)],
    }
    max_bytes = 800
    clamped = clamp_payload_size(payload, max_bytes=max_bytes)
    assert estimate_json_bytes(clamped) <= max_bytes
    assert clamped.get("truncated") is True
    # Re-clamping should not grow or remove the truncated marker
    reclamped = clamp_payload_size(clamped, max_bytes=max_bytes)
    assert reclamped.get("truncated") is True
    assert estimate_json_bytes(reclamped) <= max_bytes
