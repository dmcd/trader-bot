"""Helper script to capture IB fixtures for playback mode.

Run manually with IB Gateway/TWS up and API enabled. Keeps output compatible
with `tests/ib_fakes.FakeIB.from_fixture` so integration tests can replay
recorded sessions without live connectivity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

try:
    from ib_insync import Forex, IB, Stock
except Exception:  # pragma: no cover - optional tool
    IB = None


async def capture_fixture(output_path: Path, *, symbol: str = "BHP", exchange: str = "ASX") -> None:
    if IB is None:
        raise RuntimeError("ib_insync is required to capture live IB fixtures")

    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "7497"))
    client_id = int(os.getenv("IB_CLIENT_ID", "101"))
    account_id = os.getenv("IB_ACCOUNT_ID", "")

    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id)

    account_values = await ib.accountValuesAsync()
    stock = Stock(symbol, exchange, "AUD")
    fx = Forex("AUDUSD")

    stock_ticker = await ib.reqMktDataAsync(stock, "", False, False)
    fx_ticker = await ib.reqMktDataAsync(fx, "", False, False)
    bars = await ib.reqHistoricalDataAsync(
        stock,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )

    bundle = {
        "account_values": [
            {"tag": a.tag, "value": a.value, "currency": a.currency, "account": a.account or account_id}
            for a in account_values
        ],
        "market_data": {
            f"{symbol}AUD": {
                "price": stock_ticker.marketPrice(),
                "bid": stock_ticker.bid,
                "ask": stock_ticker.ask,
                "bid_size": stock_ticker.bidSize,
                "ask_size": stock_ticker.askSize,
                "last": stock_ticker.last,
                "volume": stock_ticker.volume,
            },
            "AUDUSD": {
                "price": fx_ticker.marketPrice(),
                "bid": fx_ticker.bid,
                "ask": fx_ticker.ask,
                "bid_size": fx_ticker.bidSize,
                "ask_size": fx_ticker.askSize,
                "last": fx_ticker.last,
                "volume": fx_ticker.volume,
            },
        },
        "historical_data": {
            f"{symbol}AUD": [
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    ib.disconnect()


def _parse_args():
    parser = argparse.ArgumentParser(description="Capture IB fixtures for playback tests")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/ib/playback_bundle.json"))
    parser.add_argument("--symbol", default="BHP", help="Symbol to capture (default: BHP)")
    parser.add_argument("--exchange", default="ASX", help="Exchange for the capture symbol")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - manual tool
    args = _parse_args()
    asyncio.run(capture_fixture(args.output, symbol=args.symbol, exchange=args.exchange))
