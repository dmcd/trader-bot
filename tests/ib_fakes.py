"""Reusable Interactive Brokers test doubles and fixture helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTicker:
    def __init__(
        self,
        price: float | None,
        bid: float | None = None,
        ask: float | None = None,
        *,
        bid_size: float | None = None,
        ask_size: float | None = None,
        last: float | None = None,
        volume: float | None = None,
    ):
        self._price = price
        self.bid = bid
        self.ask = ask
        self.bidSize = bid_size
        self.askSize = ask_size
        self.last = last
        self.volume = volume

    def marketPrice(self):
        return self._price

    def midpoint(self):
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self._price


class FakeOrderStatus:
    def __init__(
        self,
        status: str,
        *,
        filled: float | None = None,
        remaining: float | None = None,
        avg_fill_price: float | None = None,
        order_id: int | None = None,
        commission: float | None = None,
        liquidity: str | None = None,
        perm_id: int | None = None,
    ):
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avgFillPrice = avg_fill_price
        self.orderId = order_id
        self.commission = commission
        self.liquidity = liquidity
        self.permId = perm_id


class FakeTrade:
    def __init__(
        self,
        statuses,
        *,
        order_id: int | None = None,
        client_id: str | None = None,
        fills=None,
    ):
        self._statuses = list(statuses or [])
        if not self._statuses:
            self._statuses.append(FakeOrderStatus("Submitted"))
        self.orderStatus = self._statuses[0]
        self.order = None
        self.contract = None
        self.orderId = order_id or getattr(self.orderStatus, "orderId", None)
        self.clientId = client_id
        self.fills = fills or []

    def advance_status(self):
        if len(self._statuses) > 1:
            self._statuses.pop(0)
            self.orderStatus = self._statuses[0]


class FakeContract:
    def __init__(self, symbol: str, currency: str, *, sec_type: str = "STK", exchange: str = "ASX"):
        self.symbol = symbol
        self.currency = currency
        self.secType = sec_type
        self.exchange = exchange

    def pair(self):
        if self.secType == "FX":
            return f"{self.symbol}{self.currency}"
        return None


class FakePortfolioEntry:
    def __init__(self, contract: FakeContract, position: float, avg_cost: float):
        self.contract = contract
        self.position = position
        self.avgCost = avg_cost


class FakeFill:
    def __init__(self, price: float, size: float, timestamp_ms: int, *, commission: float | None = None, liquidity: str | None = None):
        self.price = price
        self.size = size
        self.time = timestamp_ms
        self.commission = commission
        self.liquidity = liquidity


class FakeBar:
    def __init__(self, date, open, high, low, close, volume):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class FakeIB:
    def __init__(
        self,
        *,
        connected: bool = False,
        fail_connect: bool = False,
        fail_ping: bool = False,
        async_disconnect: bool = False,
        account_values=None,
        market_data=None,
        order_statuses=None,
        portfolio=None,
        open_orders=None,
        trades=None,
        historical_data=None,
        md_requires_delayed: bool = False,
    ):
        self.connected = connected
        self.fail_connect = fail_connect
        self.fail_ping = fail_ping
        self.connect_calls = []
        self.req_time_calls = 0
        self.disconnect_calls = 0
        self.disconnect_async_calls = 0
        self.account_values_calls = 0
        self.req_mkt_data_calls = []
        self.req_market_data_type_calls = []
        self.place_order_calls = []
        self.account_values = account_values or []
        self.market_data = market_data or {}
        self.order_statuses = order_statuses or {}
        self.portfolio_entries = portfolio or []
        self.open_orders_list = open_orders or []
        self.trades_list = trades or []
        self.historical_data = historical_data or {}
        self.hist_calls = []
        self.market_data_type = 1
        self.md_requires_delayed = md_requires_delayed
        if not async_disconnect:
            # Shadow the coroutine so getattr returns None
            self.disconnectAsync = None

    async def connectAsync(self, host, port, clientId):
        self.connect_calls.append({"host": host, "port": port, "clientId": clientId})
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True
        return True

    def isConnected(self):
        return self.connected

    async def reqCurrentTimeAsync(self):
        if self.fail_ping:
            raise RuntimeError("ping failed")
        self.req_time_calls += 1
        return 123

    def reqCurrentTime(self):
        self.req_time_calls += 1
        return 123

    async def accountValuesAsync(self):
        self.account_values_calls += 1
        return list(self.account_values)

    def accountValues(self):
        self.account_values_calls += 1
        return list(self.account_values)

    def _contract_key(self, contract):
        pair_fn = getattr(contract, "pair", None)
        if callable(pair_fn):
            val = pair_fn()
            if isinstance(val, str):
                return val.replace(".", "")
            return val
        symbol = getattr(contract, "symbol", None)
        currency = getattr(contract, "currency", None)
        if symbol and currency:
            return f"{symbol}{currency}"
        return str(contract)

    async def reqMktDataAsync(self, contract, *args, **kwargs):
        key = self._contract_key(contract)
        self.req_mkt_data_calls.append(key)
        if self.md_requires_delayed and self.market_data_type not in (3, 4):
            err = PermissionError("Requested market data is not subscribed.")
            err.code = 354
            raise err
        return self.market_data.get(key)

    async def placeOrderAsync(self, contract, order):
        key = self._contract_key(contract)
        self.place_order_calls.append({"contract": key, "order": order})
        statuses = self.order_statuses.get(key, [])
        trade = FakeTrade(statuses)
        trade.order = order
        trade.contract = contract
        return trade

    async def portfolioAsync(self):
        return list(self.portfolio_entries)

    async def openOrdersAsync(self):
        return list(self.open_orders_list)

    async def tradesAsync(self):
        return list(self.trades_list)

    async def reqHistoricalDataAsync(self, contract, **params):
        key = self._contract_key(contract)
        self.hist_calls.append({"key": key, "params": params})
        return list(self.historical_data.get(key, []))

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    async def disconnectAsync(self):
        self.disconnect_async_calls += 1
        self.connected = False

    def reqMarketDataType(self, data_type: int):
        self.req_market_data_type_calls.append(data_type)
        self.market_data_type = data_type

    @classmethod
    def from_fixture(cls, bundle: dict[str, Any]):
        account_values = [
            SimpleNamespace(
                tag=entry.get("tag"),
                value=str(entry.get("value")),
                currency=entry.get("currency"),
                account=entry.get("account"),
            )
            for entry in bundle.get("account_values", [])
        ]

        market_data = {
            key: FakeTicker(**params) for key, params in (bundle.get("market_data") or {}).items()
        }

        order_statuses = {
            key: [FakeOrderStatus(**status) for status in statuses]
            for key, statuses in (bundle.get("order_statuses") or {}).items()
        }

        portfolio_entries = [
            FakePortfolioEntry(
                FakeContract(entry["symbol"], entry["currency"], sec_type=entry.get("sec_type", "STK"), exchange=entry.get("exchange", "ASX")),
                position=entry.get("position", 0.0),
                avg_cost=entry.get("avg_cost", 0.0),
            )
            for entry in bundle.get("portfolio", [])
        ]

        open_orders = []
        for entry in bundle.get("open_orders", []):
            status = FakeOrderStatus(
                entry.get("status", "Submitted"),
                filled=entry.get("filled"),
                remaining=entry.get("remaining"),
                avg_fill_price=entry.get("avg_fill_price"),
                order_id=entry.get("order_id"),
                perm_id=entry.get("perm_id"),
            )
            trade = FakeTrade([status], order_id=entry.get("order_id"), client_id=str(entry.get("client_id") or "fixture"))
            trade.order = type(
                "Order",
                (),
                {
                    "totalQuantity": entry.get("total_quantity", entry.get("remaining")),
                    "lmtPrice": entry.get("lmt_price"),
                    "action": entry.get("action"),
                    "orderRef": entry.get("order_ref"),
                },
            )()
            trade.contract = FakeContract(
                entry.get("symbol"), entry.get("currency"), sec_type=entry.get("sec_type", "STK"), exchange=entry.get("exchange", "ASX")
            )
            open_orders.append(trade)

        trades = []
        for entry in bundle.get("trades", []):
            statuses = [
                FakeOrderStatus(
                    entry.get("status", "Filled"),
                    filled=entry.get("filled"),
                    remaining=entry.get("remaining"),
                    avg_fill_price=entry.get("avg_fill_price"),
                    order_id=entry.get("order_id"),
                    commission=entry.get("commission"),
                    liquidity=entry.get("liquidity"),
                    perm_id=entry.get("perm_id"),
                )
            ]
            fills = [FakeFill(**fill) for fill in entry.get("fills", [])]
            trade = FakeTrade(statuses, order_id=entry.get("order_id"), client_id=str(entry.get("client_id") or "fixture"), fills=fills)
            trade.order = type(
                "Order",
                (),
                {
                    "action": entry.get("action", "BUY"),
                    "totalQuantity": entry.get("total_quantity", entry.get("filled", 0)),
                },
            )()
            trade.contract = FakeContract(
                entry.get("symbol"), entry.get("currency"), sec_type=entry.get("sec_type", "STK"), exchange=entry.get("exchange", "ASX")
            )
            trades.append(trade)

        historical_data = {}
        for key, bars in (bundle.get("historical_data") or {}).items():
            historical_data[key] = [FakeBar(**bar) for bar in bars]

        return cls(
            connected=True,
            account_values=account_values,
            market_data=market_data,
            order_statuses=order_statuses,
            portfolio=portfolio_entries,
            open_orders=open_orders,
            trades=trades,
            historical_data=historical_data,
        )


def load_ib_fixture_bundle(path: str | Path | None = None) -> dict:
    bundle_path = Path(path) if path else Path(__file__).parent / "fixtures" / "ib" / "playback_bundle.json"
    bundle_path = bundle_path.expanduser()
    with open(bundle_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_fake_ib_from_bundle(bundle: dict[str, Any]) -> FakeIB:
    return FakeIB.from_fixture(bundle)
