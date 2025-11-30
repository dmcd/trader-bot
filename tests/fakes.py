class FakeBot:
    """Shared lightweight trading bot stub for runner tests."""

    def __init__(
        self,
        *,
        equity=1000.0,
        price=100.0,
        open_orders=None,
        positions=None,
        trades=None,
        bid=None,
        ask=None,
        spread_pct=0.1,
        bid_size=1.0,
        ask_size=1.0,
        latency_ms=1,
        exchange=None,
    ):
        self.equity = equity
        self.price = price
        self.bid = bid
        self.ask = ask
        self.spread_pct = spread_pct
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.latency_ms = latency_ms
        self.open_orders = list(open_orders) if open_orders is not None else [{"id": 1, "symbol": "BTC/USD"}]
        self.positions = list(positions) if positions is not None else []
        self.trades = list(trades) if trades is not None else []
        self.exchange = exchange
        self.cancelled = []
        self.cancel_calls = 0
        self.place_calls = []
        self.closed = False

    def _resolve_price(self, symbol):
        if isinstance(self.price, dict) and self.price:
            return self.price.get(symbol, next(iter(self.price.values())))
        return self.price

    async def connect_async(self):
        return None

    async def get_equity_async(self):
        return self.equity

    async def get_market_data_async(self, symbol):
        price = self._resolve_price(symbol)
        bid = self.bid if self.bid is not None else price - 0.5
        ask = self.ask if self.ask is not None else price + 0.5
        return {
            "symbol": symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "spread_pct": self.spread_pct if self.spread_pct is not None else 0.0,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "_latency_ms": self.latency_ms,
        }

    async def get_positions_async(self):
        return list(self.positions)

    async def get_open_orders_async(self):
        return list(self.open_orders)

    async def cancel_open_order_async(self, order_id):
        self.cancel_calls += 1
        self.cancelled.append(order_id)
        self.open_orders = [order for order in self.open_orders if order.get("id") != order_id]
        return True

    async def fetch_ohlcv(self, *_args, **_kwargs):
        return []

    async def get_my_trades_async(self, *_args, **_kwargs):
        return list(self.trades)

    async def place_order_async(self, symbol, action, quantity, prefer_maker=True):
        self.place_calls.append((symbol, action, quantity, prefer_maker))
        liquidity = "maker" if prefer_maker else "taker"
        return {"order_id": str(len(self.place_calls)), "liquidity": liquidity}

    async def close(self):
        self.closed = True
        return None


class FakeExchange:
    """Minimal ccxt-like exchange stub for Gemini trader tests."""

    def __init__(self, balances=None, tickers=None):
        self._balances = balances or {}
        self._tickers = tickers or {}

    async def fetch_balance(self):
        return {"total": self._balances, "timestamp": 12345}

    async def fetch_ticker(self, symbol):
        data = self._tickers.get(symbol)
        if data is None:
            raise RuntimeError(f"no ticker for {symbol}")
        return data
