"""Interactive Brokers adapter scaffolding."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from ib_insync import Forex, IB, LimitOrder, MarketOrder

from trader_bot.config import (
    CLIENT_ORDER_PREFIX,
    IB_ACCOUNT_ID,
    IB_BASE_CURRENCY,
    IB_ALLOWED_INSTRUMENT_TYPES,
    IB_CLIENT_ID,
    IB_EXCHANGE,
    IB_HOST,
    IB_PRIMARY_EXCHANGE,
    IB_PAPER,
    IB_PORT,
)
from trader_bot.ib_contracts import build_ib_contract
from trader_bot.symbols import DEFAULT_FX_EXCHANGE, format_symbol
from trader_bot.trader import BaseTrader

logger = logging.getLogger(__name__)


class IBTrader(BaseTrader):
    """
    BaseTrader implementation for Interactive Brokers.

    This version wires connection lifecycle and heartbeat handling; trading
    methods are stubbed until the remaining tasks in docs/ib_todo.md land.
    """

    def __init__(
        self,
        *,
        host: str = IB_HOST,
        port: int | None = IB_PORT,
        client_id: int | None = IB_CLIENT_ID,
        account_id: str | None = IB_ACCOUNT_ID,
        paper: bool = IB_PAPER,
        connect_timeout: float = 5.0,
        heartbeat_interval: float = 30.0,
        base_currency: str = IB_BASE_CURRENCY,
        default_exchange: str | None = IB_EXCHANGE,
        default_primary_exchange: str | None = IB_PRIMARY_EXCHANGE,
        allowed_instrument_types: Optional[list[str]] = None,
        fx_exchange: str = DEFAULT_FX_EXCHANGE,
        fx_cache_ttl: float = 30.0,
        order_wait_timeout: float = 5.0,
        order_poll_interval: float = 0.25,
        ib_client: Optional[IB] = None,
        monotonic: Optional[Callable[[], float]] = None,
    ):
        self.host = host
        self.paper = paper
        # Default to IBKR's paper/live ports when unset
        self.port = port or (7497 if paper else 7496)
        self.client_id = client_id if client_id is not None else 1
        self.account_id = account_id
        self.connect_timeout = connect_timeout
        self.heartbeat_interval = heartbeat_interval
        self.base_currency = base_currency.upper()
        self.default_exchange = default_exchange
        self.default_primary_exchange = default_primary_exchange
        self.allowed_instrument_types = (
            list(allowed_instrument_types) if allowed_instrument_types is not None else list(IB_ALLOWED_INSTRUMENT_TYPES)
        )
        self.fx_exchange = fx_exchange
        self.fx_cache_ttl = fx_cache_ttl
        self.order_wait_timeout = order_wait_timeout
        self.order_poll_interval = order_poll_interval
        self.ib: IB = ib_client or IB()
        self.connected = False
        self._last_ping_mono: float | None = None
        self._monotonic = monotonic or time.monotonic
        self._fx_quote_cache: Dict[str, Tuple[float, float]] = {}

    async def connect_async(self) -> None:
        """
        Connect to IBKR Gateway/TWS with optional heartbeat refresh when already connected.
        """
        if self.connected and self.ib.isConnected():
            await self._maybe_ping()
            return

        logger.info(f"Connecting to IB at {self.host}:{self.port} (paper={self.paper})")
        try:
            await asyncio.wait_for(
                self.ib.connectAsync(self.host, self.port, self.client_id),
                timeout=self.connect_timeout,
            )
        except Exception as exc:
            self.connected = False
            logger.error(f"IB connection failed: {exc}")
            raise

        if not self.ib.isConnected():
            self.connected = False
            raise RuntimeError("IB connection attempt did not complete successfully.")

        self.connected = True
        await self._ping()
        logger.info("IB connection established.")

    async def _ping(self) -> None:
        """Send a lightweight heartbeat and record freshness."""
        if not self.ib.isConnected():
            self.connected = False
            raise RuntimeError("IB client is not connected.")

        try:
            ping_fn = getattr(self.ib, "reqCurrentTimeAsync", None)
            if ping_fn:
                await asyncio.wait_for(ping_fn(), timeout=self.connect_timeout)
            else:
                # ib_insync exposes a sync variant; offload to a thread to avoid blocking.
                await asyncio.wait_for(asyncio.to_thread(self.ib.reqCurrentTime), timeout=self.connect_timeout)
            self._last_ping_mono = self._monotonic()
        except Exception as exc:
            self.connected = False
            logger.error(f"IB heartbeat failed: {exc}")
            raise

    async def _maybe_ping(self) -> None:
        """Refresh heartbeat if the prior ping is stale."""
        if self._last_ping_mono is None:
            await self._ping()
            return

        age = self._monotonic() - self._last_ping_mono
        if age >= self.heartbeat_interval:
            await self._ping()

    async def close(self) -> None:
        """Disconnect gracefully."""
        try:
            disconnect_fn = getattr(self.ib, "disconnectAsync", None)
            if disconnect_fn and asyncio.iscoroutinefunction(disconnect_fn):
                await disconnect_fn()
            else:
                self.ib.disconnect()
        finally:
            self.connected = False

    async def get_account_summary_async(self) -> Any:
        if not self.connected:
            return []

        try:
            values = await self._fetch_account_values()
        except Exception as exc:
            logger.error(f"Error fetching IB account summary: {exc}")
            return []

        summary = []
        for entry in values:
            try:
                numeric_value = float(entry.value)
            except (TypeError, ValueError):
                continue

            summary.append(
                {
                    "account": getattr(entry, "account", self.account_id) or "IBKR",
                    "tag": getattr(entry, "tag", "Unknown"),
                    "value": numeric_value,
                    "currency": (getattr(entry, "currency", None) or self.base_currency).upper(),
                }
            )

        return summary

    async def get_market_data_async(self, symbol):
        if not self.connected:
            return None

        try:
            contract, spec = build_ib_contract(
                symbol,
                allowed_instrument_types=self.allowed_instrument_types,
                default_exchange=self.default_exchange,
                default_primary_exchange=self.default_primary_exchange,
                base_currency=self.base_currency,
                fx_exchange=self.fx_exchange,
            )
        except Exception as exc:
            logger.error(f"Error building IB contract for {symbol}: {exc}")
            return None

        started = self._monotonic()
        try:
            req_fn = getattr(self.ib, "reqMktDataAsync", None)
            if req_fn:
                ticker = await asyncio.wait_for(
                    req_fn(contract, "", False, False),
                    timeout=self.connect_timeout,
                )
            else:
                ticker = await asyncio.wait_for(
                    asyncio.to_thread(self.ib.reqMktData, contract, "", False, False),
                    timeout=self.connect_timeout,
                )
        except Exception as exc:
            logger.error(f"Error fetching IB market data for {symbol}: {exc}")
            return None

        if ticker is None:
            return None

        bid = getattr(ticker, "bid", None)
        ask = getattr(ticker, "ask", None)
        bid_size = getattr(ticker, "bidSize", None)
        ask_size = getattr(ticker, "askSize", None)
        last = getattr(ticker, "last", None)
        close = getattr(ticker, "close", None)
        volume = getattr(ticker, "volume", None)
        if volume is None:
            volume = getattr(ticker, "lastSize", None)

        price = last if last is not None and last > 0 else None
        if price is None:
            price_fn = getattr(ticker, "marketPrice", None)
            if callable(price_fn):
                try:
                    candidate = price_fn()
                    if candidate is not None and candidate > 0:
                        price = candidate
                except Exception:
                    price = None
        if price is None and bid is not None and ask is not None:
            price = (bid + ask) / 2
        if price is None:
            price = bid if bid is not None else ask

        spread_pct = None
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            if mid not in (None, 0):
                spread_pct = ((ask - bid) / mid) * 100

        ob_imbalance = None
        if bid_size is not None and ask_size is not None:
            denom = bid_size + ask_size
            if denom:
                ob_imbalance = (bid_size - ask_size) / denom

        latency_ms = (self._monotonic() - started) * 1000

        return {
            "symbol": spec.symbol,
            "price": price,
            "bid": bid,
            "ask": ask,
            "close": close,
            "volume": volume,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread_pct": spread_pct,
            "ob_imbalance": ob_imbalance,
            "_latency_ms": latency_ms,
        }

    async def place_order_async(self, symbol, action, quantity, prefer_maker: bool = True, force_market: bool = False):
        if not self.connected:
            return None

        side = action.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported action {action}; expected BUY or SELL")

        try:
            contract, spec = build_ib_contract(
                symbol,
                allowed_instrument_types=self.allowed_instrument_types,
                default_exchange=self.default_exchange,
                default_primary_exchange=self.default_primary_exchange,
                base_currency=self.base_currency,
                fx_exchange=self.fx_exchange,
            )
        except Exception as exc:
            logger.error(f"Error building IB contract for {symbol}: {exc}")
            return None

        client_order_id = f"{CLIENT_ORDER_PREFIX}-{int(time.time() * 1000)}"

        limit_price: float | None = None
        if not force_market:
            md = await self.get_market_data_async(spec.symbol)
            limit_price = self._compute_limit_price(md, side, prefer_maker)
            if limit_price is None:
                logger.error(f"Missing price data to place IB limit order for {symbol}")
                return None

        order = MarketOrder(side, quantity) if force_market else LimitOrder(side, quantity, limit_price)
        order.orderRef = client_order_id
        order.tif = getattr(order, "tif", None) or "DAY"

        try:
            trade = await asyncio.wait_for(self._place_order(contract, order), timeout=self.connect_timeout)
        except Exception as exc:
            logger.error(f"Error placing IB order for {symbol}: {exc}")
            return None

        status = await self._await_order_status(trade)

        return {
            "order_id": getattr(status, "orderId", None) if status else None,
            "status": self._normalize_order_status(status),
            "filled": getattr(status, "filled", None) if status else None,
            "remaining": getattr(status, "remaining", None) if status else None,
            "avg_fill_price": getattr(status, "avgFillPrice", None) if status else None,
            "liquidity": getattr(status, "liquidity", None) or getattr(status, "lastLiquidity", None) if status else None,
            "fee": getattr(status, "commission", None) if status else None,
            "client_order_id": client_order_id,
        }

    async def get_equity_async(self):
        if not self.connected:
            return 0.0

        try:
            account_values = await self._fetch_account_values()
            total, _ = await self._value_account_equity(account_values)
            return total
        except Exception as exc:
            logger.error(f"Error calculating IB equity: {exc}")
            return None

    async def get_positions_async(self):  # pragma: no cover - placeholder
        if not self.connected:
            return []

        try:
            entries = await self._fetch_portfolio()
        except Exception as exc:
            logger.error(f"Error fetching IB positions: {exc}")
            return []

        positions: list[dict[str, Any]] = []
        symbols: list[str] = []
        now_ms = int(time.time() * 1000)

        for entry in entries or []:
            contract = getattr(entry, "contract", None)
            symbol = self._contract_to_symbol(contract)
            if not symbol:
                continue
            qty_raw = getattr(entry, "position", 0)
            try:
                quantity = float(qty_raw)
            except (TypeError, ValueError):
                quantity = 0.0
            avg_cost_raw = getattr(entry, "avgCost", None)
            try:
                avg_cost = float(avg_cost_raw) if avg_cost_raw is not None else None
            except (TypeError, ValueError):
                avg_cost = None

            positions.append(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "avg_price": avg_cost,
                    "current_price": None,
                    "timestamp": now_ms,
                }
            )
            symbols.append(symbol)

        price_map = await self._fetch_prices(symbols)
        for pos in positions:
            if pos["symbol"] in price_map:
                pos["current_price"] = price_map[pos["symbol"]]

        return positions

    async def get_open_orders_async(self):  # pragma: no cover - placeholder
        raise NotImplementedError("Open orders not implemented for IB yet.")

    async def cancel_open_order_async(self, order_id):  # pragma: no cover - placeholder
        raise NotImplementedError("Order cancellation not implemented for IB yet.")

    async def get_my_trades_async(self, symbol: str, since: int = None, limit: int = None):  # pragma: no cover - placeholder
        raise NotImplementedError("Trade history not implemented for IB yet.")

    async def get_trades_from_timestamp(self, symbol: str, timestamp: int) -> list:  # pragma: no cover - placeholder
        raise NotImplementedError("Trade history not implemented for IB yet.")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> list:  # pragma: no cover - placeholder
        raise NotImplementedError("Historical data not implemented for IB yet.")

    async def _fetch_account_values(self):
        fetch_fn = getattr(self.ib, "accountValuesAsync", None)
        if fetch_fn:
            return await asyncio.wait_for(fetch_fn(), timeout=self.connect_timeout)
        return await asyncio.wait_for(asyncio.to_thread(self.ib.accountValues), timeout=self.connect_timeout)

    async def _value_account_equity(self, account_values) -> tuple[float, dict[str, float]]:
        """
        Compute total equity in base currency using NetLiquidation by currency.
        """
        per_currency: Dict[str, float] = {}
        tags = {"NetLiquidation"}
        fallback_tags = {"TotalCashValue"}

        # Prefer NetLiquidation, fallback to cash if absent
        for entry in account_values or []:
            tag = getattr(entry, "tag", "")
            if tag not in tags:
                continue
            currency = (getattr(entry, "currency", None) or self.base_currency).upper()
            try:
                value = float(getattr(entry, "value", 0))
            except (TypeError, ValueError):
                continue
            per_currency[currency] = per_currency.get(currency, 0.0) + value

        if not per_currency:
            for entry in account_values or []:
                tag = getattr(entry, "tag", "")
                if tag not in fallback_tags:
                    continue
                currency = (getattr(entry, "currency", None) or self.base_currency).upper()
                try:
                    value = float(getattr(entry, "value", 0))
                except (TypeError, ValueError):
                    continue
                per_currency[currency] = per_currency.get(currency, 0.0) + value

        total_base = 0.0
        fx_used: Dict[str, float] = {}
        for currency, amount in per_currency.items():
            if currency == self.base_currency:
                total_base += amount
                fx_used[currency] = 1.0
                continue
            rate = await self._get_fx_rate(currency)
            if rate is None:
                logger.warning(
                    f"Missing FX rate for {currency}->{self.base_currency}; skipping conversion."
                )
                continue
            fx_used[currency] = rate
            total_base += amount * rate

        return total_base, fx_used

    async def _get_fx_rate(self, currency: str) -> float | None:
        currency_upper = currency.upper()
        now = self._monotonic()

        cached = self._fx_quote_cache.get(currency_upper)
        if cached and now - cached[1] < self.fx_cache_ttl:
            return cached[0]

        rate = await self._fetch_fx_rate(currency_upper)
        if rate is not None:
            self._fx_quote_cache[currency_upper] = (rate, now)
        return rate

    async def _fetch_fx_rate(self, currency: str) -> float | None:
        """
        Fetch FX rate converting from `currency` to base_currency.
        """
        pair_primary = f"{currency}{self.base_currency}"
        rate = await self._request_fx_pair(pair_primary)
        if rate:
            return rate

        inverse_pair = f"{self.base_currency}{currency}"
        inverse_rate = await self._request_fx_pair(inverse_pair)
        if inverse_rate:
            return 1 / inverse_rate if inverse_rate != 0 else None
        return None

    async def _request_fx_pair(self, pair: str) -> float | None:
        contract = Forex(pair, exchange=self.fx_exchange)

        try:
            req_fn = getattr(self.ib, "reqMktDataAsync", None)
            if req_fn:
                ticker = await asyncio.wait_for(
                    req_fn(contract, "", False, False),
                    timeout=self.connect_timeout,
                )
            else:
                ticker = await asyncio.wait_for(
                    asyncio.to_thread(self.ib.reqMktData, contract, "", False, False),
                    timeout=self.connect_timeout,
                )
        except Exception as exc:
            logger.warning(f"FX quote request failed for {pair}: {exc}")
            return None

        price = None
        if ticker is None:
            return None

        price_fn = getattr(ticker, "marketPrice", None)
        if callable(price_fn):
            try:
                price = price_fn()
            except Exception:
                price = None

        if (price is None or price <= 0) and hasattr(ticker, "midpoint"):
            try:
                price = ticker.midpoint()
            except Exception:
                price = None

        if (price is None or price <= 0) and hasattr(ticker, "bid") and hasattr(ticker, "ask"):
            bid = getattr(ticker, "bid", None)
            ask = getattr(ticker, "ask", None)
            if bid and ask:
                price = (bid + ask) / 2

        if price is None or price <= 0:
            return None
        return price

    async def _fetch_portfolio(self):
        fetch_fn = getattr(self.ib, "portfolioAsync", None)
        if fetch_fn:
            return await asyncio.wait_for(fetch_fn(), timeout=self.connect_timeout)
        return await asyncio.wait_for(asyncio.to_thread(self.ib.portfolio), timeout=self.connect_timeout)

    def _contract_to_symbol(self, contract) -> str | None:
        if contract is None:
            return None
        pair_fn = getattr(contract, "pair", None)
        if callable(pair_fn):
            try:
                pair_val = pair_fn()
                if isinstance(pair_val, str) and len(pair_val) >= 6:
                    base = pair_val[:-3]
                    quote = pair_val[-3:]
                    return format_symbol(base, quote)
            except Exception:
                pass

        base = getattr(contract, "symbol", None) or getattr(contract, "localSymbol", None)
        quote = getattr(contract, "currency", None)
        if base and quote:
            try:
                return format_symbol(base, quote)
            except Exception:
                return None
        return None

    async def _fetch_prices(self, symbols: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                md = await self.get_market_data_async(symbol)
                if md and md.get("price") is not None:
                    prices[symbol] = md["price"]
            except Exception as exc:
                logger.warning(f"Price lookup failed for {symbol}: {exc}")
        return prices

    def _compute_limit_price(self, md: Optional[Dict[str, Any]], side: str, prefer_maker: bool) -> float | None:
        if not md:
            return None
        bid = md.get("bid")
        ask = md.get("ask")
        price = md.get("price") or md.get("close")

        if prefer_maker:
            if side == "BUY":
                base = bid or price
                return base * 0.999 if base else None
            base = ask or price
            return base * 1.001 if base else None

        if side == "BUY":
            return ask or price
        return bid or price

    async def _place_order(self, contract, order):
        place_fn = getattr(self.ib, "placeOrderAsync", None)
        if place_fn:
            return await place_fn(contract, order)
        return await asyncio.to_thread(self.ib.placeOrder, contract, order)

    async def _await_order_status(self, trade):
        deadline = self._monotonic() + self.order_wait_timeout
        status = getattr(trade, "orderStatus", None)
        while True:
            normalized = self._normalize_order_status(status)
            if normalized in {"filled", "canceled"}:
                return status
            remaining = getattr(status, "remaining", None)
            if remaining == 0:
                return status
            if self._monotonic() >= deadline:
                return status

            advance = getattr(trade, "advance_status", None)
            if callable(advance):
                advance()

            await asyncio.sleep(self.order_poll_interval)
            status = getattr(trade, "orderStatus", None) or status

    @staticmethod
    def _normalize_order_status(status) -> str:
        if status is None:
            return "unknown"
        raw = (getattr(status, "status", None) or "").lower()
        if raw in {"filled"}:
            return "filled"
        if raw in {"cancelled", "canceled", "inactive"}:
            return "canceled"
        if raw in {"submitted", "pre-submitted", "presubmitted"}:
            remaining = getattr(status, "remaining", None)
            if remaining == 0:
                return "filled"
            return "open"
        if raw in {"api pending", "pendingcancel", "pendingsubmit"}:
            return "pending"
        return raw or "unknown"
