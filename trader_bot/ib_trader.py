"""Interactive Brokers adapter scaffolding."""

import asyncio
import logging
import time
from typing import Any, Callable, Optional

from ib_insync import IB

from trader_bot.config import (
    IB_ACCOUNT_ID,
    IB_CLIENT_ID,
    IB_HOST,
    IB_PAPER,
    IB_PORT,
)
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
        self.ib: IB = ib_client or IB()
        self.connected = False
        self._last_ping_mono: float | None = None
        self._monotonic = monotonic or time.monotonic

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

    async def get_account_summary_async(self) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError("Account summary not implemented for IB yet.")

    async def get_market_data_async(self, symbol):  # pragma: no cover - placeholder
        raise NotImplementedError("Market data not implemented for IB yet.")

    async def place_order_async(self, symbol, action, quantity):  # pragma: no cover - placeholder
        raise NotImplementedError("Order placement not implemented for IB yet.")

    async def get_equity_async(self):  # pragma: no cover - placeholder
        raise NotImplementedError("Equity retrieval not implemented for IB yet.")

    async def get_positions_async(self):  # pragma: no cover - placeholder
        raise NotImplementedError("Positions not implemented for IB yet.")

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
