import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseTrader(ABC):
    @abstractmethod
    async def connect_async(self):
        """Establish connection to the venue."""
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        """Close connection and clean up resources."""
        raise NotImplementedError

    @abstractmethod
    async def get_account_summary_async(self):
        raise NotImplementedError

    @abstractmethod
    async def get_market_data_async(self, symbol):
        raise NotImplementedError

    @abstractmethod
    async def place_order_async(self, symbol, action, quantity):
        raise NotImplementedError

    @abstractmethod
    async def get_equity_async(self):
        """Return total account equity."""
        raise NotImplementedError

    @abstractmethod
    async def get_positions_async(self):
        raise NotImplementedError

    @abstractmethod
    async def get_open_orders_async(self):
        raise NotImplementedError

    @abstractmethod
    async def cancel_open_order_async(self, order_id):
        """Cancel a single open order by ID."""
        raise NotImplementedError

    @abstractmethod
    async def get_my_trades_async(self, symbol: str, since: int = None, limit: int = None):
        """Fetch recent trades."""
        raise NotImplementedError

    @abstractmethod
    async def get_trades_from_timestamp(self, symbol: str, timestamp: int) -> list:
        """Fetch trades since a timestamp (ms)."""
        raise NotImplementedError

    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> list:
        """Fetch historical candles."""
        raise NotImplementedError
