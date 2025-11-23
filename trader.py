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
    async def get_pnl_async(self):
        raise NotImplementedError

    async def get_equity_async(self):
        """Return total account equity; default to PnL if not overridden."""
        return await self.get_pnl_async()

    @abstractmethod
    async def get_positions_async(self):
        raise NotImplementedError

    @abstractmethod
    async def get_open_orders_async(self):
        raise NotImplementedError
