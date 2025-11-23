import asyncio
from ib_insync import *
import logging
from abc import ABC, abstractmethod

from config import IB_HOST, IB_PORT, IB_CLIENT_ID
import random

# Configure logging

logger = logging.getLogger(__name__)

class BaseTrader(ABC):
    @abstractmethod
    async def connect_async(self):
        pass

    @abstractmethod
    async def get_account_summary_async(self):
        pass

    @abstractmethod
    async def get_market_data_async(self, symbol):
        pass

    @abstractmethod
    async def place_order_async(self, symbol, action, quantity):
        pass
        
    @abstractmethod
    async def get_pnl_async(self):
        pass

    @abstractmethod
    async def get_positions_async(self):
        pass

    @abstractmethod
    async def get_open_orders_async(self):
        pass

    @abstractmethod
    async def get_positions_async(self):
        """Return current open positions."""
        pass

    @abstractmethod
    async def get_open_orders_async(self):
        """Return current open orders."""
        pass

    @abstractmethod
    async def get_positions_async(self):
        """Return current positions/open balances."""
        pass

    @abstractmethod
    async def get_open_orders_async(self):
        """Return current open orders."""
        pass
