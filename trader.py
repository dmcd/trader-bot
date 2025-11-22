import asyncio
from ib_insync import *
import logging
from abc import ABC, abstractmethod

from config import IB_HOST, IB_PORT, IB_CLIENT_ID
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


