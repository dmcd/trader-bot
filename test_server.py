import asyncio
import unittest

import server


class StubBot:
    def __init__(self):
        self.connected = False

    async def connect_async(self):
        self.connected = True

    async def close(self):
        self.connected = False

    async def get_account_summary_async(self):
        return [{"account": "TEST", "tag": "Cash", "value": "1000", "currency": "USD"}]

    async def get_market_data_async(self, symbol):
        return {"symbol": symbol, "price": 123.45, "bid": 123.0, "ask": 124.0, "close": 122.0}

    async def place_order_async(self, symbol, action, quantity):
        return {"order_id": "abc123", "status": "Submitted", "filled": 0, "remaining": quantity, "avg_fill_price": None}

if __name__ == "__main__":
    unittest.main()
