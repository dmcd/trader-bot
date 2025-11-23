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


class TestServerTools(unittest.TestCase):
    def setUp(self):
        # Swap the real bot for a stub so tests do not hit IB
        self.original_bot = server.bot
        server.bot = StubBot()

    def tearDown(self):
        server.bot = self.original_bot

    def test_get_account_info(self):
        result = asyncio.run(server.get_account_info())
        self.assertTrue(server.bot.connected)
        self.assertEqual(result[0]["account"], "TEST")

    def test_get_stock_price(self):
        data = asyncio.run(server.get_stock_price("BHP"))
        self.assertEqual(data["symbol"], "BHP")
        self.assertEqual(data["price"], 123.45)

    def test_buy_and_sell_stock(self):
        buy = asyncio.run(server.buy_stock("BHP", 10))
        sell = asyncio.run(server.sell_stock("BHP", 5))
        self.assertEqual(buy["order_id"], "abc123")
        self.assertEqual(sell["remaining"], 5)


if __name__ == "__main__":
    unittest.main()
