import asyncio
from ib_insync import *
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TraderBot:
    def __init__(self, host='127.0.0.1', port=4002, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.connected = False

    async def connect_async(self):
        """Connects to the IB Gateway or TWS asynchronously."""
        try:
            logger.info(f"Attempting to connect asynchronously to {self.host}:{self.port} with clientId={self.client_id}...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info("Connected successfully!")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            raise

    def connect(self):
        """Connects to the IB Gateway or TWS."""
        try:
            logger.info(f"Attempting to connect to {self.host}:{self.port} with clientId={self.client_id}...")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info("Connected successfully!")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            raise

    def disconnect(self):
        """Disconnects from the IB Gateway or TWS."""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected.")

    async def get_account_summary_async(self):
        """Fetches the account summary asynchronously."""
        if not self.connected:
            logger.warning("Not connected. Cannot fetch account summary.")
            return []

        # Use client.reqAccountUpdates to avoid IB's blocking wrapper
        # client.reqAccountUpdates(subscribe, acctCode)
        self.ib.client.reqAccountUpdates(True, '')
        
        # Wait a bit for data to populate
        await asyncio.sleep(0.5)
        
        # Read from the local cache (IB object updates this from client callbacks)
        summary = self.ib.accountValues()
        
        data = []
        for item in summary:
            if item.currency == 'AUD': # Filter for AUD to match previous behavior or just return all
                 data.append({
                    'account': item.account,
                    'tag': item.tag,
                    'value': item.value,
                    'currency': item.currency
                })
        return data

    async def get_market_data_async(self, symbol, exchange='SMART', currency='AUD'):
        """Fetches current market data for a stock asynchronously."""
        if not self.connected:
            logger.warning("Not connected. Cannot fetch market data.")
            return None

        contract = Stock(symbol, exchange, currency)
        await self.ib.qualifyContractsAsync(contract)
        
        logger.info(f"Requesting market data for {symbol}...")
        # reqMktData with snapshot=False is non-blocking
        self.ib.reqMktData(contract, '', False, False)
        
        # Wait a brief moment for data to populate
        await asyncio.sleep(2)
        
        ticker = self.ib.ticker(contract)
        price = ticker.marketPrice()
        logger.info(f"Price for {symbol}: {price}")
        
        return {
            'symbol': symbol,
            'price': price,
            'bid': ticker.bid,
            'ask': ticker.ask,
            'close': ticker.close
        }

    async def place_order_async(self, symbol, action, quantity, order_type='MKT', exchange='SMART', currency='AUD'):
        """Places an order asynchronously."""
        if not self.connected:
            logger.warning("Not connected. Cannot place order.")
            return None

        contract = Stock(symbol, exchange, currency)
        await self.ib.qualifyContractsAsync(contract)

        if order_type == 'MKT':
            order = MarketOrder(action, quantity)
        else:
            logger.warning(f"Order type {order_type} not implemented yet.")
            return None

        trade = self.ib.placeOrder(contract, order)
        logger.info(f"Order placed: {action} {quantity} {symbol}")
        
        # Wait for order status to update
        await asyncio.sleep(1)
        
        return {
            'order_id': trade.order.orderId,
            'status': trade.orderStatus.status,
            'filled': trade.orderStatus.filled,
            'remaining': trade.orderStatus.remaining,
            'avg_fill_price': trade.orderStatus.avgFillPrice
        }

    def run(self):
        """Keeps the script running to maintain connection (for standalone testing)."""
        try:
            self.ib.run()
        except KeyboardInterrupt:
            self.disconnect()

if __name__ == "__main__":
    # Standalone test
    bot = TraderBot()
    try:
        bot.connect()
        
        # Test Account Summary
        print("\n--- Account Summary ---")
        summary = bot.get_account_summary()
        for item in summary:
            if item['tag'] == 'TotalCashValue' and item['currency'] == 'AUD':
                print(f"{item['tag']}: {item['value']} {item['currency']}")

        # Test Market Data
        print("\n--- Market Data ---")
        price_data = bot.get_market_data('BHP')
        print(price_data)

        # Test Order (Paper Trading Only! Be careful)
        # print("\n--- Placing Order ---")
        # order_result = bot.place_order('BHP', 'BUY', 1)
        # print(order_result)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        bot.disconnect()
