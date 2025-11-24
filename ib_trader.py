import asyncio
from ib_insync import *
import logging
from trader import BaseTrader
from config import IB_HOST, IB_PORT, IB_CLIENT_ID
import random

# Configure logging
logger = logging.getLogger(__name__)

class IBTrader(BaseTrader):
    def __init__(self, host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID):
        self.host = host
        self.port = port
        # If client_id is the default 1, randomize it to avoid conflicts during testing
        if client_id == 1:
            self.client_id = random.randint(1000, 9999)
        else:
            self.client_id = client_id
        self.ib = IB()
        self.connected = False

    async def connect_async(self):
        """Connects to the IB Gateway asynchronously."""
        if self.connected:
            return
        
        try:
            logger.info(f"Attempting to connect to {self.host}:{self.port} with clientId={self.client_id}...")
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info("Connected successfully!")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            # Don't raise, just log, so the loop can retry or continue
            
    def disconnect(self):
        """Disconnects from the IB Gateway."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected.")

    async def close(self):
        """Async-friendly close used by strategy_runner cleanup."""
        self.disconnect()

    async def get_account_summary_async(self):
        """Fetches the account summary asynchronously."""
        if not self.connected:
            logger.warning("Not connected. Cannot fetch account summary.")
            return []

        # Request account updates (non-blocking)
        # True means subscribe
        self.ib.client.reqAccountUpdates(True, '')
        
        # Wait a brief moment for data to populate (in a real async app, we'd handle this differently)
        await asyncio.sleep(0.5)
        
        summary = self.ib.accountValues()
        # Convert to a friendly format
        data = []
        for item in summary:
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
            'close': ticker.close,
            'volume': None,
            'bid_size': None,
            'ask_size': None,
            'spread_pct': None,
            'ob_imbalance': None
        }

    async def place_order_async(self, symbol, action, quantity, order_type='MKT', exchange='SMART', currency='AUD', **kwargs):
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
        
        # Wait for order status updates; timeout after a few seconds
        try:
            await asyncio.wait_for(trade.fillEvent, timeout=5)
        except Exception:
            pass  # proceed with whatever status we have
        
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        remaining = trade.orderStatus.remaining
        avg_price = trade.orderStatus.avgFillPrice

        return {
            'order_id': trade.order.orderId,
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avg_fill_price': avg_price
        }

    async def get_equity_async(self):
        """Return IB Net Liquidation Value as total equity."""
        if not self.connected:
            return 0.0

        summary = self.ib.accountValues()
        net_liq = None
        cash = None
        for item in summary:
            if item.tag == 'NetLiquidation':
                try:
                    net_liq = float(item.value)
                except ValueError:
                    net_liq = None
            if item.tag == 'TotalCashValue':
                try:
                    cash = float(item.value)
                except ValueError:
                    cash = None
        return net_liq if net_liq is not None else (cash or 0.0)

    def run(self):
        """Keeps the script running to maintain connection (for standalone testing)."""
        self.ib.run()

    async def get_positions_async(self):
        """Fetch current IB positions."""
        if not self.connected:
            return []

        try:
            await self.ib.reqPositionsAsync()
            positions = []
            for pos in self.ib.positions():
                symbol = pos.contract.symbol
                positions.append({
                    'symbol': symbol,
                    'quantity': pos.position,
                    'avg_price': pos.avgCost,
                    'timestamp': None
                })
            return positions
        except Exception as e:
            logger.error(f"Error fetching IB positions: {e}")
            return []

    async def get_open_orders_async(self):
        """Fetch open orders."""
        if not self.connected:
            return []

        try:
            await self.ib.reqOpenOrdersAsync()
            snapshot = []
            for order in self.ib.openOrders():
                contract = order.contract
                snapshot.append({
                    'order_id': order.order.orderId,
                    'symbol': contract.symbol,
                    'side': order.order.action,
                    'price': order.order.lmtPrice,
                    'amount': order.order.totalQuantity,
                    'remaining': None,
                    'status': order.orderState.status,
                    'timestamp': None
                })
            return snapshot
        except Exception as e:
            logger.error(f"Error fetching IB open orders: {e}")
            return []

    async def cancel_open_order_async(self, order_id):
        """Cancel a single open order by ID."""
        if not self.connected or order_id is None:
            return False

        try:
            await self.ib.reqOpenOrdersAsync()
            for trade in self.ib.openOrders():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled IB order {order_id}")
                    return True
            logger.warning(f"IB order {order_id} not found among open orders")
            return False
        except Exception as e:
            logger.error(f"Error cancelling IB order {order_id}: {e}")
            return False

    async def get_trades_from_timestamp(self, symbol: str, timestamp: int) -> list:
        """Stub for interface compatibility."""
        logger.warning("get_trades_from_timestamp not implemented for IBTrader")
        return []

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> list:
        """Stub for interface compatibility."""
        logger.warning("fetch_ohlcv not implemented for IBTrader")
        return []

if __name__ == "__main__":
    # Simple test
    bot = IBTrader()
    try:
        # Since we are using async methods, we need an event loop
        # But for simple test, we can use the sync connect if we had one, or just run the loop
        # bot.connect() # We removed sync connect
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
