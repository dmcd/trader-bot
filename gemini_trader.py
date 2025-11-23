import asyncio
import ccxt.async_support as ccxt
import logging
from trader import BaseTrader
from config import (
    GEMINI_EXCHANGE_API_KEY, GEMINI_EXCHANGE_SECRET,
    GEMINI_SANDBOX_API_KEY, GEMINI_SANDBOX_SECRET,
    TRADING_MODE
)

logger = logging.getLogger(__name__)

class GeminiTrader(BaseTrader):
    def __init__(self):
        if TRADING_MODE == 'PAPER':
            self.api_key = GEMINI_SANDBOX_API_KEY
            self.secret = GEMINI_SANDBOX_SECRET
            self.sandbox = True
        else:
            self.api_key = GEMINI_EXCHANGE_API_KEY
            self.secret = GEMINI_EXCHANGE_SECRET
            self.sandbox = False

        self.exchange = None
        self.connected = False

    async def _populate_precisions(self):
        """
        Gemini's sandbox API omits precision data, which breaks ccxt's
        price/amount helpers. Pull precision from the production API as a
        fallback so we can still format orders correctly in PAPER mode.
        """
        if not self.exchange or not getattr(self.exchange, "markets", None):
            return

        missing = []
        for symbol, market in self.exchange.markets.items():
            precision = market.get('precision', {})
            if precision.get('price') is None or precision.get('amount') is None:
                missing.append(symbol)

        if not missing:
            return

        logger.warning(f"Gemini sandbox missing precision for: {', '.join(missing)}. Loading production precision metadata.")

        fallback_exchange = ccxt.gemini({'enableRateLimit': True})
        try:
            await fallback_exchange.load_markets()
            for symbol in missing:
                live_market = fallback_exchange.markets.get(symbol, {})
                live_precision = live_market.get('precision', {})
                # Use live precision when available, otherwise sensible defaults
                price_precision = live_precision.get('price', 0.01)
                amount_precision = live_precision.get('amount', 1e-8)

                market = self.exchange.markets[symbol]
                market_precision = market.setdefault('precision', {})
                market_precision['price'] = price_precision
                market_precision['amount'] = amount_precision

                # Carry over limits if sandbox omitted them
                if not market.get('limits') and live_market.get('limits'):
                    market['limits'] = live_market['limits']
        except Exception as e:
            logger.warning(f"Failed to load production precision data: {e}. Using default Gemini tick sizes.")
            for symbol in missing:
                market = self.exchange.markets[symbol]
                market_precision = market.setdefault('precision', {})
                market_precision['price'] = market_precision.get('price', 0.01)
                market_precision['amount'] = market_precision.get('amount', 1e-8)
        finally:
            try:
                await fallback_exchange.close()
            except Exception:
                pass

    async def connect_async(self):
        """Connects to Gemini Exchange."""
        if self.connected:
            return

        try:
            logger.info("Connecting to Gemini Exchange...")
            self.exchange = ccxt.gemini({
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True,
            })

            if self.sandbox:
                logger.info("Using Gemini Sandbox Environment")
                self.exchange.set_sandbox_mode(True)
                # Force correct Sandbox URL (must be a dict for ccxt)
                self.exchange.urls['api'] = {
                    'public': 'https://api.sandbox.gemini.com',
                    'private': 'https://api.sandbox.gemini.com',
                }
                logger.info(f"Gemini URLs: {self.exchange.urls}")

            # Test connection by loading markets
            logger.info("Loading markets...")
            await self.exchange.load_markets()
            await self._populate_precisions()
            self.connected = True
            logger.info("Connected to Gemini Exchange successfully!")
        except Exception as e:
            logger.error(f"Gemini Connection failed: {e}")
            self.connected = False

    async def close(self):
        if self.exchange:
            await self.exchange.close()
            self.connected = False

    async def get_account_summary_async(self):
        """Fetches account balance."""
        if not self.connected:
            return []

        try:
            balance = await self.exchange.fetch_balance()
            data = []
            # ccxt returns 'total' dictionary
            for currency, value in balance['total'].items():
                if value > 0:
                    data.append({
                        'account': 'Gemini',
                        'tag': 'Balance',
                        'value': value,
                        'currency': currency
                    })
            return data
        except Exception as e:
            logger.error(f"Error fetching Gemini balance: {e}")
            return []

    async def get_market_data_async(self, symbol):
        """Fetches ticker for a symbol (e.g., 'BTC/USD')."""
        if not self.connected:
            return None

        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'close': ticker['close']
            }
        except Exception as e:
            logger.error(f"Error fetching Gemini ticker for {symbol}: {e}")
            return None

    async def place_order_async(self, symbol, action, quantity):
        """Places a limit order (Gemini requires limit orders)."""
        if not self.connected:
            return None

        try:
            side = 'buy' if action == 'BUY' else 'sell'
            logger.info(f"Placing Gemini order: {side} {quantity} {symbol}")

            # Get current market price to set limit price
            ticker = await self.exchange.fetch_ticker(symbol)
            market = self.exchange.market(symbol)

            # Ensure precision exists (sandbox omits it)
            market_precision = market.setdefault('precision', {})
            if market_precision.get('price') is None:
                market_precision['price'] = 0.01
            if market_precision.get('amount') is None:
                market_precision['amount'] = 1e-8

            # For immediate execution (like market order):
            # - BUY: use ask price (willing to pay the current ask)
            # - SELL: use bid price (willing to accept the current bid)
            if side == 'buy':
                limit_price = ticker.get('ask') or ticker.get('last')
            else:
                limit_price = ticker.get('bid') or ticker.get('last')

            if limit_price is None:
                raise ValueError(f"Ticker for {symbol} missing price data: {ticker}")

            # Use ccxt's precision methods to ensure correct formatting
            # These methods use the exchange's market precision data
            limit_price = self.exchange.price_to_precision(symbol, limit_price)
            quantity = self.exchange.amount_to_precision(symbol, quantity)

            logger.info(f"Placing limit order: {quantity} at ${limit_price}")

            # Create limit order
            order = await self.exchange.create_limit_order(symbol, side, quantity, limit_price)

            return {
                'order_id': order['id'],
                'status': order['status'],
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', 0),
                'avg_fill_price': order.get('average')
            }
        except Exception as e:
            logger.error(f"Error placing Gemini order: {e}")
            return None

    async def get_pnl_async(self):
        """Calculates total USD value of assets, minus starting balance in PAPER mode."""
        if not self.connected:
            return 0.0

        try:
            balance = await self.exchange.fetch_balance()
            total_usd = 0.0

            # Simple estimation: sum of (balance * current_price)
            # Note: This is expensive if we have many coins.
            # For now, let's just check USD and BTC

            if 'USD' in balance['total']:
                total_usd += balance['total']['USD']

            # Check BTC
            btc_price = 0
            if 'BTC' in balance['total'] and balance['total']['BTC'] > 0:
                ticker = await self.exchange.fetch_ticker('BTC/USD')
                btc_price = ticker['last']
                total_usd += balance['total']['BTC'] * btc_price

            # In PAPER mode (sandbox), subtract starting balances to show only trading PnL
            # Sandbox starts with: $100,000 USD + 1000 BTC
            # In LIVE mode, show actual total value
            if self.sandbox and btc_price > 0:
                starting_usd = 100000.0
                starting_btc_value = 1000 * btc_price
                starting_total = starting_usd + starting_btc_value
                trading_pnl = total_usd - starting_total
                return trading_pnl
            else:
                return total_usd
        except Exception as e:
            logger.error(f"Error calculating Gemini PnL: {e}")
            return 0.0
