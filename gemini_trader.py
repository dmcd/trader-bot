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
    # Sandbox starting balances
    SANDBOX_STARTING_USD = 100000.0
    SANDBOX_STARTING_BTC = 1000.0
    
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

    async def place_order_async(self, symbol, action, quantity, prefer_maker=True):
        """Places a limit order. Gemini requires limit orders; prefer maker when requested, fallback to taker."""
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

            bid = ticker.get('bid') or ticker.get('last')
            ask = ticker.get('ask') or ticker.get('last')

            def compute_price(post_only: bool):
                if post_only:
                    # Nudge price inside the spread to stay maker; small tick buffer to avoid taker
                    if side == 'buy':
                        base_price = bid or ticker.get('last')
                        return base_price * 0.9995 if base_price else ticker.get('last')
                    else:
                        base_price = ask or ticker.get('last')
                        return base_price * 1.0005 if base_price else ticker.get('last')
                else:
                    # Taker-style limit-at-touch
                    return ask if side == 'buy' else bid

            limit_price = compute_price(prefer_maker)

            if limit_price is None:
                raise ValueError(f"Ticker for {symbol} missing price data: {ticker}")

            # Use ccxt's precision methods to ensure correct formatting
            # These methods use the exchange's market precision data
            limit_price = self.exchange.price_to_precision(symbol, limit_price)
            quantity = self.exchange.amount_to_precision(symbol, quantity)

            logger.info(f"Placing limit order: {quantity} at ${limit_price}")

            # Create limit order
            params = {}
            if prefer_maker:
                params['postOnly'] = True

            liquidity = "maker_intent" if prefer_maker else "taker"
            try:
                order = await self.exchange.create_limit_order(symbol, side, quantity, limit_price, params)
                # Detect maker/taker if exchange reports it
                info = order.get('info', {}) if isinstance(order, dict) else {}
                reported = info.get('liquidity') or info.get('fillLiquidity')
                if reported:
                    liquidity = reported.lower()
                # If postOnly rejected, fallback to taker
                status = order.get('status')
                if prefer_maker and status in ('canceled', 'rejected'):
                    logger.warning(f"Post-only rejected ({status}), retrying as taker for {symbol}")
                    limit_price = compute_price(False)
                    order = await self.exchange.create_limit_order(symbol, side, quantity, limit_price)
                    liquidity = "taker"
            except Exception as maker_err:
                if prefer_maker:
                    logger.warning(f"Maker attempt failed ({maker_err}); retrying as taker")
                    limit_price = compute_price(False)
                    order = await self.exchange.create_limit_order(symbol, side, quantity, limit_price)
                    liquidity = "taker"
                else:
                    raise

            # Fetch order status to handle partial fills/timeouts
            try:
                order = await self.exchange.fetch_order(order['id'], symbol)
            except Exception as e:
                logger.warning(f"Could not fetch order status: {e}")

            return {
                'order_id': order.get('id'),
                'status': order.get('status'),
                'filled': order.get('filled', 0),
                'remaining': order.get('remaining', 0),
                'avg_fill_price': order.get('average'),
                'liquidity': liquidity,
                'fee': order.get('fee', {}).get('cost', 0.0) if order.get('fee') else 0.0
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
            total_usd, btc_price = await self._calculate_total_usd(balance)

            # In PAPER mode (sandbox), subtract starting balances to show only trading PnL
            # In LIVE mode, show actual total value
            if self.sandbox and btc_price > 0:
                starting_btc_value = self.SANDBOX_STARTING_BTC * btc_price
                starting_total = self.SANDBOX_STARTING_USD + starting_btc_value
                trading_pnl = total_usd - starting_total
                return trading_pnl
            else:
                return total_usd
        except Exception as e:
            logger.error(f"Error calculating Gemini PnL: {e}")
            return 0.0

    async def get_equity_async(self):
        """Returns total USD-equivalent account value (no sandbox adjustment)."""
        if not self.connected:
            return 0.0

        try:
            balance = await self.exchange.fetch_balance()
            total_usd, _ = await self._calculate_total_usd(balance)
            return total_usd
        except Exception as e:
            logger.error(f"Error calculating Gemini equity: {e}")
            return 0.0

    async def get_adjusted_btc_balance(self, btc_balance: float) -> float:
        """
        Returns the BTC balance adjusted for sandbox starting balance.
        In sandbox mode, subtracts the initial 1000 BTC to show only trading BTC.
        In live mode, returns the actual balance.
        """
        if self.sandbox:
            # Return max(0, balance - starting) to handle edge case where balance < starting
            return max(0.0, btc_balance - self.SANDBOX_STARTING_BTC)
        else:
            return btc_balance

    async def get_positions_async(self):
        """Return spot balances as positions."""
        if not self.connected:
            return []

        try:
            balance = await self.exchange.fetch_balance()
            positions = []
            for currency, total in balance.get('total', {}).items():
                if total and total != 0:
                    # Represent USD as USD/USD for consistency
                    symbol = f"{currency}/USD" if currency != 'USD' else 'USD'
                    positions.append({
                        'symbol': symbol,
                        'quantity': total,
                        'avg_price': None,
                        'timestamp': balance.get('timestamp')
                    })
            return positions
        except Exception as e:
            logger.error(f"Error fetching Gemini positions: {e}")
            return []

    async def get_open_orders_async(self):
        """Return open orders snapshot."""
        if not self.connected:
            return []

        try:
            orders = await self.exchange.fetch_open_orders()
            snapshot = []
            for o in orders:
                snapshot.append({
                    'order_id': o.get('id'),
                    'symbol': o.get('symbol'),
                    'side': o.get('side'),
                    'price': o.get('price'),
                    'amount': o.get('amount'),
                    'remaining': o.get('remaining'),
                    'status': o.get('status'),
                    'timestamp': o.get('timestamp')
                })
            return snapshot
        except Exception as e:
            logger.error(f"Error fetching Gemini open orders: {e}")
            return []

    async def _calculate_total_usd(self, balance: dict):
        """Helper to value holdings in USD; returns (total_usd, btc_price_used)."""
        total_usd = 0.0
        btc_price = 0.0

        if 'USD' in balance.get('total', {}):
            total_usd += balance['total']['USD']

        btc_qty = balance.get('total', {}).get('BTC', 0)
        if btc_qty and btc_qty > 0:
            ticker = await self.exchange.fetch_ticker('BTC/USD')
            btc_price = ticker.get('last') or 0.0
            total_usd += btc_qty * btc_price

        return total_usd, btc_price
    async def get_my_trades_async(self, symbol: str, since: int = None, limit: int = None):
        """Fetch past trades for a symbol."""
        if not self.connected:
            return []

        try:
            trades = await self.exchange.fetch_my_trades(symbol, since, limit)
            return trades
        except Exception as e:
            logger.error(f"Error fetching Gemini trades for {symbol}: {e}")
            return []
