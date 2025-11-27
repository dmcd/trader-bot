import asyncio
import ccxt.async_support as ccxt
import logging
from datetime import datetime, timedelta, timezone
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
            order_book = None
            try:
                order_book = await self.exchange.fetch_order_book(symbol, limit=5)
            except Exception as e:
                logger.warning(f"Order book fetch failed for {symbol}: {e}")

            bid = ticker.get('bid')
            ask = ticker.get('ask')
            price = ticker.get('last')
            volume = ticker.get('baseVolume') or ticker.get('quoteVolume')

            top_bid_size = top_ask_size = None
            spread_pct = None
            ob_imbalance = None
            if order_book:
                if order_book.get('bids'):
                    top_bid = order_book['bids'][0]
                    bid = bid or (top_bid[0] if top_bid else None)
                    top_bid_size = top_bid[1] if top_bid else None
                if order_book.get('asks'):
                    top_ask = order_book['asks'][0]
                    ask = ask or (top_ask[0] if top_ask else None)
                    top_ask_size = top_ask[1] if top_ask else None
                if bid and ask:
                    mid = (bid + ask) / 2
                    if mid:
                        spread_pct = ((ask - bid) / mid) * 100
                if top_bid_size and top_ask_size:
                    denom = top_bid_size + top_ask_size
                    if denom > 0:
                        ob_imbalance = (top_bid_size - top_ask_size) / denom

            return {
                'symbol': symbol,
                'price': price,
                'bid': bid,
                'ask': ask,
                'close': ticker.get('close'),
                'volume': volume,
                'bid_size': top_bid_size,
                'ask_size': top_ask_size,
                'spread_pct': spread_pct,
                'ob_imbalance': ob_imbalance,
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
            return None

    async def get_positions_async(self):
        """Return spot balances as positions."""
        if not self.connected:
            return []

        try:
            balance = await self.exchange.fetch_balance()
            positions = []
            
            # Sandbox filtering: ignore positions not traded in the last 24 hours
            cutoff_time = None
            if self.sandbox:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                logger.info(f"Sandbox mode: filtering positions older than {cutoff_time}")

            for currency, total in balance.get('total', {}).items():
                if total and total != 0:
                    # Represent USD as USD/USD for consistency
                    symbol = f"{currency}/USD" if currency != 'USD' else 'USD'
                    
                    # Apply sandbox filter
                    if self.sandbox and cutoff_time and symbol != 'USD':
                        try:
                            # Fetch last trade to check age
                            trades = await self.get_my_trades_async(symbol, limit=1)
                            if not trades:
                                logger.info(f"Sandbox: Ignoring {symbol} (no recent trades found)")
                                continue
                            
                            last_trade_ts = trades[0]['timestamp']
                            last_trade_time = datetime.fromtimestamp(last_trade_ts / 1000, timezone.utc)
                            
                            if last_trade_time < cutoff_time:
                                logger.info(f"Sandbox: Ignoring {symbol} (last trade {last_trade_time} < {cutoff_time})")
                                continue
                        except Exception as e:
                            logger.warning(f"Sandbox: Could not verify age of {symbol}, keeping it. Error: {e}")

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

    async def cancel_open_order_async(self, order_id):
        """Cancel a single open order by ID."""
        if not self.connected or not order_id:
            return False

        try:
            await self.exchange.cancel_order(order_id)
            logger.info(f"Cancelled Gemini order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling Gemini order {order_id}: {e}")
            return False

    async def _calculate_total_usd(self, balance: dict):
        """Helper to value holdings in USD; returns (total_usd, price_map)."""
        total_usd = 0.0
        price_map = {}
        totals = balance.get('total', {}) or {}

        usd_val = totals.get('USD', 0) or 0
        total_usd += usd_val

        for currency, qty in totals.items():
            if currency == 'USD' or not qty or qty == 0:
                continue
            price = None
            symbol_direct = f"{currency}/USD"
            symbol_invert = f"USD/{currency}"
            try:
                if symbol_direct in self.exchange.markets:
                    ticker = await self.exchange.fetch_ticker(symbol_direct)
                    price = ticker.get('last') or ticker.get('close')
                elif symbol_invert in self.exchange.markets:
                    ticker = await self.exchange.fetch_ticker(symbol_invert)
                    px = ticker.get('last') or ticker.get('close')
                    price = (1 / px) if px else None
                if price:
                    price_map[currency] = price
                    total_usd += qty * price
                else:
                    logger.debug(f"Could not value {currency} (no USD market)")
            except Exception as e:
                logger.warning(f"Error valuing {currency}: {e}")

        return total_usd, price_map
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

    async def get_trades_from_timestamp(self, symbol: str, timestamp: int) -> list:
        """
        Fetch trades since a specific timestamp (ms).
        Handles pagination if necessary (though fetch_my_trades usually returns enough for daily).
        """
        if not self.connected:
            return []
        
        try:
            # Gemini/CCXT fetch_my_trades 'since' is in milliseconds
            trades = await self.exchange.fetch_my_trades(symbol, since=timestamp)
            return trades
        except Exception as e:
            logger.error(f"Error fetching trades since {timestamp}: {e}")
            return []

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> list:
        """Fetch OHLCV data for technical analysis."""
        if not self.connected:
            return []

        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            # Convert to list of dicts for easier consumption
            # CCXT structure: [timestamp, open, high, low, close, volume]
            formatted_data = []
            for candle in ohlcv:
                formatted_data.append({
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'price': candle[4], # TA expects 'price'
                    'volume': candle[5]
                })
            return formatted_data
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return []
