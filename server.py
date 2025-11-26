import logging
from typing import Any, Dict

from trader import BaseTrader

logger = logging.getLogger(__name__)

# Global bot instance; tests swap this with a stub.
bot: BaseTrader | Any = None


async def _ensure_bot_connected() -> None:
    """Connect the bot once per process; no-op if already connected."""
    if bot is None:
        raise RuntimeError("Bot is not configured")
    connect_fn = getattr(bot, "connect_async", None)
    if connect_fn:
        await connect_fn()


async def get_account_info() -> Any:
    await _ensure_bot_connected()
    return await bot.get_account_summary_async()


async def get_stock_price(symbol: str) -> Dict[str, Any]:
    await _ensure_bot_connected()
    return await bot.get_market_data_async(symbol)


async def buy_stock(symbol: str, quantity: float) -> Dict[str, Any]:
    await _ensure_bot_connected()
    return await bot.place_order_async(symbol, "BUY", quantity)


async def sell_stock(symbol: str, quantity: float) -> Dict[str, Any]:
    await _ensure_bot_connected()
    return await bot.place_order_async(symbol, "SELL", quantity)
