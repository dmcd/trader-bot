import asyncio
import logging
from mcp.server.fastmcp import FastMCP
from ib_trader import IBTrader as TraderBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize TraderBot
# We'll initialize it globally for now, but in a real app we might want better lifecycle management
bot = TraderBot()

# Initialize FastMCP Server
mcp = FastMCP("TraderBot")

@mcp.tool()
async def get_account_info() -> list[dict]:
    """
    Retrieves the current account summary, including cash balance and currency.
    Returns a list of dictionaries containing account details.
    """
    if not bot.connected:
        try:
            await bot.connect_async()
        except Exception as e:
            return [{"error": f"Failed to connect: {str(e)}"}]
            
    return await bot.get_account_summary_async()

@mcp.tool()
async def get_stock_price(symbol: str) -> dict:
    """
    Retrieves the current market price for a specific stock symbol.
    Args:
        symbol: The stock ticker symbol (e.g., 'BHP', 'AAPL').
    Returns:
        A dictionary with price information (price, bid, ask, close).
    """
    if not bot.connected:
        try:
            await bot.connect_async()
        except Exception as e:
            return {"error": f"Failed to connect: {str(e)}"}

    data = await bot.get_market_data_async(symbol)
    if data is None:
        return {"error": "Failed to retrieve market data. Check symbol or connection."}
    return data

@mcp.tool()
async def buy_stock(symbol: str, quantity: int) -> dict:
    """
    Places a market BUY order for the specified stock.
    Args:
        symbol: The stock ticker symbol.
        quantity: The number of shares to buy.
    Returns:
        A dictionary with order status and details.
    """
    if not bot.connected:
        try:
            await bot.connect_async()
        except Exception as e:
            return {"error": f"Failed to connect: {str(e)}"}

    return await bot.place_order_async(symbol, 'BUY', quantity)

@mcp.tool()
async def sell_stock(symbol: str, quantity: int) -> dict:
    """
    Places a market SELL order for the specified stock.
    Args:
        symbol: The stock ticker symbol.
        quantity: The number of shares to sell.
    Returns:
        A dictionary with order status and details.
    """
    if not bot.connected:
        try:
            await bot.connect_async()
        except Exception as e:
            return {"error": f"Failed to connect: {str(e)}"}

    return await bot.place_order_async(symbol, 'SELL', quantity)

@mcp.resource("trading://portfolio")
async def get_portfolio() -> str:
    """
    Returns a JSON representation of the current portfolio holdings.
    """
    if not bot.connected:
        try:
            await bot.connect_async()
        except Exception:
            return "Error: Not connected"

    # In a real implementation, we would fetch positions from IB
    # For now, we'll return a placeholder or fetch account summary as a proxy
    summary = await bot.get_account_summary_async()
    return str(summary)

if __name__ == "__main__":
    # Connect the bot when the server starts
    # We can't await here easily in top-level code without a loop, 
    # but FastMCP will handle the loop.
    # We can rely on lazy connection in the tools, or use a startup event if FastMCP supports it.
    # FastMCP doesn't have explicit startup events in the simple API, so lazy connection is fine.
    
    # Run the MCP server
    mcp.run()
