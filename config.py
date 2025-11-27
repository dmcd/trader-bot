import os
from dotenv import load_dotenv

load_dotenv()

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'PAPER') # PAPER or LIVE

# Connection Settings


# Risk Management Limits
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '1000.0')) # absolute currency stop
MAX_DAILY_LOSS_PERCENT = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '3.0')) # percent of equity stop
MAX_ORDER_VALUE = float(os.getenv('MAX_ORDER_VALUE', '5000.0')) # currency depends on venue
ORDER_VALUE_BUFFER = float(os.getenv('ORDER_VALUE_BUFFER', '5.0')) # trim near-cap trades by this buffer
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', '10000.0')) # Total portfolio exposure limit
MIN_TRADE_SIZE = float(os.getenv('MIN_TRADE_SIZE', '500.0')) # Minimum trade size in currency
MIN_RR = float(os.getenv('MIN_RR', '1.2'))  # Minimum risk/reward ratio when stop/target are present
MAX_SLIPPAGE_PCT = float(os.getenv('MAX_SLIPPAGE_PCT', '0.5'))  # Max price move % allowed between decision and execution
HIGH_VOL_SIZE_FACTOR = float(os.getenv('HIGH_VOL_SIZE_FACTOR', '0.6'))  # Scale size in high vol regimes
MED_VOL_SIZE_FACTOR = float(os.getenv('MED_VOL_SIZE_FACTOR', '0.8'))  # Scale size in medium vol regimes

# Cadence & spacing
LOOP_INTERVAL_SECONDS = int(os.getenv('LOOP_INTERVAL_SECONDS', '10'))  # main loop sleep (default 5 min)
MIN_TRADE_INTERVAL_SECONDS = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '120'))  # min spacing between trades
FEE_RATIO_COOLDOWN = float(os.getenv('FEE_RATIO_COOLDOWN', '50.0'))  # if fees > X% of gross PnL, pause trading
PRIORITY_MOVE_PCT = float(os.getenv('PRIORITY_MOVE_PCT', '1.5'))  # % move over short window to break cooldown
PRIORITY_LOOKBACK_MIN = int(os.getenv('PRIORITY_LOOKBACK_MIN', '5'))  # minutes to measure move
BREAK_GLASS_COOLDOWN_MIN = int(os.getenv('BREAK_GLASS_COOLDOWN_MIN', '60'))  # min between break-glass uses
BREAK_GLASS_SIZE_FACTOR = float(os.getenv('BREAK_GLASS_SIZE_FACTOR', '0.6'))  # reduce size on break-glass trades

# Market microstructure guards
MAX_SPREAD_PCT = float(os.getenv('MAX_SPREAD_PCT', '0.20'))  # Skip trading if spread exceeds this % of mid
MIN_TOP_OF_BOOK_NOTIONAL = float(os.getenv('MIN_TOP_OF_BOOK_NOTIONAL', '100.0'))  # Require at least this notional at best bid/ask

# LLM tool/data fetch limits
TOOL_MAX_BARS = int(os.getenv('TOOL_MAX_BARS', '2000'))  # cap per-timeframe bars returned to the LLM
TOOL_MAX_TRADES = int(os.getenv('TOOL_MAX_TRADES', '500'))  # cap recent trades returned
TOOL_MAX_DEPTH = int(os.getenv('TOOL_MAX_DEPTH', '200'))  # cap order book depth levels
TOOL_DEFAULT_TIMEFRAMES = os.getenv('TOOL_DEFAULT_TIMEFRAMES', '1m,5m,15m,1h,6h,1d').split(',')
# Allowed/whitelisted timeframes for market data fetches. Values not in this list
# will be dropped during tool request normalization.
TOOL_ALLOWED_TIMEFRAMES = os.getenv('TOOL_ALLOWED_TIMEFRAMES', '1m,5m,15m,30m,1h,6h,1d').split(',')
TOOL_MAX_JSON_BYTES = int(os.getenv('TOOL_MAX_JSON_BYTES', '200000'))  # fail-safe cap on JSON payload size
TOOL_CACHE_TTL_SECONDS = int(os.getenv('TOOL_CACHE_TTL_SECONDS', '5'))  # reuse fresh fetches within this window
TOOL_ALLOWED_SYMBOLS = [s.strip().upper() for s in os.getenv('TOOL_ALLOWED_SYMBOLS', '*').split(',') if s.strip()]
TOOL_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('TOOL_RATE_LIMIT_WINDOW_SECONDS', '60'))
TOOL_RATE_LIMIT_MARKET_DATA = int(os.getenv('TOOL_RATE_LIMIT_MARKET_DATA', '12'))  # per window
TOOL_RATE_LIMIT_ORDER_BOOK = int(os.getenv('TOOL_RATE_LIMIT_ORDER_BOOK', '12'))
TOOL_RATE_LIMIT_RECENT_TRADES = int(os.getenv('TOOL_RATE_LIMIT_RECENT_TRADES', '12'))

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '') # LLM Key
GEMINI_EXCHANGE_API_KEY = os.getenv('GEMINI_EXCHANGE_API_KEY', '') # Trading Key
GEMINI_EXCHANGE_SECRET = os.getenv('GEMINI_EXCHANGE_SECRET', '') # Trading Secret
GEMINI_SANDBOX_API_KEY = os.getenv('GEMINI_SANDBOX_API_KEY', '') # Sandbox Key
GEMINI_SANDBOX_SECRET = os.getenv('GEMINI_SANDBOX_SECRET', '') # Sandbox Secret

# Exchange Selection
# Options: 'GEMINI'
ACTIVE_EXCHANGE = os.getenv('ACTIVE_EXCHANGE', 'GEMINI').upper()

# Exchange Fees
GEMINI_MAKER_FEE = float(os.getenv('GEMINI_MAKER_FEE', '0.0020'))  # 0.20%
GEMINI_TAKER_FEE = float(os.getenv('GEMINI_TAKER_FEE', '0.0040'))  # 0.40%


# LLM Costs (Gemini 2.5 Flash pricing per token)
GEMINI_INPUT_COST_PER_TOKEN = float(os.getenv('GEMINI_INPUT_COST', '0.000000075'))   # $0.075 per 1M tokens
GEMINI_OUTPUT_COST_PER_TOKEN = float(os.getenv('GEMINI_OUTPUT_COST', '0.00000030'))  # $0.30 per 1M tokens
