import os
from dotenv import load_dotenv

load_dotenv()

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'PAPER') # PAPER or LIVE

# Connection Settings
IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
IB_PORT = int(os.getenv('IB_PORT', '4002')) # 4002 for Paper, 7497 for TWS Live
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))

# Risk Management Limits (defaults: moderate tier)
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '500.0')) # absolute currency stop
MAX_DAILY_LOSS_PERCENT = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '3.0')) # percent of equity stop
MAX_ORDER_VALUE = float(os.getenv('MAX_ORDER_VALUE', '500.0')) # currency depends on venue
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', '1000.0')) # Total portfolio exposure limit
MIN_TRADE_SIZE = float(os.getenv('MIN_TRADE_SIZE', '200.0')) # Minimum trade size in currency

# Cadence & spacing
LOOP_INTERVAL_SECONDS = int(os.getenv('LOOP_INTERVAL_SECONDS', '300'))  # main loop sleep (default 5 min)
MIN_TRADE_INTERVAL_SECONDS = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '300'))  # min spacing between trades
FEE_RATIO_COOLDOWN = float(os.getenv('FEE_RATIO_COOLDOWN', '50.0'))  # if fees > X% of gross PnL, pause trading
PRIORITY_MOVE_PCT = float(os.getenv('PRIORITY_MOVE_PCT', '1.5'))  # % move over short window to break cooldown
PRIORITY_LOOKBACK_MIN = int(os.getenv('PRIORITY_LOOKBACK_MIN', '5'))  # minutes to measure move
BREAK_GLASS_COOLDOWN_MIN = int(os.getenv('BREAK_GLASS_COOLDOWN_MIN', '60'))  # min between break-glass uses
BREAK_GLASS_SIZE_FACTOR = float(os.getenv('BREAK_GLASS_SIZE_FACTOR', '0.6'))  # reduce size on break-glass trades

# Sizing tiers (override via env if needed)
SIZE_TIER = os.getenv('SIZE_TIER', 'MODERATE').upper()  # CONSERVATIVE, MODERATE, AGGRESSIVE
ORDER_SIZE_BY_TIER = {
    'CONSERVATIVE': float(os.getenv('ORDER_SIZE_CONSERVATIVE', '200.0')),
    'MODERATE': float(os.getenv('ORDER_SIZE_MODERATE', '500.0')),
    'AGGRESSIVE': float(os.getenv('ORDER_SIZE_AGGRESSIVE', '1000.0')),
}
DAILY_LOSS_PCT_BY_TIER = {
    'CONSERVATIVE': float(os.getenv('DAILY_LOSS_PCT_CONSERVATIVE', '5.0')),
    'MODERATE': float(os.getenv('DAILY_LOSS_PCT_MODERATE', '3.0')),
    'AGGRESSIVE': float(os.getenv('DAILY_LOSS_PCT_AGGRESSIVE', '2.5')),
}

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '') # LLM Key
GEMINI_EXCHANGE_API_KEY = os.getenv('GEMINI_EXCHANGE_API_KEY', '') # Trading Key
GEMINI_EXCHANGE_SECRET = os.getenv('GEMINI_EXCHANGE_SECRET', '') # Trading Secret
GEMINI_SANDBOX_API_KEY = os.getenv('GEMINI_SANDBOX_API_KEY', '') # Sandbox Key
GEMINI_SANDBOX_SECRET = os.getenv('GEMINI_SANDBOX_SECRET', '') # Sandbox Secret

# Exchange Selection
# Options: 'IB' or 'GEMINI'
ACTIVE_EXCHANGE = os.getenv('ACTIVE_EXCHANGE', 'IB').upper()

# Exchange Fees
GEMINI_MAKER_FEE = float(os.getenv('GEMINI_MAKER_FEE', '0.0020'))  # 0.20%
GEMINI_TAKER_FEE = float(os.getenv('GEMINI_TAKER_FEE', '0.0040'))  # 0.40%
IB_STOCK_FEE_PER_SHARE = float(os.getenv('IB_STOCK_FEE_PER_SHARE', '0.005'))  # $0.005 per share
IB_MIN_FEE = float(os.getenv('IB_MIN_FEE', '1.00'))  # $1 minimum

# LLM Costs (Gemini 2.5 Flash pricing per token)
GEMINI_INPUT_COST_PER_TOKEN = float(os.getenv('GEMINI_INPUT_COST', '0.000000075'))   # $0.075 per 1M tokens
GEMINI_OUTPUT_COST_PER_TOKEN = float(os.getenv('GEMINI_OUTPUT_COST', '0.00000030'))  # $0.30 per 1M tokens
