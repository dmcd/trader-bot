import os
from dotenv import load_dotenv

load_dotenv()

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'PAPER') # PAPER or LIVE

# Connection Settings
IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
IB_PORT = int(os.getenv('IB_PORT', '4002')) # 4002 for Paper, 7497 for TWS Live
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))

# Risk Management Limits
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '50.0')) # AUD (absolute)
MAX_DAILY_LOSS_PERCENT = float(os.getenv('MAX_DAILY_LOSS_PERCENT', '0.1')) # Percentage (0.1 = 0.1%)
MAX_ORDER_VALUE = float(os.getenv('MAX_ORDER_VALUE', '100.0')) # AUD
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))
MAX_TOTAL_EXPOSURE = float(os.getenv('MAX_TOTAL_EXPOSURE', '1000.0')) # Total portfolio exposure limit

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

