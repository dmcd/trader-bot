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

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '') # LLM Key
GEMINI_EXCHANGE_API_KEY = os.getenv('GEMINI_EXCHANGE_API_KEY', '') # Trading Key
GEMINI_EXCHANGE_SECRET = os.getenv('GEMINI_EXCHANGE_SECRET', '') # Trading Secret
GEMINI_SANDBOX_API_KEY = os.getenv('GEMINI_SANDBOX_API_KEY', '') # Sandbox Key
GEMINI_SANDBOX_SECRET = os.getenv('GEMINI_SANDBOX_SECRET', '') # Sandbox Secret

# Exchange Selection
# Options: 'IB' or 'GEMINI'
ACTIVE_EXCHANGE = os.getenv('ACTIVE_EXCHANGE', 'IB').upper()

