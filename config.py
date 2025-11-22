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
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '50.0')) # AUD
MAX_ORDER_VALUE = float(os.getenv('MAX_ORDER_VALUE', '100.0')) # AUD
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '3'))

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
