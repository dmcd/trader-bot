import os
from dotenv import load_dotenv

from trader_bot.symbols import normalize_symbols

load_dotenv()

# Trading Mode
TRADING_MODE = os.getenv('TRADING_MODE', 'PAPER') # PAPER or LIVE

# LLM Provider
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'GEMINI').upper()
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-2.5-flash')

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
EXCHANGE_ERROR_THRESHOLD = int(os.getenv('EXCHANGE_ERROR_THRESHOLD', '3'))  # consecutive exchange errors before auto-pause
EXCHANGE_PAUSE_SECONDS = int(os.getenv('EXCHANGE_PAUSE_SECONDS', '60'))  # pause duration after exchange circuit trips
TOOL_ERROR_THRESHOLD = int(os.getenv('TOOL_ERROR_THRESHOLD', '3'))  # consecutive tool failures before auto-pause
TOOL_PAUSE_SECONDS = int(os.getenv('TOOL_PAUSE_SECONDS', '60'))  # pause duration after tool circuit trips
TICKER_MAX_AGE_SECONDS = int(os.getenv('TICKER_MAX_AGE_SECONDS', '15'))  # stale ticker window
TICKER_MAX_LATENCY_MS = int(os.getenv('TICKER_MAX_LATENCY_MS', '5000'))  # warn/skip when ticker fetch latency exceeds this
TOOL_DATA_MAX_AGE_SECONDS = int(os.getenv('TOOL_DATA_MAX_AGE_SECONDS', '60'))  # mark tool payloads stale beyond this age

# Cadence & spacing
LOOP_INTERVAL_SECONDS = int(os.getenv('LOOP_INTERVAL_SECONDS', '300'))  # main loop sleep (default 5 min)
MIN_TRADE_INTERVAL_SECONDS = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '120'))  # min spacing between trades
FEE_RATIO_COOLDOWN = float(os.getenv('FEE_RATIO_COOLDOWN', '50.0'))  # if fees > X% of gross PnL, pause trading
PRIORITY_MOVE_PCT = float(os.getenv('PRIORITY_MOVE_PCT', '1.5'))  # % move over short window to break cooldown
PRIORITY_LOOKBACK_MIN = int(os.getenv('PRIORITY_LOOKBACK_MIN', '5'))  # minutes to measure move
BREAK_GLASS_COOLDOWN_MIN = int(os.getenv('BREAK_GLASS_COOLDOWN_MIN', '60'))  # min between break-glass uses
BREAK_GLASS_SIZE_FACTOR = float(os.getenv('BREAK_GLASS_SIZE_FACTOR', '0.6'))  # reduce size on break-glass trades
OHLCV_MIN_CAPTURE_SPACING_SECONDS = int(os.getenv('OHLCV_MIN_CAPTURE_SPACING_SECONDS', '60'))  # min spacing per timeframe fetch
OHLCV_MAX_ROWS_PER_TIMEFRAME = int(os.getenv('OHLCV_MAX_ROWS_PER_TIMEFRAME', '1000'))  # retention per symbol/timeframe
MARKET_DATA_RETENTION_MINUTES = int(os.getenv('MARKET_DATA_RETENTION_MINUTES', '720'))  # minutes of market snapshots to keep per session
LLM_TRACE_RETENTION_DAYS = int(os.getenv('LLM_TRACE_RETENTION_DAYS', '7'))  # days of full prompt/response traces to retain
COMMAND_RETENTION_DAYS = int(os.getenv('COMMAND_RETENTION_DAYS', '7'))  # days to keep executed/cancelled commands
DASHBOARD_REFRESH_SECONDS = int(os.getenv('DASHBOARD_REFRESH_SECONDS', '5'))  # Streamlit auto-refresh cadence
SANDBOX_IGNORE_INITIAL_POSITIONS = os.getenv('SANDBOX_IGNORE_INITIAL_POSITIONS', 'true').lower() == 'true'  # Hide sandbox airdrop inventory from exposure calcs/prompts

# Market microstructure guards
MAX_SPREAD_PCT = float(os.getenv('MAX_SPREAD_PCT', '0.20'))  # Skip trading if spread exceeds this % of mid
MIN_TOP_OF_BOOK_NOTIONAL = float(os.getenv('MIN_TOP_OF_BOOK_NOTIONAL', '100.0'))  # Require at least this notional at best bid/ask

# Exchange selection and symbol allowlist
# Options: 'GEMINI'
ACTIVE_EXCHANGE = os.getenv('ACTIVE_EXCHANGE', 'GEMINI').upper()
# Comma-separated symbols to trade/monitor and allow for tool access; preserves order
_ALLOWED_SYMBOLS_RAW = os.getenv('ALLOWED_SYMBOLS', 'BTC/USD')
ALLOWED_SYMBOLS = normalize_symbols(_ALLOWED_SYMBOLS_RAW)

# Interactive Brokers settings
IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
IB_PORT = int(os.getenv('IB_PORT', '7497'))
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))
IB_ACCOUNT_ID = os.getenv('IB_ACCOUNT_ID', '')
IB_PAPER = os.getenv('IB_PAPER', 'true').lower() == 'true'
IB_BASE_CURRENCY = os.getenv('IB_BASE_CURRENCY', 'AUD').upper()
IB_EXCHANGE = os.getenv('IB_EXCHANGE', 'SMART')
IB_PRIMARY_EXCHANGE = os.getenv('IB_PRIMARY_EXCHANGE', 'ASX')

def _parse_ib_allowed_types(raw: str):
    return [token.strip().upper() for token in raw.split(',') if token.strip()]

IB_ALLOWED_INSTRUMENT_TYPES = _parse_ib_allowed_types(os.getenv('IB_ALLOWED_INSTRUMENT_TYPES', 'STK,FX'))
IB_STOCK_COMMISSION_PER_SHARE = float(os.getenv('IB_STOCK_COMMISSION_PER_SHARE', '0.005'))  # AUD/USD depending on listing
IB_STOCK_MIN_COMMISSION = float(os.getenv('IB_STOCK_MIN_COMMISSION', '1.0'))
IB_FX_COMMISSION_PCT = float(os.getenv('IB_FX_COMMISSION_PCT', '0.0'))

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
TOOL_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv('TOOL_RATE_LIMIT_WINDOW_SECONDS', '60'))
TOOL_RATE_LIMIT_MARKET_DATA = int(os.getenv('TOOL_RATE_LIMIT_MARKET_DATA', '12'))  # per window
TOOL_RATE_LIMIT_ORDER_BOOK = int(os.getenv('TOOL_RATE_LIMIT_ORDER_BOOK', '12'))
TOOL_RATE_LIMIT_RECENT_TRADES = int(os.getenv('TOOL_RATE_LIMIT_RECENT_TRADES', '12'))

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '') # LLM Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')
GEMINI_EXCHANGE_API_KEY = os.getenv('GEMINI_EXCHANGE_API_KEY', '') # Trading Key
GEMINI_EXCHANGE_SECRET = os.getenv('GEMINI_EXCHANGE_SECRET', '') # Trading Secret
GEMINI_SANDBOX_API_KEY = os.getenv('GEMINI_SANDBOX_API_KEY', '') # Sandbox Key
GEMINI_SANDBOX_SECRET = os.getenv('GEMINI_SANDBOX_SECRET', '') # Sandbox Secret

# Exchange Fees
GEMINI_MAKER_FEE = float(os.getenv('GEMINI_MAKER_FEE', '0.0020'))  # 0.20%
GEMINI_TAKER_FEE = float(os.getenv('GEMINI_TAKER_FEE', '0.0040'))  # 0.40%

# LLM Costs (Gemini 2.5 Flash pricing per token)
GEMINI_INPUT_COST_PER_TOKEN = float(os.getenv('GEMINI_INPUT_COST', '0.000000075'))   # $0.075 per 1M tokens
GEMINI_OUTPUT_COST_PER_TOKEN = float(os.getenv('GEMINI_OUTPUT_COST', '0.00000030'))  # $0.30 per 1M tokens
# LLM Costs (OpenAI GPT-4o pricing per token)
OPENAI_INPUT_COST_PER_TOKEN = float(os.getenv('OPENAI_INPUT_COST_PER_TOKEN', '0.000005'))   # $5.00 per 1M tokens
OPENAI_OUTPUT_COST_PER_TOKEN = float(os.getenv('OPENAI_OUTPUT_COST_PER_TOKEN', '0.000015'))  # $15.00 per 1M tokens

# LLM cost/frequency guards
LLM_MAX_SESSION_COST = float(os.getenv('LLM_MAX_SESSION_COST', '10.0'))  # USD cap per session before auto HOLD
LLM_MIN_CALL_INTERVAL_SECONDS = int(os.getenv('LLM_MIN_CALL_INTERVAL_SECONDS', '5'))  # min spacing between planner/decision calls
LLM_MAX_CONSECUTIVE_ERRORS = int(os.getenv('LLM_MAX_CONSECUTIVE_ERRORS', '3'))  # errors before forcing HOLD
AUTO_REPLACE_PLAN_ON_CAP = os.getenv('AUTO_REPLACE_PLAN_ON_CAP', 'false').lower() == 'true'  # replace oldest plan when cap hit
PLAN_MAX_PER_SYMBOL = int(os.getenv('PLAN_MAX_PER_SYMBOL', '2'))
PLAN_MAX_AGE_MINUTES = int(os.getenv('PLAN_MAX_AGE_MINUTES', '60'))
PLAN_TRAIL_TO_BREAKEVEN_PCT = float(os.getenv('PLAN_TRAIL_TO_BREAKEVEN_PCT', '0.01'))  # e.g., 0.01 = 1%
LLM_DECISION_BYTE_BUDGET = int(os.getenv('LLM_DECISION_BYTE_BUDGET', '16000'))  # clamp decision prompts to this many bytes

# Bot versioning
BOT_VERSION = os.getenv('BOT_VERSION', 'v1')

# Order routing
BOT_VERSION = BOT_VERSION or 'v1'
CLIENT_ORDER_PREFIX = os.getenv('CLIENT_ORDER_PREFIX') or f'BOT-{BOT_VERSION}'

# Trade sync guardrails
TRADE_SYNC_CUTOFF_MINUTES = int(os.getenv('TRADE_SYNC_CUTOFF_MINUTES', '1440'))  # ignore trades older than this window when syncing

# Maker/taker preferences
MAKER_PREFERENCE_DEFAULT = os.getenv('MAKER_PREFERENCE_DEFAULT', 'true').lower() == 'true'
def _parse_maker_overrides(raw: str):
    result = {}
    for entry in raw.split(','):
        if not entry.strip() or ':' not in entry:
            continue
        sym, val = entry.split(':', 1)
        key = sym.strip().upper()
        v = val.strip().lower()
        if v in ('true', '1', 'yes'):
            result[key] = True
        elif v in ('false', '0', 'no'):
            result[key] = False
    return result
MAKER_PREFERENCE_OVERRIDES = _parse_maker_overrides(os.getenv('MAKER_PREFERENCE_OVERRIDES', ''))

# Correlation buckets (semicolon-separated groups of comma symbols)
def _parse_correlation_buckets(raw: str):
    buckets = {}
    bucket_idx = 1
    for bucket in raw.split(';'):
        if not bucket.strip():
            continue
        parts = [sym.strip().upper() for sym in bucket.split(',') if sym.strip()]
        if not parts:
            continue
        key = f"bucket_{bucket_idx}"
        buckets[key] = parts
        bucket_idx += 1
    return buckets

CORRELATION_BUCKETS = _parse_correlation_buckets(os.getenv('CORRELATION_BUCKETS', 'BTC/USD,ETH/USD;SOL/USD,ADA/USD'))
BUCKET_MAX_POSITIONS = int(os.getenv('BUCKET_MAX_POSITIONS', '2'))
