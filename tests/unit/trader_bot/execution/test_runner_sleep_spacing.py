import asyncio
import time
from types import SimpleNamespace

import pytest

from trader_bot.services.command_processor import CommandResult
from trader_bot.services.strategy_orchestrator import RiskCheckResult
from trader_bot.strategy import StrategySignal
from trader_bot.strategy_runner import StrategyRunner
from tests.factories import make_strategy_signal


class DummyBot:
    def __init__(self, price: float = 100.0):
        self.price = price
        self.place_calls = []

    async def get_equity_async(self):
        return 1_000.0

    async def get_market_data_async(self, symbol):
        return {"symbol": symbol, "price": self.price, "bid": self.price - 1, "ask": self.price + 1, "spread_pct": 0.1}

    async def get_positions_async(self):
        return []

    async def get_open_orders_async(self):
        return []

    async def get_my_trades_async(self, *_, **__):
        return []

    async def place_order_async(self, symbol, action, qty, prefer_maker=True):
        self.place_calls.append((symbol, action, qty, prefer_maker))
        return {"order_id": str(len(self.place_calls)), "liquidity": "maker"}


class DummyDB:
    def log_equity_snapshot(self, *_, **__):
        return None

    def log_market_data(self, *_, **__):
        return None

    def prune_market_data(self, *_, **__):
        return None

    def replace_positions(self, *_, **__):
        return None

    def replace_open_orders(self, *_, **__):
        return None

    def get_positions(self, *_, **__):
        return []

    def get_open_orders(self, *_, **__):
        return []

    def get_open_trade_plans(self, *_, **__):
        return []

    def get_recent_market_data(self, *_, **__):
        return []

    def count_open_trade_plans_for_symbol(self, *_, **__):
        return 0

    def create_trade_plan(self, *_, **__):
        return 1

    def log_estimated_fee(self, *_, **__):
        return None

    def set_health_state(self, *_, **__):
        return None

    def get_trade_plan_reason_by_order(self, *_, **__):
        return None

    def get_trade_count(self, *_, **__):
        return 0

    def update_llm_trace_execution(self, *_, **__):
        return None

    def get_processed_trade_ids(self, *_, **__):
        return []

    def get_latest_trade_timestamp(self, *_, **__):
        return None


class DummyHealth:
    def should_pause(self, *_args):
        return False

    def pause_remaining(self, *_args):
        return 0

    def record_exchange_failure(self, *_, **__):
        return None

    def record_tool_failure(self, *_, **__):
        return None

    def record_tool_success(self, *_, **__):
        return None

    def reset_exchange_errors(self):
        return None

    async def maybe_reconnect(self, *_, **__):
        return None


class DummyOrchestrator:
    def __init__(self):
        self.running = True

    async def start(self, initialize_cb):
        await initialize_cb()
        self.running = True

    def request_stop(self, *_args, **__):
        self.running = False

    async def process_commands(self, *_, **__):
        return CommandResult()

    async def enforce_risk_budget(self, *_, **__):
        return RiskCheckResult(should_stop=False, kill_switch=False)

    def emit_market_health(self, *_args, **__):
        return True, {}

    def emit_operational_metrics(self, *_args, **__):
        return None

    async def monitor_trade_plans(self, *_, **__):
        return None

    async def cleanup(self, cleanup_cb):
        if asyncio.iscoroutinefunction(cleanup_cb):
            return await cleanup_cb()
        if callable(cleanup_cb):
            return cleanup_cb()
        return None


class DummyStrategy:
    def __init__(self, signals):
        self.signals = list(signals)
        self.rejections = []
        self.executions = 0

    async def generate_signal(self, *_args, **__):
        return self.signals.pop(0) if self.signals else None

    def on_trade_rejected(self, reason):
        self.rejections.append(reason)

    def on_trade_executed(self, _ts):
        self.executions += 1


def _build_runner(signal: StrategySignal, *, risk_allowed=True, slippage_ok=True, execute_orders=True):
    runner = StrategyRunner(execute_orders=execute_orders)
    async def _init():
        return None
    runner.initialize = _init
    async def _cleanup():
        return None
    runner.cleanup = _cleanup
    runner.orchestrator = DummyOrchestrator()
    runner.bot = DummyBot()
    runner.db = DummyDB()
    runner.health_manager = DummyHealth()
    runner.risk_manager = SimpleNamespace(
        update_equity=lambda _eq: None,
        start_of_day_equity=1_000.0,
        daily_loss=0.0,
        positions={},
        pending_orders_by_symbol={},
        get_total_exposure=lambda *_args, **_kwargs: 0.0,
        check_trade_allowed=lambda *_: SimpleNamespace(allowed=risk_allowed, reason="blocked" if not risk_allowed else None),
        update_positions=lambda _positions=None, price_overrides=None: None,
        update_pending_orders=lambda *_args, **_kwargs: None,
    )
    runner._get_active_symbols = lambda: ["BTC/USD"]
    async def _noop(*_args, **_kwargs):
        return None
    runner._capture_ohlcv = _noop
    runner._monitor_trade_plans = _noop
    runner._liquidity_ok = lambda *_: True
    runner._stacking_block = lambda *_: False
    runner._slippage_within_limit = lambda *_args, **__: (slippage_ok, 0.0 if slippage_ok else 1.0)
    runner.session_id = 1
    runner.exchange_name = "TEST"
    runner.daily_loss_pct = 1_000.0
    runner._monotonic = time.monotonic
    runner.strategy = DummyStrategy([signal])
    runner._estimated_fees = {}
    runner._open_trade_plans = {}
    runner.order_reasons = {}
    runner.session_stats = {"total_trades": 0}
    runner.context = {}
    return runner


async def _run_once_with_sleep_capture(runner: StrategyRunner):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    orig_sleep = asyncio.sleep
    asyncio.sleep = fake_sleep
    try:
        await runner.run_loop(max_loops=1)
    finally:
        asyncio.sleep = orig_sleep
    return sleep_calls


@pytest.mark.asyncio
async def test_hold_decision_sleeps():
    runner = _build_runner(make_strategy_signal("HOLD", "BTC/USD", 0, "hold-branch"))
    sleep_calls = await _run_once_with_sleep_capture(runner)
    assert len(sleep_calls) == 1


@pytest.mark.asyncio
async def test_risk_block_sleeps():
    runner = _build_runner(make_strategy_signal("BUY", "BTC/USD", 0.1, "risk-block"), risk_allowed=False)
    sleep_calls = await _run_once_with_sleep_capture(runner)
    assert len(sleep_calls) == 1


@pytest.mark.asyncio
async def test_slippage_block_sleeps():
    runner = _build_runner(make_strategy_signal("BUY", "BTC/USD", 0.1, "slip-block"), risk_allowed=True, slippage_ok=False)
    sleep_calls = await _run_once_with_sleep_capture(runner)
    assert len(sleep_calls) == 1


@pytest.mark.asyncio
async def test_shadow_mode_sleeps():
    runner = _build_runner(
        make_strategy_signal("BUY", "BTC/USD", 0.1, "shadow"), risk_allowed=True, slippage_ok=True, execute_orders=False
    )
    sleep_calls = await _run_once_with_sleep_capture(runner)
    assert len(sleep_calls) == 1
