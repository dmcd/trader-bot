import atexit
import os
import tempfile
import unittest
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock

from trader_bot.config import MIN_TRADE_INTERVAL_SECONDS
from trader_bot.risk_manager import RiskManager
from trader_bot.strategy import StrategySignal
from trader_bot.strategy_runner import StrategyRunner

pytestmark = pytest.mark.integration


# Ensure tests never write to the production trading.db
_fd, _db_path = tempfile.mkstemp(prefix="trader-bot-test-", suffix=".db")
os.close(_fd)
os.environ.setdefault("TRADING_DB_PATH", _db_path)


@atexit.register
def _cleanup_db_path():
    if os.path.exists(_db_path):
        try:
            os.remove(_db_path)
        except OSError:
            pass


class StubBot:
    def __init__(self, prices):
        self._prices = list(prices)
        self.place_calls = []
        self.exchange = SimpleNamespace(symbols=["BTC/USD"])

    async def get_market_data_async(self, symbol):
        price = self._prices.pop(0) if self._prices else 100.0
        return {
            "symbol": symbol,
            "price": price,
            "bid": price - 1,
            "ask": price + 1,
            "bid_size": 1.0,
            "ask_size": 1.0,
            "spread_pct": 0.5,
        }

    async def place_order_async(self, symbol, action, qty, prefer_maker=True):
        self.place_calls.append((symbol, action, qty, prefer_maker))
        return {"order_id": f"{len(self.place_calls)}", "liquidity": "maker"}


class StubStrategy:
    def __init__(self, signals):
        self.signals = list(signals)
        self.rejections = []
        self.last_trade_ts = None

    async def generate_signal(self, *_args, **_kwargs):
        return self.signals.pop(0) if self.signals else None

    def on_trade_executed(self, timestamp):
        self.last_trade_ts = timestamp

    def on_trade_rejected(self, reason):
        self.rejections.append(reason)


class TestRunnerIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_runner_blocks_on_risk_spacing_and_slippage(self):
        # Prices consumed in order: risk block, execute, hold (spacing), slippage
        bot = StubBot(prices=[100.0, 100.0, 100.0, 100.0, 100.0, 120.0])
        runner = StrategyRunner(execute_orders=True)
        runner.bot = bot
        runner.cost_tracker = MagicMock()
        runner.cost_tracker.calculate_trade_fee.return_value = 0.0
        runner.db = MagicMock()
        runner.db.log_trade = MagicMock()
        runner.db.log_llm_call = MagicMock()

        # Use real risk manager with stubbed decision results
        runner.risk_manager = RiskManager(bot)
        risk_results = [
            SimpleNamespace(allowed=False, reason="exposure_cap"),
            SimpleNamespace(allowed=True, reason=None),
            SimpleNamespace(allowed=True, reason=None),
        ]
        runner.risk_manager.check_trade_allowed = MagicMock(side_effect=risk_results)

        signals = [
            StrategySignal("BUY", "BTC/USD", 0.5, "risk_block"),
            StrategySignal("BUY", "BTC/USD", 0.5, "ok"),
            StrategySignal("HOLD", "BTC/USD", 0, "cooldown"),
            StrategySignal("BUY", "BTC/USD", 0.5, "slip"),
        ]
        strategy = StubStrategy(signals)
        runner.strategy = strategy

        async def drive_signal(signal):
            md = await bot.get_market_data_async(signal.symbol)
            if signal.action == "HOLD":
                return
            risk = runner.risk_manager.check_trade_allowed(signal.symbol, signal.action, signal.quantity, md["price"])
            if not risk.allowed:
                strategy.on_trade_rejected(risk.reason)
                return
            latest = await bot.get_market_data_async(signal.symbol)
            ok_slip, _ = runner._slippage_within_limit(md["price"], latest["price"], latest)
            if not ok_slip:
                strategy.on_trade_rejected("Slippage over limit")
                return
            await bot.place_order_async(signal.symbol, signal.action, signal.quantity, prefer_maker=True)
            strategy.on_trade_executed(0.0)

        for sig in list(strategy.signals):
            out = await strategy.generate_signal(None, {sig.symbol: {}}, 1000.0, 0.0, None, {})
            if out is None:
                break
            await drive_signal(out)

        self.assertEqual(len(bot.place_calls), 1)  # only the second signal should place
        self.assertIn("exposure_cap", strategy.rejections)
        self.assertIn("Slippage over limit", strategy.rejections)
        # Spacing/HOLD respected: no order for HOLD leg
        self.assertEqual(strategy.last_trade_ts, 0.0)


if __name__ == "__main__":
    unittest.main()
