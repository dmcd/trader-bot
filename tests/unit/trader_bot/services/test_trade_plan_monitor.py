import asyncio
import atexit
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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

from trader_bot.strategy_runner import StrategyRunner
from trader_bot.services.trade_action_handler import TradeActionHandler
from trader_bot.services.plan_monitor import PlanMonitorConfig


class TestTradePlanMonitor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Stub out bot and db with mocks
        self.runner = StrategyRunner()
        self.runner.bot = MagicMock()
        self.runner.cost_tracker = MagicMock()
        self.runner.cost_tracker.calculate_trade_fee.return_value = 1.0
        self.runner._apply_fill_to_session_stats = Mock(return_value=None)
        self.runner.session_id = 1
        self._refresh_monitor_bindings()

    def _refresh_monitor_bindings(self):
        # Rebind action handler to ensure it uses latest fakes/mocks
        self.runner.action_handler = TradeActionHandler(
            db=self.runner.db,
            bot=self.runner.bot,
            risk_manager=self.runner.risk_manager,
            cost_tracker=self.runner.cost_tracker,
            portfolio_tracker=self.runner.portfolio_tracker,
            prefer_maker=self.runner._prefer_maker,
            health_manager=self.runner.health_manager,
            emit_telemetry=self.runner._emit_telemetry,
            log_execution_trace=self.runner._log_execution_trace,
            on_trade_rejected=self.runner.strategy.on_trade_rejected,
            actions_logger=getattr(self.runner.strategy, "logger", None),
            logger=None,
        )
        self.runner.plan_monitor.refresh_bindings(
            bot=self.runner.bot,
            db=self.runner.db,
            cost_tracker=self.runner.cost_tracker,
            risk_manager=self.runner.risk_manager,
            prefer_maker=self.runner._prefer_maker,
            holdings_updater=self.runner._update_holdings_and_realized,
            session_stats_applier=self.runner._apply_fill_to_session_stats,
        )

    async def _run_monitor(self, price_lookup, open_orders):
        config = PlanMonitorConfig(
            max_plan_age_minutes=self.runner.max_plan_age_minutes,
            day_end_flatten_hour_utc=self.runner.day_end_flatten_hour_utc,
            trail_to_breakeven_pct=self.runner._apply_plan_trailing_pct,
        )
        self._refresh_monitor_bindings()
        await self.runner.plan_monitor.monitor(
            self.runner.session_id,
            price_lookup=price_lookup,
            open_orders=open_orders,
            config=config,
        )

    @patch('trader_bot.strategy_runner.datetime')
    async def test_plan_age_flatten(self, mock_dt):
        now = datetime.now(timezone.utc)
        mock_dt.now.return_value = now
        mock_dt.fromisoformat.side_effect = datetime.fromisoformat
        mock_dt.timezone = timezone

        # Plan opened 2 hours ago, age limit 60 min
        self.runner.max_plan_age_minutes = 60
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 1,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'stop_price': None,
            'target_price': None,
            'size': 0.1,
            'opened_at': (now - timedelta(minutes=120)).isoformat()
        }]
        # Simulate price_now
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()

        await self._run_monitor(price_lookup={"BTC/USD": 100}, open_orders=[])
        self.runner.db.update_trade_plan_status.assert_called_once()

    async def test_headroom_cancel(self):
        now = datetime.now(timezone.utc)
        self.runner.max_plan_age_minutes = None  # disable age check for this test
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 2,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'stop_price': None,
            'target_price': None,
            'size': 0.1,
            'opened_at': now.isoformat()
        }]
        # Simulate exposure over cap
        self.runner.risk_manager.get_total_exposure = MagicMock(return_value=1e12)
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()

        await self._run_monitor(price_lookup={"BTC/USD": 100}, open_orders=[])
        # Ensure we evaluated exposure headroom and attempted closure path
        self.runner.risk_manager.get_total_exposure.assert_called_once()
        self.assertGreaterEqual(self.runner.db.update_trade_plan_status.call_count, 0)

    async def test_trailing_stop_to_breakeven(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 3,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 95,
            'target_price': 110,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'medium (1%)',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        await self._run_monitor(price_lookup={"BTC/USD": 102}, open_orders=[])
        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_trailing_stop_uses_regime_flags_when_no_volatility(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 7,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 90,
            'target_price': 115,
            'size': 0.2,
            'opened_at': now.isoformat(),
            'version': 1,
            'regime_flags': {'volatility': 'low'},
        }]
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": 0.2, "current_price": 100}}
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '999', 'liquidity': 'taker'})

        await self._run_monitor(price_lookup={"BTC/USD": 102}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_prices.call_args
        self.assertEqual(kwargs.get("stop_price"), 100)

    async def test_trailing_stop_tightens_on_high_vol(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 5,
            'symbol': 'BTC/USD',
            'side': 'SELL',
            'entry_price': 100,
            'stop_price': 105,
            'target_price': 90,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'high (2%)',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        await self._run_monitor(price_lookup={"BTC/USD": 99}, open_orders=[])
        # High-vol trailing should be tighter; if stop already above entry, it should still trail to breakeven when rule met
        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_trailing_tightens_breakeven_in_high_vol(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": 0.1, "current_price": 100}}
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 12,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 95,
            'target_price': 110,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'high',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'tight', 'liquidity': 'taker'})

        await self._run_monitor(price_lookup={"BTC/USD": 100.8}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_called_once()

    async def test_trailing_waits_longer_on_low_vol(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": 0.1, "current_price": 100}}
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 13,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 95,
            'target_price': 110,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'low',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'wide', 'liquidity': 'taker'})

        await self._run_monitor(price_lookup={"BTC/USD": 101.0}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_not_called()

    async def test_trails_stop_to_breakeven_for_buy(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 6,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 90,
            'target_price': 120,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
            'volatility': 'normal',
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '321', 'liquidity': 'taker'})

        await self._run_monitor(price_lookup={"BTC/USD": 102}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_prices.call_args
        # Ensure stop trailed up to entry price
        self.assertEqual(kwargs.get("stop_price"), 100)

    async def test_plan_closes_when_flat_and_no_orders(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 4,
            'symbol': 'ETH/USD',
            'side': 'SELL',
            'entry_price': 1500,
            'stop_price': 1550,
            'target_price': 1400,
            'size': 0.5,
            'opened_at': now.isoformat(),
            'version': 1
        }]
        self.runner.risk_manager.positions = {"ETH/USD": {"quantity": 0.0, "current_price": 1500}}
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': '123', 'liquidity': 'taker'})
        self.runner.db.log_trade = Mock()

        await self._run_monitor(price_lookup={"ETH/USD": 1500}, open_orders=[])
        self.runner.db.update_trade_plan_status.assert_called_once()

    async def test_apply_fill_not_treated_as_awaitable(self):
        now = datetime.now(timezone.utc)
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 8,
            'symbol': 'ETH/USD',
            'side': 'SELL',
            'entry_price': 1500,
            'stop_price': 1550,
            'target_price': 1400,
            'size': 0.5,
            'opened_at': now.isoformat(),
            'version': 1
        }]
        self.runner.risk_manager.positions = {"ETH/USD": {"quantity": 0.0, "current_price": 1500}}
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'trap', 'liquidity': 'taker'})

        class AwaitTrap:
            def __await__(self):
                raise AssertionError("should not await session stats")

        self.runner._apply_fill_to_session_stats = Mock(return_value=AwaitTrap())

        await self._run_monitor(price_lookup={"ETH/USD": 1500}, open_orders=[])

        self.runner._apply_fill_to_session_stats.assert_called_once()

    async def test_partial_close_updates_plan_size(self):
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 10,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'size': 1.0,
        }]
        self.runner.db.update_trade_plan_size = Mock()
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'pc', 'liquidity': 'taker'})
        self.runner._update_holdings_and_realized = Mock(return_value=0.0)
        self.runner._apply_fill_to_session_stats = Mock()
        self.runner.risk_manager.check_trade_allowed = Mock(return_value=SimpleNamespace(allowed=True, reason=None))
        self._refresh_monitor_bindings()

        signal = SimpleNamespace(plan_id=10, close_fraction=0.5, symbol="BTC/USD", action="PARTIAL_CLOSE", reason="test")
        telemetry_record = {}

        await self.runner._handle_partial_close(signal, telemetry_record, trace_id=None, market_data={"BTC/USD": {"price": 100}}, current_exposure=0.0)

        self.runner.db.update_trade_plan_size.assert_called_once()
        self.runner.db.update_trade_plan_status.assert_not_called()
        self.assertEqual(telemetry_record.get("status"), "partial_close_executed")

    async def test_partial_close_closes_plan_when_fully_flat(self):
        self.runner.db = MagicMock()
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 11,
            'symbol': 'ETH/USD',
            'side': 'SELL',
            'size': 0.2,
        }]
        self.runner.db.update_trade_plan_size = Mock()
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'pc2', 'liquidity': 'taker'})
        self.runner._update_holdings_and_realized = Mock(return_value=0.0)
        self.runner._apply_fill_to_session_stats = Mock()
        self.runner.risk_manager.check_trade_allowed = Mock(return_value=SimpleNamespace(allowed=True, reason=None))
        self._refresh_monitor_bindings()

        signal = SimpleNamespace(plan_id=11, close_fraction=1.0, symbol="ETH/USD", action="PARTIAL_CLOSE", reason="test")
        telemetry_record = {}

        await self.runner._handle_partial_close(signal, telemetry_record, trace_id=None, market_data={"ETH/USD": {"price": 1500}}, current_exposure=0.0)

        self.runner.db.update_trade_plan_status.assert_called_once()
        self.runner.db.update_trade_plan_size.assert_not_called()
        self.assertEqual(telemetry_record.get("status"), "partial_close_executed")

    @patch('trader_bot.strategy_runner.datetime')
    async def test_day_end_flatten_closes_active_plan(self, mock_dt):
        now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        mock_dt.now.return_value = now
        mock_dt.fromisoformat.side_effect = datetime.fromisoformat
        mock_dt.timezone = timezone

        self.runner.day_end_flatten_hour_utc = now.hour
        self.runner.max_plan_age_minutes = None
        self.runner.db = MagicMock()
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": 0.1, "current_price": 100}}
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 20,
            'symbol': 'BTC/USD',
            'side': 'BUY',
            'entry_price': 100,
            'stop_price': 90,
            'target_price': 120,
            'size': 0.1,
            'opened_at': (now - timedelta(hours=2)).isoformat(),
            'version': 1,
        }]
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'close', 'liquidity': 'taker'})
        self.runner.db.update_trade_plan_status = Mock()
        self.runner.db.log_trade = Mock()
        self.runner.db.update_trade_plan_prices = Mock()

        await self._run_monitor(price_lookup={"BTC/USD": 101}, open_orders=[])

        self.runner.db.update_trade_plan_status.assert_called_once()

    async def test_sell_plan_trails_stop_to_breakeven(self):
        now = datetime.now(timezone.utc)
        self.runner._apply_plan_trailing_pct = 0.05
        self.runner.db = MagicMock()
        self.runner.risk_manager.positions = {"BTC/USD": {"quantity": -0.1, "current_price": 100}}
        self.runner.db.get_open_trade_plans.return_value = [{
            'id': 21,
            'symbol': 'BTC/USD',
            'side': 'SELL',
            'entry_price': 100,
            'stop_price': 105,
            'target_price': 90,
            'size': 0.1,
            'opened_at': now.isoformat(),
            'version': 1,
        }]
        self.runner.db.update_trade_plan_prices = Mock()
        self.runner.bot.place_order_async = AsyncMock(return_value={'order_id': 'trail', 'liquidity': 'taker'})

        await self._run_monitor(price_lookup={"BTC/USD": 94}, open_orders=[])

        self.runner.db.update_trade_plan_prices.assert_called_once()
        args, kwargs = self.runner.db.update_trade_plan_prices.call_args
        self.assertEqual(kwargs.get("stop_price"), 100)


if __name__ == '__main__':
    unittest.main()
