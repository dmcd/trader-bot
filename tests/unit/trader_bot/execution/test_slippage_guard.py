from unittest.mock import MagicMock

from trader_bot.strategy_runner import MAX_SLIPPAGE_PCT, MAX_SPREAD_PCT, MIN_TOP_OF_BOOK_NOTIONAL, StrategyRunner


def test_slippage_guard_delegates_to_action_handler():
    runner = StrategyRunner()
    runner.action_handler = MagicMock()
    runner.action_handler.slippage_within_limit.return_value = (True, 0.0)

    ok, move = runner._slippage_within_limit(100.0, 101.0, {"symbol": "BTC/USD"})

    assert ok is True
    assert move == 0.0
    runner.action_handler.slippage_within_limit.assert_called_with(
        100.0,
        101.0,
        {"symbol": "BTC/USD"},
        max_slippage_pct=MAX_SLIPPAGE_PCT,
        max_spread_pct=MAX_SPREAD_PCT,
        min_top_of_book_notional=MIN_TOP_OF_BOOK_NOTIONAL,
    )
