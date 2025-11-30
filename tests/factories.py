from trader_bot.strategy import StrategySignal


def make_strategy_signal(action="HOLD", symbol="BTC/USD", quantity=0.0, reason=""):
    return StrategySignal(action, symbol, quantity, reason)


def make_market_data(symbol="BTC/USD", price=100.0, bid=None, ask=None, **extra):
    payload = {
        "symbol": symbol,
        "price": price,
        "bid": bid if bid is not None else price - 1,
        "ask": ask if ask is not None else price + 1,
    }
    payload.update(extra)
    return payload


def make_trade_plan(symbol="BTC/USD", side="long", entry_price=100.0, size=1.0, **overrides):
    plan = {
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "stop_price": entry_price * 0.95,
        "target_price": entry_price * 1.05,
        "size": size,
        "reason": "test-plan",
    }
    plan.update(overrides)
    return plan
