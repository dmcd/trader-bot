from trader_bot.accounting import AccountSnapshot, estimate_commissions_for_orders
from trader_bot.cost_tracker import CostTracker


def test_account_snapshot_parses_ib_entries():
    entries = [
        {"tag": "NetLiquidation", "value": "10000", "currency": "AUD"},
        {"tag": "AvailableFunds", "value": 5000, "currency": "AUD"},
        {"tag": "ExcessLiquidity", "value": "4000", "currency": "AUD"},
        {"tag": "BuyingPower", "value": 15000, "currency": "AUD"},
        {"tag": "TotalCashValue", "value": 1200, "currency": "USD"},
        {"tag": "TotalCashValue", "value": 800, "currency": "AUD"},
    ]

    snap = AccountSnapshot.from_entries(entries, base_currency="AUD", source="IB")

    assert snap is not None
    assert snap.base_currency == "AUD"
    assert snap.net_liquidation == 10000
    assert snap.available_funds == 5000
    assert snap.excess_liquidity == 4000
    assert snap.buying_power == 15000
    assert snap.cash_balances == {"USD": 1200, "AUD": 800}


def test_estimate_commissions_uses_price_lookup_and_skips_incomplete():
    cost_tracker = CostTracker("IB")
    open_orders = [
        {"symbol": "BHP/AUD", "side": "BUY", "remaining": 10, "price": None},
        {"symbol": "AUD/USD", "side": "SELL", "amount": 1_000, "price": 0.65},
        {"symbol": None, "side": "SELL", "amount": 1, "price": 1},  # malformed
    ]
    price_lookup = {"BHP/AUD": 5.0}

    estimates = estimate_commissions_for_orders(open_orders, price_lookup, cost_tracker)

    assert len(estimates) == 2
    bhp = next(e for e in estimates if e["symbol"] == "BHP/AUD")
    fx = next(e for e in estimates if e["symbol"] == "AUD/USD")
    assert bhp["price"] == 5.0
    assert bhp["estimated_fee"] >= 1.0  # min commission kicks in
    assert fx["price"] == 0.65
    assert fx["estimated_fee"] >= 0
