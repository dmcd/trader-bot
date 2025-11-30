import pytest

from trader_bot.cost_tracker import CostTracker


def test_ib_stock_fee_respects_minimum(monkeypatch):
    monkeypatch.setattr("trader_bot.cost_tracker.IB_STOCK_COMMISSION_PER_SHARE", 0.005)
    monkeypatch.setattr("trader_bot.cost_tracker.IB_STOCK_MIN_COMMISSION", 1.0)

    tracker = CostTracker("IB")
    fee = tracker.calculate_trade_fee("BHP/AUD", quantity=10, price=100.0, liquidity="taker")

    assert fee == pytest.approx(1.0)


def test_ib_stock_fee_scales_with_size(monkeypatch):
    monkeypatch.setattr("trader_bot.cost_tracker.IB_STOCK_COMMISSION_PER_SHARE", 0.01)
    monkeypatch.setattr("trader_bot.cost_tracker.IB_STOCK_MIN_COMMISSION", 0.5)

    tracker = CostTracker("IB")
    fee = tracker.calculate_trade_fee("CSL/AUD", quantity=200, price=25.0, liquidity="maker")

    assert fee == pytest.approx(2.0)  # 200 * 0.01


def test_ib_fx_commission_pct(monkeypatch):
    monkeypatch.setattr("trader_bot.cost_tracker.IB_FX_COMMISSION_PCT", 0.00015)
    tracker = CostTracker("IB")

    fee = tracker.calculate_trade_fee("AUD/USD", quantity=100000, price=0.65, liquidity="taker")

    assert fee == pytest.approx(100000 * 0.65 * 0.00015)
