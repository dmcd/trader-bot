import pytest

import trader_bot.risk_manager as rm_module
from trader_bot.risk_manager import RiskManager


@pytest.fixture(autouse=True)
def risk_config():
    originals = {
        "MAX_ORDER_VALUE": rm_module.MAX_ORDER_VALUE,
        "MAX_TOTAL_EXPOSURE": rm_module.MAX_TOTAL_EXPOSURE,
        "MIN_TRADE_SIZE": rm_module.MIN_TRADE_SIZE,
        "ORDER_VALUE_BUFFER": rm_module.ORDER_VALUE_BUFFER,
        "MAX_POSITIONS": rm_module.MAX_POSITIONS,
        "CORRELATION_BUCKETS": rm_module.CORRELATION_BUCKETS,
        "BUCKET_MAX_POSITIONS": rm_module.BUCKET_MAX_POSITIONS,
    }
    rm_module.MAX_ORDER_VALUE = 500.0
    rm_module.MAX_TOTAL_EXPOSURE = 1000.0
    rm_module.MIN_TRADE_SIZE = 1.0
    rm_module.ORDER_VALUE_BUFFER = 1.0
    rm_module.MAX_POSITIONS = 3
    rm_module.CORRELATION_BUCKETS = {"majors": ["BTC/USD", "ETH/USD"]}
    rm_module.BUCKET_MAX_POSITIONS = 1
    yield
    for key, value in originals.items():
        setattr(rm_module, key, value)


@pytest.fixture
def risk_manager():
    rm = RiskManager()
    rm.update_equity(1000.0)
    return rm


def test_invalid_price_or_quantity_rejected(risk_manager):
    assert risk_manager.check_trade_allowed("BHP", "BUY", 0, 100.0).allowed is False
    assert risk_manager.check_trade_allowed("BHP", "BUY", 1, 0.0).allowed is False


def test_set_baseline_stores_metadata(risk_manager):
    assert risk_manager.baseline_equity is None
    assert risk_manager.baseline_timestamp is None

    risk_manager.set_baseline(123.4, timestamp="2024-01-01T00:00:00Z")

    assert risk_manager.baseline_equity == pytest.approx(123.4)
    assert risk_manager.baseline_timestamp == "2024-01-01T00:00:00Z"


def test_order_value_cap(risk_manager):
    over_size_qty = (rm_module.MAX_ORDER_VALUE / 10.0) + 1
    result = risk_manager.check_trade_allowed("BHP", "BUY", over_size_qty, price=10.0)
    assert result.allowed is False
    assert "Order value" in result.reason


def test_update_equity_tracks_current_value(risk_manager):
    risk_manager.update_equity(1234.5)
    assert risk_manager.current_equity == pytest.approx(1234.5)
    allowed = risk_manager.check_trade_allowed("BHP", "BUY", 1, 10.0)
    assert allowed.allowed


def test_exposure_limit_with_existing_positions(risk_manager):
    risk_manager.update_positions({"ETH/USD": {"quantity": 80.0, "current_price": 10.0}})

    ok_result = risk_manager.check_trade_allowed("ETH/USD", "BUY", 10, price=10.0)
    assert ok_result.allowed

    qty_to_exceed = 25
    blocked = risk_manager.check_trade_allowed("ETH/USD", "BUY", qty_to_exceed, price=10.0)
    assert blocked.allowed is False
    assert "Total exposure" in blocked.reason


def test_safe_buffer_allows_near_cap(risk_manager):
    near_cap_price = rm_module.MAX_TOTAL_EXPOSURE / 9.5
    risk_manager.update_positions({"ABC": {"quantity": 9.0, "current_price": near_cap_price}})
    assert risk_manager.check_trade_allowed("XYZ", "BUY", 1, price=1.0).allowed


def test_apply_order_value_buffer_trims_small_overage(risk_manager):
    price = 100.0
    qty = (rm_module.MAX_ORDER_VALUE / price) + 0.02

    adjusted_qty, overage = risk_manager.apply_order_value_buffer(qty, price)

    assert overage > 0
    assert adjusted_qty < qty
    expected_cap = rm_module.MAX_ORDER_VALUE - rm_module.ORDER_VALUE_BUFFER
    assert adjusted_qty * price <= expected_cap

    near_cap_price = 50.0
    capped_value = rm_module.MAX_ORDER_VALUE - rm_module.ORDER_VALUE_BUFFER
    near_cap_qty = (capped_value / near_cap_price) + 0.01

    buffered_qty, buffered_overage = risk_manager.apply_order_value_buffer(near_cap_qty, near_cap_price)
    assert buffered_overage > 0
    assert buffered_qty < near_cap_qty
    assert pytest.approx(buffered_qty * near_cap_price) == capped_value

    kept_qty, kept_overage = risk_manager.apply_order_value_buffer(1.0, 10.0)
    assert kept_qty == 1.0
    assert kept_overage == 0.0


def test_apply_order_value_buffer_handles_zero_price_or_quantity(risk_manager):
    qty, overage = risk_manager.apply_order_value_buffer(0.0, 100.0)
    assert qty == 0.0
    assert overage == 0.0

    qty_zero_price, overage_zero_price = risk_manager.apply_order_value_buffer(1.0, 0.0)
    assert qty_zero_price == 1.0
    assert overage_zero_price == 0.0


def test_get_total_exposure_respects_overrides(risk_manager):
    risk_manager.update_positions({"BTC/USD": {"quantity": 5.0, "current_price": 30000.0}})
    exposure = risk_manager.get_total_exposure(price_overrides={"BTC/USD": 20000.0})
    assert exposure == pytest.approx(100000.0)


def test_pending_buy_orders_reduce_headroom(risk_manager):
    risk_manager.update_positions({"ETH/USD": {"quantity": 20.0, "current_price": 20.0}})
    pending = [
        {"symbol": "ETH/USD", "side": "buy", "price": 30.0, "amount": 10.0, "remaining": 10.0},
    ]
    risk_manager.update_pending_orders(pending)

    result = risk_manager.check_trade_allowed("ETH/USD", "BUY", 20, price=20.0)
    assert result.allowed is False
    assert "Total exposure" in result.reason

    pending_sell = [{"symbol": "ETH/USD", "side": "sell", "price": 30.0, "amount": 10.0, "remaining": 10.0}]
    risk_manager.update_pending_orders(pending_sell)
    result_ok = risk_manager.check_trade_allowed("ETH/USD", "BUY", 5, price=20.0)
    assert result_ok.allowed


def test_pending_sells_offset_longs(risk_manager):
    risk_manager.update_positions({"ETH/USD": {"quantity": 20.0, "current_price": 20.0}})
    pending_sell = [{"symbol": "ETH/USD", "side": "sell", "price": 20.0, "amount": 10.0, "remaining": 10.0}]
    risk_manager.update_pending_orders(pending_sell)

    exposure = risk_manager.get_total_exposure()
    assert exposure == pytest.approx(200.0)

    pending_short = [{"symbol": "BTC/USD", "side": "sell", "price": 100.0, "amount": 5.0, "remaining": 5.0}]
    risk_manager.update_positions({"BTC/USD": {"quantity": 2.0, "current_price": 100.0}})
    risk_manager.update_pending_orders(pending_short)
    exposure_short = risk_manager.get_total_exposure()
    assert exposure_short == pytest.approx(300.0)


def test_pending_orders_by_symbol_tracks_count_and_notional(risk_manager):
    pending = [
        {"symbol": "BTC/USD", "side": "buy", "price": 30000.0, "amount": 0.1, "remaining": 0.05},
        {"symbol": "BTC/USD", "side": "buy", "price": 31000.0, "amount": 0.2, "remaining": 0.1},
        {"symbol": "ETH/USD", "side": "buy", "price": 2000.0, "amount": 1.0, "remaining": 1.0},
        {"symbol": "ETH/USD", "side": "sell", "price": 2100.0, "amount": 0.5, "remaining": 0.5},
    ]
    risk_manager.update_pending_orders(pending)

    sym = risk_manager.pending_orders_by_symbol
    assert sym["BTC/USD"]["count_buy"] == 2
    assert sym["BTC/USD"]["buy"] > 0
    assert sym["ETH/USD"]["count_buy"] == 1
    assert sym["ETH/USD"]["buy"] == pytest.approx(2000.0)
    assert sym["ETH/USD"]["count_sell"] == 1
    assert sym["ETH/USD"]["sell"] == pytest.approx(1050.0)


def test_exposure_converts_to_base_currency(risk_config):
    fx_rates = {"USD": 1.5}
    fx_provider = lambda currency, **_: fx_rates.get(currency)
    rm = RiskManager(base_currency="AUD", fx_rate_provider=fx_provider)
    rm.update_positions({"AAPL/USD": {"quantity": 2.0, "current_price": 100.0}})

    exposure = rm.get_total_exposure()
    assert exposure == pytest.approx(300.0)


def test_order_value_conversion_blocks_when_fx_pushes_over_cap(risk_config):
    fx_provider = lambda currency, **_: 1.5 if currency == "USD" else None
    rm = RiskManager(base_currency="AUD", fx_rate_provider=fx_provider)

    result = rm.check_trade_allowed("AAPL/USD", "BUY", 5.0, price=200.0)
    assert result.allowed is False
    assert "exceeds limit" in result.reason


def test_min_trade_size_uses_converted_value(risk_config):
    fx_provider = lambda currency, **_: 2.0 if currency == "USD" else None
    rm = RiskManager(base_currency="AUD", fx_rate_provider=fx_provider)

    result = rm.check_trade_allowed("AAPL/USD", "BUY", 0.2, price=2.0)
    assert result.allowed is False
    assert "below minimum" in result.reason


def test_short_exposure_converts_to_base_currency(risk_config):
    fx_provider = lambda currency, **_: 1.5 if currency == "USD" else 1.0
    rm = RiskManager(base_currency="AUD", fx_rate_provider=fx_provider)
    rm.update_positions({"MSFT/USD": {"quantity": -3.0, "current_price": 50.0}})

    exposure = rm.get_total_exposure()
    assert exposure == pytest.approx(225.0)


def test_order_value_buffer_converts_fx_for_shorts(risk_config):
    fx_provider = lambda currency, **_: 1.6 if currency == "USD" else 1.0
    rm = RiskManager(base_currency="AUD", fx_rate_provider=fx_provider)

    qty, overage = rm.apply_order_value_buffer(20.0, 40.0, symbol="AAPL/USD")

    assert overage > 0
    assert qty < 10.0


def test_projected_exposure_for_shorts_adds_incremental_notional(risk_config):
    rm = RiskManager(base_currency="USD")
    rm.update_positions({"BBB/USD": {"quantity": -10.0, "current_price": 80.0}})

    result = rm.check_trade_allowed("BBB/USD", "SELL", 1.0, price=80.0)
    assert result.allowed


def test_max_positions_blocks_new_symbol_when_full(risk_config):
    rm_module.MAX_POSITIONS = 1
    rm = RiskManager()
    rm.update_positions({"AAA": {"quantity": 1.0, "current_price": 10.0}})

    result = rm.check_trade_allowed("BBB", "BUY", 1, price=10.0)
    assert result.allowed is False
    assert "Max positions" in result.reason


def test_pending_order_cap_per_symbol(risk_config):
    rm_module.MAX_POSITIONS = 1
    rm = RiskManager()
    pending = [{"symbol": "AAA", "side": "buy", "price": 10.0, "amount": 1.0, "remaining": 1.0}]
    rm.update_pending_orders(pending)

    result = rm.check_trade_allowed("AAA", "BUY", 1, price=10.0)
    assert result.allowed is False
    assert "Open order cap" in result.reason


def test_correlation_bucket_blocks_new_symbol(risk_manager):
    risk_manager.update_positions({"BTC/USD": {"quantity": 1.0, "current_price": 100.0}})
    result = risk_manager.check_trade_allowed("ETH/USD", "BUY", 1.0, price=100.0)
    assert result.allowed is False
    assert "Correlation bucket limit" in result.reason


def test_correlation_bucket_blocks_pending_buys(risk_manager):
    pending = [{"symbol": "BTC/USD", "side": "buy", "price": 100.0, "amount": 1.0, "remaining": 1.0}]
    risk_manager.update_pending_orders(pending)
    result = risk_manager.check_trade_allowed("ETH/USD", "BUY", 0.5, price=100.0)
    assert result.allowed is False
    assert "Correlation bucket limit" in result.reason


def test_baseline_exposure_and_pending_offsets(risk_config):
    rm = RiskManager(ignore_baseline_positions=True)
    rm.position_baseline = {"BTC/USD": 1.0, "ETH/USD": -1.0}
    rm.update_positions(
        {
            "BTC/USD": {"quantity": 1.0, "current_price": 20000.0},
            "ETH/USD": {"quantity": -2.0, "current_price": 1000.0},
        }
    )
    rm.update_pending_orders(
        [
            {"symbol": "BTC/USD", "side": "buy", "price": 20000.0, "amount": 0.5, "remaining": 0.5},
            {"symbol": "ETH/USD", "side": "sell", "price": 1000.0, "amount": 1.0, "remaining": 1.0},
        ]
    )

    exposure = rm.get_total_exposure()
    assert exposure == pytest.approx(10000.0)
    assert rm._net_quantity_for_exposure(1.0, 1.0) == 0.0
    assert rm._net_quantity_for_exposure(-2.0, -1.0) == -1.0


def test_baseline_and_setters_cover_noops(risk_config):
    rm_with_ignore = RiskManager(ignore_baseline_positions=True)
    rm_with_ignore.update_positions({"BTC/USD": {"quantity": 2.0, "current_price": 10.0}})
    assert "BTC/USD" in rm_with_ignore.position_baseline

    rm_with_ignore.set_position_baseline({})
    assert rm_with_ignore.position_baseline["BTC/USD"] == 2.0

    rm = RiskManager()
    rm.seed_start_of_day(None)
    assert rm.current_equity is None
    rm.update_equity(None)
    assert rm.current_equity is None


def test_pending_order_update_skips_incomplete_rows(risk_manager):
    risk_manager.update_pending_orders(
        [
            {"symbol": "BTC/USD", "side": "buy", "price": 0.0, "amount": 1.0},
            {"symbol": "ETH/USD", "side": "sell", "price": 2000.0, "amount": 0.0},
        ]
    )
    assert risk_manager.pending_buy_exposure == 0.0
    assert risk_manager.pending_orders_by_symbol == {}


def test_min_trade_size_rejection(risk_manager):
    result = risk_manager.check_trade_allowed("SMALL", "BUY", 0.05, price=10.0)
    assert result.allowed is False
    assert "below minimum" in result.reason


def test_sell_projected_exposure_paths(risk_manager):
    risk_manager.update_positions(
        {
            "AAA": {"quantity": 9.0, "current_price": 100.0},
            "BBB": {"quantity": 0.5, "current_price": 100.0},
        }
    )

    reduce_result = risk_manager.check_trade_allowed("AAA", "SELL", 5.0, price=100.0)
    assert reduce_result.allowed

    short_result = risk_manager.check_trade_allowed("BBB", "SELL", 5.0, price=100.0)
    assert short_result.allowed is False
    assert "Total exposure" in short_result.reason


def test_bucket_lookup_handles_missing_symbol():
    rm = RiskManager()
    assert rm._get_bucket("") is None
    assert rm._get_bucket("DOGE/USD") is None


def test_net_quantity_for_exposure_sign_edges():
    rm = RiskManager(ignore_baseline_positions=True)
    assert rm._net_quantity_for_exposure(-1.0, 2.0) == -1.0
    assert rm._net_quantity_for_exposure(1.0, -1.0) == 1.0
