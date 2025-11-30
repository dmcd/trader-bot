import pytest

from trader_bot.ib_contracts import build_ib_contract
from trader_bot.symbols import (
    DEFAULT_FX_EXCHANGE,
    build_symbol_spec,
    normalize_symbol,
    normalize_symbols,
)


def test_normalize_symbol_enforces_format_and_uppercase():
    assert normalize_symbol(" bhp/aud ") == "BHP/AUD"
    assert normalize_symbol("aud/usd") == "AUD/USD"
    with pytest.raises(ValueError):
        normalize_symbol("BHP")
    with pytest.raises(ValueError):
        normalize_symbol("BHP/")
    with pytest.raises(ValueError):
        normalize_symbol(123)  # type: ignore[arg-type]


def test_normalize_symbols_dedupes_and_filters_blanks():
    symbols = normalize_symbols(["bhp/aud", "AUD/USD", "bhp/aud", " "])
    assert symbols == ["BHP/AUD", "AUD/USD"]


def test_build_symbol_spec_infers_stock_defaults():
    spec = build_symbol_spec(
        "bhp/aud",
        allowed_instrument_types=["STK", "FX"],
        default_exchange="ASX",
        default_primary_exchange="SMART",
        base_currency="AUD",
    )
    assert spec.symbol == "BHP/AUD"
    assert spec.instrument_type == "STK"
    assert spec.exchange == "ASX"
    assert spec.primary_exchange == "SMART"


def test_build_symbol_spec_infers_fx_and_fx_exchange():
    spec = build_symbol_spec(
        "AUD/USD",
        allowed_instrument_types=["STK", "FX"],
        default_exchange="SMART",
        default_primary_exchange="ASX",
        base_currency="AUD",
    )
    assert spec.instrument_type == "FX"
    assert spec.exchange == DEFAULT_FX_EXCHANGE
    assert spec.primary_exchange is None


def test_build_symbol_spec_rejects_disallowed_type():
    with pytest.raises(ValueError):
        build_symbol_spec(
            "AUD/USD",
            allowed_instrument_types=["STK"],
            default_exchange="SMART",
            base_currency="AUD",
        )


def test_build_ib_contract_maps_stock_and_fx_contracts():
    stock_contract, stock_spec = build_ib_contract(
        "BHP/AUD",
        allowed_instrument_types=["STK", "FX"],
        default_exchange="ASX",
        default_primary_exchange="SMART",
        base_currency="AUD",
    )
    assert stock_spec.symbol == "BHP/AUD"
    assert stock_contract.secType == "STK"
    assert stock_contract.symbol == "BHP"
    assert stock_contract.exchange == "ASX"
    assert stock_contract.currency == "AUD"
    assert stock_contract.primaryExchange == "SMART"

    fx_contract, fx_spec = build_ib_contract(
        "AUD/USD",
        allowed_instrument_types=["STK", "FX"],
        default_exchange="SMART",
        base_currency="AUD",
        fx_exchange="IDEALPRO",
    )
    assert fx_spec.instrument_type == "FX"
    assert fx_contract.secType == "CASH"
    assert fx_contract.symbol == "AUD"
    assert fx_contract.currency == "USD"
    assert fx_contract.exchange == "IDEALPRO"
