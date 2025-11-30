"""Interactive Brokers contract resolution helpers."""

from typing import Iterable, Tuple

from ib_insync import Contract, Forex, Stock

from trader_bot.symbols import (
    SymbolSpec,
    build_symbol_spec,
    normalize_symbol,
)


def build_ib_contract(
    raw_symbol: str | SymbolSpec,
    *,
    instrument_type: str | None = None,
    exchange: str | None = None,
    primary_exchange: str | None = None,
    allowed_instrument_types: Iterable[str] | None = None,
    default_exchange: str | None = None,
    default_primary_exchange: str | None = None,
    base_currency: str | None = None,
    fx_exchange: str | None = None,
) -> Tuple[Contract, SymbolSpec]:
    """
    Build an ib_insync Contract from a normalized symbol.

    Returns the contract and the resolved SymbolSpec.
    """
    if isinstance(raw_symbol, SymbolSpec):
        spec = raw_symbol
    else:
        normalized = normalize_symbol(raw_symbol)
        spec = build_symbol_spec(
            normalized,
            instrument_type=instrument_type,
            exchange=exchange,
            primary_exchange=primary_exchange,
            allowed_instrument_types=allowed_instrument_types,
            default_exchange=default_exchange,
            default_primary_exchange=default_primary_exchange,
            base_currency=base_currency,
            fx_exchange=fx_exchange or "IDEALPRO",
        )

    if spec.instrument_type == "FX":
        contract = Forex(f"{spec.base}{spec.quote}")
        if spec.exchange:
            contract.exchange = spec.exchange
        return contract, spec

    if spec.instrument_type == "STK":
        contract = Stock(
            symbol=spec.base,
            exchange=spec.exchange,
            currency=spec.quote,
            primaryExchange=spec.primary_exchange,
        )
        return contract, spec

    raise ValueError(f"Unsupported instrument type '{spec.instrument_type}' for {spec.symbol}.")
