"""Symbol normalization helpers shared across venues."""

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# Common ISO currency codes used to infer FX contracts
COMMON_CURRENCIES = {
    "AUD",
    "CAD",
    "CHF",
    "CNY",
    "EUR",
    "GBP",
    "HKD",
    "JPY",
    "MXN",
    "NZD",
    "SGD",
    "USD",
}

# IBKR default FX exchange
DEFAULT_FX_EXCHANGE = "IDEALPRO"
_SYMBOL_PART_PATTERN = re.compile(r"^[A-Z0-9.\-]+$")


@dataclass(frozen=True)
class SymbolSpec:
    """Normalized representation of a trading symbol and routing metadata."""

    base: str
    quote: str
    symbol: str
    instrument_type: str
    exchange: str | None = None
    primary_exchange: str | None = None


def _validate_symbol_parts(base: str, quote: str) -> Tuple[str, str]:
    base_clean = base.strip().upper()
    quote_clean = quote.strip().upper()
    if not base_clean or not quote_clean:
        raise ValueError("Symbol entries must include base and quote (e.g., BHP/AUD).")
    if not _SYMBOL_PART_PATTERN.match(base_clean):
        raise ValueError(f"Symbol base '{base}' must be alphanumeric (., - allowed).")
    if not _SYMBOL_PART_PATTERN.match(quote_clean):
        raise ValueError(f"Symbol quote '{quote}' must be alphanumeric (., - allowed).")
    return base_clean, quote_clean


def format_symbol(base: str, quote: str) -> str:
    """Format a canonical SYMBOL/QUOTE pair."""
    base_clean, quote_clean = _validate_symbol_parts(base, quote)
    return f"{base_clean}/{quote_clean}"


def normalize_symbol(raw: str) -> str:
    """
    Normalize a raw symbol string to SYMBOL/QUOTE format.

    Raises ValueError with actionable guidance when malformed.
    """
    if raw is None:
        raise ValueError("Symbol entry is required (e.g., BHP/AUD).")
    if not isinstance(raw, str):
        raise ValueError("Symbol entries must be strings like 'BHP/AUD'.")
    trimmed = raw.strip()
    if not trimmed:
        raise ValueError("Symbol entries must not be empty.")
    if "/" not in trimmed:
        raise ValueError(
            f"Symbol '{raw}' must use SYMBOL/QUOTE format (e.g., BHP/AUD). "
            "Update ALLOWED_SYMBOLS to include a quote currency."
        )
    base_part, quote_part = trimmed.split("/", 1)
    if not base_part or not quote_part:
        raise ValueError(
            f"Symbol '{raw}' must include both base and quote (e.g., BHP/AUD)."
        )
    return format_symbol(base_part, quote_part)


def normalize_symbols(raw_symbols: str | Sequence[str]) -> List[str]:
    """
    Normalize and deduplicate a list or comma string of symbols.

    Empty/whitespace entries are ignored; invalid entries raise.
    """
    if isinstance(raw_symbols, str):
        raw_list = [token.strip() for token in raw_symbols.split(",")]
    else:
        raw_list = list(raw_symbols or [])

    normalized: List[str] = []
    seen = set()
    for raw in raw_list:
        if raw is None:
            continue
        if not isinstance(raw, str):
            raise ValueError("Symbol entries must be strings like 'BHP/AUD'.")
        if not raw.strip():
            continue
        symbol = normalize_symbol(raw)
        if symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _is_fx_pair(base: str, quote: str) -> bool:
    return base in COMMON_CURRENCIES and quote in COMMON_CURRENCIES


def infer_instrument_type(
    base: str,
    quote: str,
    *,
    allowed_instrument_types: Iterable[str] | None = None,
    base_currency: str | None = None,
) -> str:
    """
    Infer instrument type (FX vs STK) from base/quote and guard against disallowed types.
    """
    allowed = {token.upper() for token in allowed_instrument_types or []}
    base_upper = base.strip().upper()
    quote_upper = quote.strip().upper()

    instrument = "FX" if _is_fx_pair(base_upper, quote_upper) else "STK"
    if base_currency and quote_upper == base_currency.upper():
        instrument = "STK"

    if allowed and instrument not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(
            f"Instrument type '{instrument}' not permitted for {base_upper}/{quote_upper}. "
            f"Update IB_ALLOWED_INSTRUMENT_TYPES (current allowed: {allowed_list})."
        )
    return instrument


def build_symbol_spec(
    raw_symbol: str,
    instrument_type: str | None = None,
    exchange: str | None = None,
    primary_exchange: str | None = None,
    *,
    allowed_instrument_types: Iterable[str] | None = None,
    default_exchange: str | None = None,
    default_primary_exchange: str | None = None,
    base_currency: str | None = None,
    fx_exchange: str = DEFAULT_FX_EXCHANGE,
) -> SymbolSpec:
    """
    Normalize a symbol and attach routing metadata for Interactive Brokers.

    Defaults:
    - STK: uses default_exchange/primary_exchange
    - FX: uses fx_exchange and omits primary exchange
    """
    symbol = normalize_symbol(raw_symbol)
    base, quote = symbol.split("/", 1)

    inferred_type = (
        instrument_type.strip().upper()
        if instrument_type
        else infer_instrument_type(
            base,
            quote,
            allowed_instrument_types=allowed_instrument_types,
            base_currency=base_currency,
        )
    )
    allowed = {token.upper() for token in allowed_instrument_types or []}
    if allowed and inferred_type not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(
            f"Instrument type '{inferred_type}' not permitted for {symbol}. "
            f"Allowed: {allowed_list}."
        )

    if inferred_type == "FX":
        exchange_val = exchange or fx_exchange
        return SymbolSpec(
            base=base,
            quote=quote,
            symbol=symbol,
            instrument_type="FX",
            exchange=exchange_val,
            primary_exchange=None,
        )

    exchange_val = exchange or default_exchange
    primary_val = primary_exchange or default_primary_exchange
    if not exchange_val:
        raise ValueError(
            f"Symbol {symbol} requires an exchange. "
            "Set IB_EXCHANGE or specify an exchange per symbol."
        )

    return SymbolSpec(
        base=base,
        quote=quote,
        symbol=symbol,
        instrument_type=inferred_type,
        exchange=exchange_val,
        primary_exchange=primary_val,
    )
