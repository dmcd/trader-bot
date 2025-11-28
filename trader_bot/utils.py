"""Utility helpers for cross-module reuse."""

from typing import Any, Dict


def get_client_order_id(order: Dict[str, Any] | None) -> str:
    """
    Extract a client order id from common ccxt/exchange shapes.

    Handles camelCase, snake_case, and nested info fields.
    """
    if not order:
        return ""
    return str(
        order.get("clientOrderId")
        or order.get("client_order_id")
        or order.get("client_order")
        or order.get("info", {}).get("clientOrderId")
        or order.get("info", {}).get("client_order_id")
        or order.get("info", {}).get("client_order")
        or ""
    )
