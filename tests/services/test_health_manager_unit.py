import logging
from unittest.mock import AsyncMock

import pytest

from trader_bot.services.health_manager import HealthCircuitManager


class _Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:  # pragma: no cover - trivial
        return self.now


def test_pause_window_accumulates_longer_durations():
    clock = _Clock(10.0)
    manager = HealthCircuitManager(monotonic=clock)

    first = manager.request_pause(20)
    assert first == pytest.approx(30.0)

    # Shorter subsequent pauses should not shrink the window
    clock.now = 25.0
    second = manager.request_pause(2)
    assert second == pytest.approx(30.0)
    assert manager.should_pause(clock.now) is True
    assert manager.pause_remaining(clock.now) == pytest.approx(5.0)

    # Past the window trading should resume
    clock.now = 31.0
    assert manager.should_pause(clock.now) is False
    assert manager.pause_remaining(clock.now) == 0.0


@pytest.mark.asyncio
async def test_maybe_reconnect_throttles_calls():
    clock = _Clock(0.0)
    manager = HealthCircuitManager(monotonic=clock, reconnect_cooldown_seconds=30, logger=logging.getLogger("test"))
    bot = type("Bot", (), {})()
    bot.connect_async = AsyncMock()

    first = await manager.maybe_reconnect(bot)
    assert first is True
    bot.connect_async.assert_awaited_once()

    # Within cooldown, no reconnect attempt should happen
    clock.now = 10.0
    second = await manager.maybe_reconnect(bot)
    assert second is False
    assert bot.connect_async.call_count == 1

    # After cooldown expires, reconnect should be attempted again
    clock.now = 45.0
    third = await manager.maybe_reconnect(bot)
    assert third is True
    assert bot.connect_async.call_count == 2
