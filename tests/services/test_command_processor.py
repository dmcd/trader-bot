import pytest
from unittest.mock import AsyncMock, MagicMock

from trader_bot.services.command_processor import CommandProcessor


@pytest.mark.asyncio
async def test_close_all_positions_command_executes_and_marks():
    db = MagicMock()
    db.get_pending_commands.return_value = [{"id": 1, "command": "CLOSE_ALL_POSITIONS"}]
    db.mark_command_executed = MagicMock()
    close_cb = AsyncMock()
    stop_cb = MagicMock()

    processor = CommandProcessor(db)
    result = await processor.process(close_cb, stop_cb)

    close_cb.assert_awaited_once()
    db.mark_command_executed.assert_called_once_with(1)
    stop_cb.assert_not_called()
    assert result.stop_requested is False


@pytest.mark.asyncio
async def test_stop_bot_command_sets_reason_and_returns_stop():
    db = MagicMock()
    db.get_pending_commands.return_value = [{"id": 3, "command": "STOP_BOT"}]
    db.mark_command_executed = MagicMock()
    close_cb = AsyncMock()
    stop_cb = MagicMock()

    processor = CommandProcessor(db)
    result = await processor.process(close_cb, stop_cb)

    close_cb.assert_not_called()
    db.mark_command_executed.assert_called_once_with(3)
    stop_cb.assert_called_once_with("manual stop")
    assert result.stop_requested is True
    assert result.shutdown_reason == "manual stop"
