import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

bot_actions_logger = logging.getLogger("bot_actions")
logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    stop_requested: bool = False
    shutdown_reason: str | None = None


class CommandProcessor:
    """Handle dashboard-issued commands outside the runner loop."""

    def __init__(self, db):
        self.db = db

    async def process(
        self,
        close_positions_cb: Callable[[], Awaitable[None]],
        stop_cb: Callable[[str], None],
    ) -> CommandResult:
        """Execute pending commands and return whether the loop should stop."""
        pending_commands = self.db.get_pending_commands()
        result = CommandResult()

        for cmd in pending_commands:
            command = cmd.get("command")
            command_id = cmd.get("id")

            if command == "CLOSE_ALL_POSITIONS":
                logger.info("Executing command: CLOSE_ALL_POSITIONS")
                bot_actions_logger.info("🛑 Manual Command: Closing all positions...")
                await close_positions_cb()
                bot_actions_logger.info("✅ All positions closed")
                self._mark_executed_safely(command_id)

            elif command == "STOP_BOT":
                logger.info("Executing command: STOP_BOT")
                bot_actions_logger.info("🛑 Manual Command: Stopping bot...")
                self._mark_executed_safely(command_id)
                stop_cb("manual stop")
                result.stop_requested = True
                result.shutdown_reason = "manual stop"
                break

        return result

    def _mark_executed_safely(self, command_id: int | None) -> None:
        if command_id is None:
            return
        try:
            self.db.mark_command_executed(command_id)
        except Exception as exc:
            logger.warning(f"Could not mark command {command_id} executed: {exc}")

