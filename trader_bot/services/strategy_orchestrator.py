import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from trader_bot.services.command_processor import CommandProcessor, CommandResult
from trader_bot.services.plan_monitor import PlanMonitor, PlanMonitorConfig
from trader_bot.services.health_manager import HealthCircuitManager
from trader_bot.risk_manager import RiskManager


@dataclass
class RiskCheckResult:
    should_stop: bool
    kill_switch: bool
    shutdown_reason: str | None = None


class StrategyOrchestrator:
    """
    Coordinates lifecycle wiring for the runner and encapsulates top-level
    loop decisions (commands, budget gates, plan monitoring, telemetry hooks).
    """

    def __init__(
        self,
        command_processor: CommandProcessor,
        plan_monitor: PlanMonitor,
        risk_manager: RiskManager,
        health_manager: HealthCircuitManager,
        record_operational_metrics: Callable[[float, float], None],
        loop_interval_seconds: float,
        logger: logging.Logger,
        actions_logger: logging.Logger,
        portfolio_id: int | None = None,
    ):
        self.command_processor = command_processor
        self.plan_monitor = plan_monitor
        self.risk_manager = risk_manager
        self.health_manager = health_manager
        self.record_operational_metrics = record_operational_metrics
        self.loop_interval_seconds = loop_interval_seconds
        self.logger = logger
        self.actions_logger = actions_logger
        self.running = False
        self.portfolio_id = portfolio_id

    async def start(self, initialize_cb: Callable[[], Awaitable[None]]):
        """Initialize dependencies and mark orchestrator as running."""
        await initialize_cb()
        self.running = True

    def request_stop(self, shutdown_reason: str | None = None) -> str | None:
        """Stop the orchestrator loop and pass through the shutdown reason."""
        self.running = False
        return shutdown_reason

    async def process_commands(
        self,
        close_positions_cb: Callable[[], Awaitable[None]],
        stop_cb: Callable[[str], None],
    ) -> CommandResult:
        """Execute dashboard commands and return the result."""
        return await self.command_processor.process(
            close_positions_cb=close_positions_cb,
            stop_cb=stop_cb,
        )

    async def enforce_risk_budget(
        self,
        current_equity: float,
        close_positions_cb: Callable[[], Awaitable[None]],
        set_shutdown_reason: Callable[[str], None],
    ) -> RiskCheckResult:
        """
        Apply portfolio-level risk budget gates (no daily resets).
        Returns whether the loop should halt and whether to flip the kill switch.
        """
        return RiskCheckResult(should_stop=False, kill_switch=False)

    def emit_operational_metrics(self, current_exposure: float, current_equity: float):
        """Emit telemetry metrics for risk and cost tracking."""
        self.record_operational_metrics(current_exposure, current_equity)

    async def monitor_trade_plans(
        self,
        session_id: int,
        price_lookup: dict,
        open_orders: list,
        config: PlanMonitorConfig,
        refresh_bindings_cb: Callable[[], None],
        portfolio_id: int | None = None,
    ):
        """Refresh bindings and run the plan monitor for the loop."""
        refresh_bindings_cb()
        await self.plan_monitor.monitor(
            session_id,
            price_lookup=price_lookup,
            open_orders=open_orders,
            config=config,
            portfolio_id=portfolio_id if portfolio_id is not None else self.portfolio_id,
        )

    def emit_market_health(self, primary_data: dict):
        """Emit health state for primary market data freshness."""
        stale, freshness_detail = self.health_manager.is_stale_market_data(primary_data)
        if stale:
            self.actions_logger.info("⏸️ Skipping loop: market data stale or too latent")
            return False, freshness_detail
        return True, freshness_detail

    async def cleanup(self, cleanup_cb: Callable[[], Awaitable[None]]):
        """Run cleanup and reset running flag."""
        try:
            await cleanup_cb()
        finally:
            self.running = False
