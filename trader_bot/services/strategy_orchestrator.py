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
        daily_loss_pct_limit: float,
        max_daily_loss: float,
        trading_mode: str,
        close_positions_cb: Callable[[], Awaitable[None]],
        set_shutdown_reason: Callable[[str], None],
    ) -> RiskCheckResult:
        """
        Apply daily loss limits and trigger shutdown if breached.
        Returns whether the loop should halt and whether to flip the kill switch.
        """
        sod_equity = self.risk_manager.start_of_day_equity or 0.0
        loss_percent = (self.risk_manager.daily_loss / sod_equity * 100) if sod_equity else 0.0

        if sod_equity and loss_percent > daily_loss_pct_limit:
            reason = f"daily loss {loss_percent:.2f}% > {daily_loss_pct_limit}%"
            self.logger.error(f"Max daily loss exceeded: {loss_percent:.2f}% > {daily_loss_pct_limit}%. Stopping loop.")
            set_shutdown_reason(reason)
            self.actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded ({loss_percent:.2f}%)")
            await close_positions_cb()
            return RiskCheckResult(should_stop=True, kill_switch=True, shutdown_reason=reason)

        if self.risk_manager.daily_loss > max_daily_loss:
            reason = f"daily loss ${self.risk_manager.daily_loss:.2f} > ${max_daily_loss:.2f}"
            if trading_mode == "PAPER":
                self.logger.warning(
                    f"Sandbox: Absolute daily loss exceeded (${self.risk_manager.daily_loss:.2f} > ${max_daily_loss:.2f}), but continuing loop."
                )
                return RiskCheckResult(should_stop=False, kill_switch=False)

            self.logger.error(f"Max daily loss exceeded: ${self.risk_manager.daily_loss:.2f} > ${max_daily_loss:.2f}. Stopping loop.")
            set_shutdown_reason(reason)
            self.actions_logger.info(f"🛑 Trading Stopped: Daily loss limit exceeded (${self.risk_manager.daily_loss:.2f})")
            await close_positions_cb()
            return RiskCheckResult(should_stop=True, kill_switch=True, shutdown_reason=reason)

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
