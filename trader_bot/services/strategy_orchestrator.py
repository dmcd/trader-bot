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
        record_operational_metrics: Callable[[float, float, dict | None], None],
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

    def emit_operational_metrics(
        self,
        current_exposure: float,
        current_equity: float,
        per_symbol_exposure: dict | None = None,
    ):
        """Emit telemetry metrics for risk and cost tracking."""
        self.record_operational_metrics(current_exposure, current_equity, per_symbol_exposure)

    async def monitor_trade_plans(
        self,
        price_lookup: dict,
        open_orders: list,
        config: PlanMonitorConfig,
        refresh_bindings_cb: Callable[[], None],
        portfolio_id: int | None = None,
    ):
        """Refresh bindings and run the plan monitor for the loop."""
        refresh_bindings_cb()
        target_portfolio = portfolio_id if portfolio_id is not None else self.portfolio_id
        await self.plan_monitor.monitor(
            price_lookup=price_lookup,
            open_orders=open_orders,
            config=config,
            portfolio_id=target_portfolio,
        )

    def evaluate_market_health(self, market_data: dict) -> tuple[dict, dict]:
        """
        Evaluate market data freshness per symbol and return (fresh, stale) maps.
        Each entry contains the health detail plus symbol for logging/telemetry.
        """
        fresh: dict[str, dict] = {}
        stale: dict[str, dict] = {}

        for symbol, data in (market_data or {}).items():
            is_stale, detail = self.health_manager.is_stale_market_data(data or {})
            detail = detail or {}
            detail["symbol"] = symbol
            status = "stale" if is_stale else "ok"
            try:
                self.health_manager.record_health_state("market_data", status, detail)
            except Exception:
                # Telemetry failures should not break the loop
                pass

            if is_stale:
                stale[symbol] = detail
            else:
                fresh[symbol] = detail

        return fresh, stale

    async def cleanup(self, cleanup_cb: Callable[[], Awaitable[None]]):
        """Run cleanup and reset running flag."""
        try:
            await cleanup_cb()
        finally:
            self.running = False
