import logging
import sys
import os

class RunContextFilter(logging.Filter):
    """Injects portfolio/run context into log records for traceability."""

    def __init__(self):
        super().__init__()
        self.portfolio_id = os.getenv("PORTFOLIO_ID")
        self.run_id = os.getenv("RUN_ID")

    def set_context(self, portfolio_id=None, run_id=None):
        if portfolio_id is not None:
            self.portfolio_id = portfolio_id
        if run_id is not None:
            self.run_id = run_id

    def filter(self, record):
        record.portfolio_id = self.portfolio_id or "-"
        record.run_id = self.run_id or "-"
        return True


_RUN_CONTEXT_FILTER = RunContextFilter()


def set_logging_context(portfolio_id=None, run_id=None):
    """Update the global logging context so records carry portfolio/run ids."""
    _RUN_CONTEXT_FILTER.set_context(portfolio_id, run_id)


class LoggerWriter:
    """
    Redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        if message != '\n':
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def setup_logging():
    """
    Configures the logging system.
    - bot.log: User-friendly log showing only trading decisions and reasoning (via bot_actions logger)
    - console.log: Technical DEBUG level log (for Debugging, captures stdout/stderr)
    - telemetry.log: Structured JSON per-loop telemetry for analysis
    - Terminal: INFO level
    """
    # Save original streams to avoid infinite loops
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # Capture everything at root
    
    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatters
    simple_formatter = logging.Formatter('%(asctime)s - %(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - portfolio=%(portfolio_id)s run=%(run_id)s - %(message)s')

    test_mode = (
        "PYTEST_RUNNING" in os.environ
        or "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
    )

    log_dir = "logs/test" if test_mode else "."
    os.makedirs(log_dir, exist_ok=True)

    # 1. Console Log (console*.log) - Technical debug log
    # DEBUG and above, detailed format, captures everything
    log_filename = "console_test.log" if test_mode else "console.log"
    console_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode='w') # Clear log on startup
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)
    console_handler.addFilter(_RUN_CONTEXT_FILTER)
    logger.addHandler(console_handler)

    # 2. Real Terminal Output
    # So the user still sees what's going on
    stream_handler = logging.StreamHandler(original_stdout)
    stream_handler.setLevel(logging.DEBUG if test_mode else logging.INFO)
    stream_handler.setFormatter(detailed_formatter)
    stream_handler.addFilter(_RUN_CONTEXT_FILTER)
    logger.addHandler(stream_handler)

    # 3. Bot Actions Log (bot.log) - User-friendly log
    # Create a separate logger specifically for user-facing bot actions
    bot_actions_logger = logging.getLogger('bot_actions')
    bot_actions_logger.setLevel(logging.INFO)
    bot_actions_logger.propagate = False  # Don't propagate to root logger
    
    bot_log_filename = "bot_test.log" if test_mode else "bot.log"
    bot_handler = logging.FileHandler(os.path.join(log_dir, bot_log_filename), mode='w')  # Overwrite on startup
    bot_handler.setLevel(logging.INFO)
    bot_handler.setFormatter(simple_formatter)
    bot_handler.addFilter(_RUN_CONTEXT_FILTER)
    bot_actions_logger.addHandler(bot_handler)

    # 4. Telemetry Log (telemetry.log) - structured JSON per loop
    telemetry_logger = logging.getLogger('telemetry')
    telemetry_logger.setLevel(logging.INFO)
    telemetry_logger.propagate = False
    # Reset telemetry log each startup to keep runs isolated
    telemetry_log_filename = "telemetry_test.log" if test_mode else "telemetry.log"
    telemetry_handler = logging.FileHandler(os.path.join(log_dir, telemetry_log_filename), mode='w')
    telemetry_handler.setLevel(logging.INFO)
    telemetry_handler.setFormatter(logging.Formatter('%(message)s'))
    telemetry_handler.addFilter(_RUN_CONTEXT_FILTER)
    telemetry_logger.addHandler(telemetry_handler)

    # Redirect stdout and stderr to the logger
    # We use specific loggers for these so we can identify the source
    sys.stdout = LoggerWriter(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger('STDERR'), logging.ERROR)

    logging.info("Logging initialized. stdout/stderr redirected to console.log")
    
    return bot_actions_logger
