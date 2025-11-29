import logging
import sys
import os

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
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Console Log (console.log) - Technical debug log
    # DEBUG and above, detailed format, captures everything
    console_handler = logging.FileHandler('console.log', mode='w') # Clear log on startup
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # 2. Real Terminal Output
    # So the user still sees what's going on
    stream_handler = logging.StreamHandler(original_stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(detailed_formatter)
    logger.addHandler(stream_handler)

    # 3. Bot Actions Log (bot.log) - User-friendly log
    # Create a separate logger specifically for user-facing bot actions
    bot_actions_logger = logging.getLogger('bot_actions')
    bot_actions_logger.setLevel(logging.INFO)
    bot_actions_logger.propagate = False  # Don't propagate to root logger
    
    bot_handler = logging.FileHandler('bot.log', mode='w')  # Overwrite on startup
    bot_handler.setLevel(logging.INFO)
    bot_handler.setFormatter(simple_formatter)
    bot_actions_logger.addHandler(bot_handler)

    # 4. Telemetry Log (telemetry.log) - structured JSON per loop
    telemetry_logger = logging.getLogger('telemetry')
    telemetry_logger.setLevel(logging.INFO)
    telemetry_logger.propagate = False
    # Reset telemetry log each startup to keep sessions isolated
    telemetry_handler = logging.FileHandler('telemetry.log', mode='w')
    telemetry_handler.setLevel(logging.INFO)
    telemetry_handler.setFormatter(logging.Formatter('%(message)s'))
    telemetry_logger.addHandler(telemetry_handler)

    # Redirect stdout and stderr to the logger
    # We use specific loggers for these so we can identify the source
    sys.stdout = LoggerWriter(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger('STDERR'), logging.ERROR)

    logging.info("Logging initialized. stdout/stderr redirected to console.log")
    
    return bot_actions_logger
