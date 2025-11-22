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
    - bot.log: INFO level (for Dashboard)
    - console.log: DEBUG level (for Debugging, captures stdout/stderr)
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
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Dashboard Log (bot.log)
    # Only INFO and above, simplified format
    bot_handler = logging.FileHandler('bot.log', mode='a')
    bot_handler.setLevel(logging.INFO)
    bot_handler.setFormatter(simple_formatter)
    logger.addHandler(bot_handler)

    # 2. Debug Log (console.log)
    # DEBUG and above, detailed format, captures everything
    console_handler = logging.FileHandler('console.log', mode='w') # Overwrite for fresh debug log each run
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # 3. Real Terminal Output
    # So the user still sees what's going on
    stream_handler = logging.StreamHandler(original_stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(simple_formatter)
    logger.addHandler(stream_handler)

    # Redirect stdout and stderr to the logger
    # We use specific loggers for these so we can identify the source
    sys.stdout = LoggerWriter(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger('STDERR'), logging.ERROR)

    logging.info("Logging initialized. stdout/stderr redirected to console.log")
