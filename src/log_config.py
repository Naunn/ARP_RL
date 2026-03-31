import logging

# Logging configuration for the models
# By default logs are written to a dedicated 'logs' folder next to this module.
# You can override LOG_FILE with an absolute path or set to None for console-only.
import os

# __file__ may not be defined in interactive sessions; fall back to cwd
base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
LOG_DIR = os.path.join(base_dir, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)


def configure_logging(script_name: str = "ARP") -> logging.Logger:
    """Configure logging for a specific script with script-specific log file."""

    # Create script-specific log file
    script_log_file = os.path.join(LOG_DIR, f"{script_name}.log")

    # Create a logger specific to this script
    logger = logging.getLogger(script_name)

    # Only configure if not already configured for this logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create format with script name
        formatter = logging.Formatter(
            f"{script_name} ---------------: %(asctime)s %(levelname)s %(message)s"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(script_log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(script_name: str = "ARP") -> logging.Logger:
    """Get or create a logger configured for the specified script."""
    return configure_logging(script_name)
