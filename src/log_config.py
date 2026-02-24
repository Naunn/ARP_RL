import logging

# Logging configuration for the ARP model
# By default logs are written to a dedicated 'logs' folder next to this module.
# You can override LOG_FILE with an absolute path or set to None for console-only.
import os

# __file__ may not be defined in interactive sessions; fall back to cwd
base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
LOG_DIR = os.path.join(base_dir, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "arp_run.log")


def configure_logging():
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE, mode="w"))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


# module-level logger; import this from other modules
logger = configure_logging()
