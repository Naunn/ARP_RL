"""Unified logging for the pipeline: one logger, one line format, and a few structural helpers.

Layout of a run's log:
    === <section> ======   run / iteration / run-of-a-sweep boundaries   (log_section)
    --- <subsection> ---   phases inside them: training, evaluation, ... (log_subsection)
      <indented lines>     per-agent progress and table rows
"""

import logging
import os

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

RULE_WIDTH = 100


class LineFormatter(logging.Formatter):
    """`<time>  <message>`; the level is shown only when it isn't INFO, and every line of a
    multi-line message gets the prefix so tables stay aligned."""

    def __init__(self, datefmt: str):
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        prefix = self.formatTime(record, self.datefmt) + "  "
        if record.levelno != logging.INFO:
            prefix += f"{record.levelname}: "
        return "\n".join(prefix + line for line in record.getMessage().split("\n"))


CONSOLE_FORMATTER = LineFormatter("%H:%M:%S")
FILE_FORMATTER = LineFormatter("%Y-%m-%d %H:%M:%S")


def get_logger(script_name: str = "plane_assignment") -> logging.Logger:
    """Retrieves or spins up the standardized console + file logger."""
    logger = logging.getLogger(script_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setFormatter(CONSOLE_FORMATTER)
        logger.addHandler(console)

        file_out = logging.FileHandler(os.path.join(LOG_DIR, f"{script_name}.log"), mode="w")
        file_out.setFormatter(FILE_FORMATTER)
        logger.addHandler(file_out)

    return logger


# Primary instance handle used across modules
logger = get_logger("plane_assignment")


def _rule(char: str, title: str) -> str:
    head = f"{char * 3} {title} "
    return head + char * max(3, RULE_WIDTH - len(head))


def log_section(title: str) -> None:
    """Top-level boundary: a run, an iteration, one run of a multi-run sweep."""
    logger.info(_rule("=", title))


def log_subsection(title: str) -> None:
    """A phase inside a section: training, evaluation, disruption step, ..."""
    logger.info(_rule("-", title))


def log_progress(
    label: str,
    episode: int,
    n_episodes: int,
    epsilon: float,
    avg_reward: float,
    window: int,
    eta_str: str,
) -> None:
    """One training-progress line; avg_reward is the mean episode reward over the last `window` episodes."""
    width = len(str(n_episodes))
    logger.info(
        f"  {label:<20} ep {episode:>{width}}/{n_episodes} ({episode / n_episodes:>4.0%}) | "
        f"eps {epsilon:.3f} | avg reward (last {window:>3} ep) {avg_reward:>13,.0f} | ETA {eta_str}"
    )


def log_checkpoint(label: str, best_rolling_reward: float, episode: int) -> None:
    logger.info(
        f"  {label:<20} ep {episode}: new best rolling avg reward {best_rolling_reward:,.0f} -> checkpoint saved"
    )


def log_early_stop(label: str, episode: int, total_episodes: int, patience: int, best_rolling_reward: float) -> None:
    logger.info(
        f"  {label:<20} early stop at ep {episode}/{total_episodes}: no improvement for {patience} episodes "
        f"(best rolling avg reward {best_rolling_reward:,.0f})"
    )
