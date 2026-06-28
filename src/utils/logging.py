"""Unified logging engine and training telemetry formatting for the pipeline."""

import logging
import os

# --- SETUP GLOBAL LOGGING PROPERTIES ---
BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(script_name: str = "plane_assignment") -> logging.Logger:
    """Retrieves or spins up a standardized console + file logger stream."""
    logger = logging.getLogger(script_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(f"{script_name} ---------------: %(asctime)s %(levelname)s %(message)s")

        # Console Output Stream
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File Output Stream
        file_out = logging.FileHandler(os.path.join(LOG_DIR, f"{script_name}.log"), mode="w")
        file_out.setFormatter(formatter)
        logger.addHandler(file_out)

    return logger


# Primary instance handle used across modules
logger = get_logger("plane_assignment")


# --- STREAMLINED TRAINING METRICS FORMATTERS ---
def log_iteration_start(iteration: int, total_iterations: int):
    logger.info("\n" + "=" * 80)
    logger.info(f"STARTING SCHEDULE REGENERATION ITERATION {iteration}/{total_iterations}")
    logger.info("=" * 80)


def log_progress(
    iteration: int,
    progress_pct: float,
    epsilon: float,
    avg_score: float,
    eta_str: str,
    model_name: str = "MODEL",
    training_name: str | None = None,
):
    logger.info(
        f"[{model_name}] (Training mode: {training_name}) [Iter {iteration}] Progress: {progress_pct:>5.1f}% | "
        f"Epsilon: {epsilon:.4f} | Avg Profit: ${avg_score:>10.0f} | ETA: {eta_str}"
    )


def log_checkpoint(best_rolling_profit: float, episode_num: int):
    logger.info(f"[CHECKPOINT] Episode {episode_num}: New best rolling avg: ${best_rolling_profit:,.0f}")


def log_early_stop(episode_num: int, total_episodes: int, patience: int, best_rolling_profit: float):
    logger.info(
        f"\n[EARLY STOP] Terminated at Episode {episode_num}/{total_episodes}. "
        f"No improvement for {patience} episodes. Final policy rolling average: ${best_rolling_profit:,.0f}."
    )


def log_feasibility(iteration: int, utilization: float, max_req_planes: int, num_planes: int):
    logger.info(f"Schedule Iteration {iteration} Feasibility:")
    logger.info(f"Global Utilization: {utilization:.1f}% | Peak Concurrency: {max_req_planes} planes")
    if max_req_planes > num_planes or utilization > 100:
        logger.warning("Status: CURRENT ENVIRONMENT IS UNSOLVABLE - Triage mode active.")
    else:
        logger.info("Status: ENVIRONMENT IS SOLVABLE.")
