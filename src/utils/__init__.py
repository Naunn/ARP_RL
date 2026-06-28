"""Utils package for organizing utils modules."""

from src.utils.disruptions import DisruptionGenerator
from src.utils.logging import (
    log_checkpoint,
    log_early_stop,
    log_iteration_start,
    log_progress,
    logger,
)

from .dist import create_dist_dict_from_airports
from .envs import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
    run_unified_execution,
)
from .fleet import build_planes
from .schedule import (
    build_flight_pool,
    check_global_feasibility,
    generate_random_flights,
    generate_trap_schedule,
)
from .training_engine import (
    get_model_filename,
    initialize_dqn_agent,
    reset_agent_exploration,
    setup_checkpoint_dir,
    train_dqn_iteration,
)

__all__ = [
    "DisruptionGenerator",
    "create_dist_dict_from_airports",
    "AirlineEnv",
    "ClosestPlaneGreedySolver",
    "DQNSolver",
    "RandomSolver",
    "run_unified_execution",
    "build_planes",
    "build_flight_pool",
    "check_global_feasibility",
    "log_checkpoint",
    "log_early_stop",
    "log_iteration_start",
    "log_progress",
    "logger",
    "get_model_filename",
    "initialize_dqn_agent",
    "reset_agent_exploration",
    "setup_checkpoint_dir",
    "train_dqn_iteration",
    "generate_random_flights",
    "generate_trap_schedule",
]
