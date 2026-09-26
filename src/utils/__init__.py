# ruff: noqa: F401
"""Utils package for organizing utils modules."""

from src.utils.logging import (
    log_checkpoint,
    log_early_stop,
    log_iteration_start,
    log_progress,
    logger,
)
from src.utils.seeding import set_seed

__all__ = [
    "log_checkpoint",
    "log_early_stop",
    "log_iteration_start",
    "log_progress",
    "logger",
    "set_seed",
]

try:
    from src.utils.disruptions import DisruptionGenerator

    __all__.append("DisruptionGenerator")
except ModuleNotFoundError:
    pass

try:
    from .dist import create_dist_dict_from_airports

    __all__.append("create_dist_dict_from_airports")
except ModuleNotFoundError:
    pass

try:
    from .envs import (
        AirlineEnv,
        ClosestPlaneGreedySolver,
        DQNSolver,
        RandomSolver,
        run_unified_execution,
    )

    __all__.extend(
        [
            "AirlineEnv",
            "ClosestPlaneGreedySolver",
            "DQNSolver",
            "RandomSolver",
            "run_unified_execution",
        ]
    )
except ModuleNotFoundError:
    pass

try:
    from .training_engine import (
        get_model_filename,
        initialize_agent,
        reset_agent_exploration,
        train_dqn_iteration,
    )

    __all__.extend(
        [
            "get_model_filename",
            "initialize_agent",
            "reset_agent_exploration",
            "train_dqn_iteration",
        ]
    )
except ModuleNotFoundError:
    pass
