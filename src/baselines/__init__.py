"""Non-learning reference solvers ("baselines") the RL agents are compared against.

- `cplex_solver.CplexSolver`: the optimal baseline (exact CPLEX model of AirlineEnv's reward);
  needs the optional cplex extra (`uv sync --extra cplex`).
- Random and greedy baselines live next to the environment: src.utils.envs.RandomSolver and
  ClosestPlaneGreedySolver.

This package doesn't import cplex_solver eagerly, so code that doesn't use CPLEX works without
the cplex extra installed.
"""


class BaselineSolverError(RuntimeError):
    """An external optimization baseline (CPLEX, or e.g. Gurobi later) failed on a schedule.

    retryable=True means the failure is specific to this schedule (no solution returned, numerical
    trouble, a plan that disagrees with the env), so experiments draw a new schedule and retry the
    same iteration. retryable=False means retrying cannot help (e.g. the solver's license size
    limit), so the run stops.
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable
