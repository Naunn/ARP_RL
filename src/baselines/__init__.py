"""Non-learning reference solvers ("baselines") the RL agents are compared against.

- `optimal.OptimalBaselineSolver`: the optimal baseline -- an exact model of AirlineEnv's reward,
  solved by Gurobi (`gurobi_solver.py`, `uv sync --extra gurobi`) or CPLEX (`cplex_solver.py`,
  `uv sync --extra cplex`), chosen with config.BASELINE_SOLVER_CONFIG["solver"]. Verify it with
  `python -m src.baselines.verify_baseline`.
- Random and greedy baselines live next to the environment: src.utils.envs.RandomSolver and
  ClosestPlaneGreedySolver.

Solver packages are imported only when that solver is used, so neither is required otherwise.
"""


class BaselineSolverError(RuntimeError):
    """An external optimization baseline (Gurobi, CPLEX, ...) failed on a schedule.

    retryable=True means the failure is specific to this schedule (no solution returned, numerical
    trouble, a plan that disagrees with the env), so experiments draw a new schedule and retry the
    same iteration. retryable=False means retrying cannot help (e.g. the solver's license size
    limit), so the run stops.
    """

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable
