"""Solver-independent core of the optimal baseline: an exact model of AirlineEnv's reward.

The model (implemented once per solver API, in cplex_solver.py and gurobi_solver.py) finds the
plane assignment that maximizes the environment's total reward for a schedule, with the reward
rules mirrored from AirlineEnv.step() term by term:

- Flights are processed in list order; each plane starts at t=0 at its initial_airport and flies
  ONE sequence (plane p can fly g then f only if g precedes f in the list).
- A start-time variable per flight: >= its scheduled start and >= its plane's ready time (previous
  actual arrival, or t=0 at home, plus relocation), so delays propagate down each plane's sequence.
- Each flight's cost is the env's min(30000, linear costs + min(delay penalty, 20000)) with
  delay penalty d * pax * penalty_per_min * (1 + min(d/60, 2)). Every flight picks one of three
  modes -- "normal" (pays the full delay penalty), "capped" (pays 20000; its delay is tracked but
  not penalized) or "floored" (whole flight costs 30000) -- and minimization picks the cheapest,
  which is exactly that min(...). Without clipping only "normal" exists. The penalty's d^2 part
  sits in the (convex) objective, so the model is a MIQP with only linear/indicator constraints.
- Fixed cost on a plane's first flight only; hourly cost on relocation + flight time; per-km
  relocation penalty (incl. from home base); the env's capacity_slack_penalty; under-capacity
  planes allowed with full revenue; missing distances default to DEFAULT_DIST_KM.
Every cost term is nondecreasing in delay, so minimizing over start times recovers the env's
max(...) start times exactly for the chosen assignment. Conditional constraints are indicator
constraints (enforced exactly), not Big-M ones (which leak through integrality tolerance at these
cost magnitudes).

All model coefficients come from `build_assignment_data`, so the solver backends only translate
one formulation and cannot drift apart from each other. Plans go through `plan_schedule`, which
caches them per schedule and replays each new plan through the real environment, raising
BaselineSolverError if the reward differs from the prediction -- so a mis-solve, or an AirlineEnv
reward change this model doesn't mirror, can never silently produce a wrong baseline number.

Only plans the solver has PROVEN optimal are accepted unless BASELINE_SOLVER_CONFIG
["require_optimal"] is False. Proving optimality gets hard fast: ~10 flights take about a second,
20 flights x 3 planes can run for many minutes without closing the gap.
"""

import copy
import importlib
from dataclasses import dataclass, field

from src.baselines import BaselineSolverError
from src.config import BASELINE_SOLVER_CONFIG
from src.utils.envs import (
    DEFAULT_DIST_KM,
    RELOCATION_PENALTY_PER_KM,
    REWARD_FLOOR,
    AirlineEnv,
    BaseSolver,
    capacity_slack_penalty,
)

STEP_COST_CAP = -REWARD_FLOOR  # the env's per-flight reward floor, as a cap on per-flight cost

# solver name -> (module, function, display name); modules are imported only when used, so each
# solver's package is needed only if that solver is selected.
SOLVERS = {
    "cplex": ("src.baselines.cplex_solver", "solve_with_cplex", "CPLEX"),
    "gurobi": ("src.baselines.gurobi_solver", "solve_with_gurobi", "Gurobi"),
}


@dataclass
class SolverPlan:
    planes: list[str]  # planes[i] is the plane assigned to flights[i]
    starts_from_home: list[bool]  # True where flights[i] is its plane's first flight
    objective: float  # minimized total cost, i.e. -(predicted total env reward)
    proven_optimal: bool  # False only if accepted with require_optimal=False
    gap: float  # the solver's relative optimality gap (0 when proven optimal)
    solve_seconds: float
    solver: str

    @property
    def predicted_reward(self) -> float:
        return -self.objective


@dataclass
class AssignmentData:
    """Every coefficient of the model, computed from AirlineEnv's rules; shared by all backends."""

    n: int
    planes: list[str]
    starts: list[float]  # scheduled start of each flight
    revenue: list[float]
    delay_k: list[float]  # delay penalty per minute (pax * penalty_per_min) of each flight
    arcs: list[tuple[str, int, int]]  # (p, g, i): plane p flies g, then i (g listed before i)
    flight_min: dict[tuple[str, int], float] = field(default_factory=dict)  # (p, i) -> minutes flying i
    home_min: dict[tuple[str, int], float] = field(default_factory=dict)  # (p, i) -> minutes home -> i's origin
    arc_min: dict[tuple[str, int, int], float] = field(default_factory=dict)  # (p, g, i) -> minutes g's dest -> i
    flight_cost: dict[tuple[str, int], float] = field(default_factory=dict)  # operating + empty-seat cost
    home_cost: dict[tuple[str, int], float] = field(default_factory=dict)  # fixed + relocation from home
    arc_cost: dict[tuple[str, int, int], float] = field(default_factory=dict)  # relocation g -> i


def build_assignment_data(
    flights: list[dict], plane_configs: dict, dist_dict: dict, penalty_per_min: float
) -> AssignmentData:
    n, planes = len(flights), list(plane_configs)

    def dist(a: str, b: str) -> float:
        return float(dist_dict.get((a, b), DEFAULT_DIST_KM))

    def minutes(p: str, km: float) -> float:
        return km / (plane_configs[p]["speed"] / 60)

    def operating_cost(p: str, km: float) -> float:
        return plane_configs[p]["hourly_cost"] * minutes(p, km) / 60

    data = AssignmentData(
        n=n,
        planes=planes,
        starts=[f["start"] for f in flights],
        revenue=[float(f.get("total_ticket_price", 0.0)) for f in flights],
        delay_k=[f["pass"] * penalty_per_min for f in flights],
        arcs=[(p, g, i) for p in planes for i in range(n) for g in range(i)],
    )
    for p in planes:
        for i, f in enumerate(flights):
            flight_km = dist(f["origin"], f["dest"])
            home_km = dist(plane_configs[p]["initial_airport"], f["origin"])
            data.flight_min[p, i] = minutes(p, flight_km)
            data.home_min[p, i] = minutes(p, home_km)
            data.flight_cost[p, i] = operating_cost(p, flight_km) + capacity_slack_penalty(f, plane_configs[p]["seats"])
            data.home_cost[p, i] = (
                plane_configs[p]["fixed_cost"] + operating_cost(p, home_km) + RELOCATION_PENALTY_PER_KM * home_km
            )
    for p, g, i in data.arcs:
        km = dist(flights[g]["dest"], flights[i]["origin"])
        data.arc_min[p, g, i] = minutes(p, km)
        data.arc_cost[p, g, i] = operating_cost(p, km) + RELOCATION_PENALTY_PER_KM * km
    return data


def solver_display_name(solver: str | None = None) -> str:
    return SOLVERS[solver or BASELINE_SOLVER_CONFIG["solver"]][2]


def solve_assignment(
    flights: list[dict],
    plane_configs: dict,
    dist_dict: dict,
    penalty_per_min: float,
    use_clipping: bool,
    solver: str | None = None,
    time_limit_s: float | None = None,
    require_optimal: bool | None = None,
) -> SolverPlan:
    """Optimal plane-per-flight assignment for AirlineEnv's reward (see module docstring).

    solver / time_limit_s / require_optimal default to BASELINE_SOLVER_CONFIG. With
    require_optimal, a plan the solver has not proven optimal (e.g. time limit hit) is rejected.

    Raises BaselineSolverError: retryable for schedule-specific failures (no solution, no proof of
    optimality within the time limit), not retryable when retrying can't help (license/size limit).
    Raises KeyboardInterrupt if the solve was interrupted (e.g. Ctrl+C), so an interrupt stops the
    run instead of its partial result being used.
    """
    solver = solver or BASELINE_SOLVER_CONFIG["solver"]
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; expected one of {sorted(SOLVERS)}")
    module_name, function_name, _ = SOLVERS[solver]
    solve = getattr(importlib.import_module(module_name), function_name)
    return solve(
        build_assignment_data(flights, plane_configs, dist_dict, penalty_per_min),
        use_clipping,
        BASELINE_SOLVER_CONFIG["time_limit_s"] if time_limit_s is None else time_limit_s,
        BASELINE_SOLVER_CONFIG["require_optimal"] if require_optimal is None else require_optimal,
    )


def rewards_agree(predicted: float, actual: float) -> bool:
    """Equal up to solver precision: 0.5 reward units, plus 1e-6 relative for very large totals."""
    return abs(predicted - actual) <= 0.5 + 1e-6 * abs(actual)


def replay_plan(env: AirlineEnv, plane_names: list[str]) -> float:
    """Total env reward of flying `plane_names[i]` on env.flights[i], from a reset env copy."""
    env = copy.deepcopy(env)
    env.reset()
    total = 0.0
    for plane in plane_names:
        _, reward, _, _ = env.step(env.planes.index(plane))
        total += reward
    return total


_PLAN_CACHE: dict = {}


def _schedule_key(env: AirlineEnv, solver: str) -> tuple:
    flights = tuple(
        (f["origin"], f["dest"], f["start"], f["pass"], f.get("total_ticket_price", 0.0)) for f in env.flights
    )
    planes = tuple((p, tuple(sorted(env.plane_configs[p].items()))) for p in env.planes)
    return solver, flights, planes, tuple(sorted(env.dist_dict.items())), env.penalty_per_min, env.use_clipping


def plan_schedule(env: AirlineEnv, solver: str | None = None) -> SolverPlan:
    """Optimal plan for env's schedule and reward settings, solved once per schedule and checked
    against the env itself. Raises BaselineSolverError (see solve_assignment); a plan whose env
    reward disagrees with the solver's prediction is reported as a retryable failure."""
    solver = solver or BASELINE_SOLVER_CONFIG["solver"]
    key = _schedule_key(env, solver)
    if key not in _PLAN_CACHE:
        plan = solve_assignment(
            env.flights, env.plane_configs, env.dist_dict, env.penalty_per_min, env.use_clipping, solver=solver
        )
        actual = replay_plan(env, plan.planes)
        if not rewards_agree(plan.predicted_reward, actual):
            raise BaselineSolverError(
                f"{solver_display_name(solver)} plan disagrees with AirlineEnv: predicted reward "
                f"{plan.predicted_reward:,.2f}, env gives {actual:,.2f}. If this happens on every schedule, "
                "AirlineEnv.step() has changed and src/baselines/optimal.py must be updated to match it."
            )
        if len(_PLAN_CACHE) > 256:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = plan
    return _PLAN_CACHE[key]


class OptimalBaselineSolver(BaseSolver):
    """Optimal baseline: plans the whole schedule when an episode starts (see plan_schedule), then
    returns that plan's plane for each flight in turn."""

    def __init__(self, solver: str | None = None):
        self.solver = solver
        self.plan: SolverPlan | None = None

    def choose_action(self, state, env: AirlineEnv) -> int:
        if env.current_f_idx == 0 or self.plan is None:
            self.plan = plan_schedule(env, self.solver)
        return env.planes.index(self.plan.planes[env.current_f_idx])
