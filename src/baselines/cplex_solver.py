"""Optimal baseline: an exact CPLEX model of AirlineEnv's reward.

`solve_assignment` finds the plane assignment that maximizes the environment's total reward for a
schedule, with the reward rules mirrored from AirlineEnv.step() term by term:

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
max(...) start times exactly for the chosen assignment. Conditional constraints are CPLEX
indicator constraints (enforced exactly), not Big-M ones (which leak through integrality
tolerance at these cost magnitudes).

`CplexSolver` wraps this as a baseline with the same choose_action interface as RandomSolver,
ClosestPlaneGreedySolver and DQNSolver. Plans go through `plan_schedule`, which caches them per
schedule and replays each new plan through the real environment, raising BaselineSolverError if
the reward differs from the prediction -- so a mis-solve, or an AirlineEnv reward change that this
model doesn't mirror, can never silently produce a wrong baseline number.

Size: the free CPLEX edition allows 1000 variables/constraints, i.e. roughly 10-12 flights with
3 planes; larger schedules need a full CPLEX license.
"""

import copy
from dataclasses import dataclass

from cplex.exceptions import CplexError
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException, DOcplexLimitsExceeded

from src.baselines import BaselineSolverError
from src.utils.envs import (
    DEFAULT_DIST_KM,
    DELAY_PENALTY_CAP,
    RELOCATION_PENALTY_PER_KM,
    REWARD_FLOOR,
    AirlineEnv,
    BaseSolver,
    capacity_slack_penalty,
)

STEP_COST_CAP = -REWARD_FLOOR  # the env's per-flight reward floor, as a cap on per-flight cost


@dataclass
class CplexPlan:
    planes: list[str]  # planes[i] is the plane assigned to flights[i]
    starts_from_home: list[bool]  # True where flights[i] is its plane's first flight
    objective: float  # minimized total cost, i.e. -(predicted total env reward)

    @property
    def predicted_reward(self) -> float:
        return -self.objective


def rewards_agree(predicted: float, actual: float) -> bool:
    """Equal up to solver precision: 0.5 reward units, plus 1e-6 relative for very large totals."""
    return abs(predicted - actual) <= 0.5 + 1e-6 * abs(actual)


def solve_assignment(
    flights: list[dict],
    plane_configs: dict,
    dist_dict: dict,
    penalty_per_min: float,
    use_clipping: bool,
) -> CplexPlan:
    """Optimal plane-per-flight assignment for AirlineEnv's reward (see module docstring).

    Raises BaselineSolverError: retryable for schedule-specific failures, not retryable when the
    model exceeds the CPLEX edition's size limits.
    """
    try:
        return _solve(flights, plane_configs, dist_dict, penalty_per_min, use_clipping)
    except DOcplexLimitsExceeded as e:
        raise BaselineSolverError(
            f"schedule too large for this CPLEX edition ({e}); use fewer flights/planes or a full CPLEX license",
            retryable=False,
        ) from e
    except (DOcplexException, CplexError) as e:
        raise BaselineSolverError(f"CPLEX error: {e}") from e


def _solve(flights, plane_configs, dist_dict, penalty_per_min, use_clipping) -> CplexPlan:
    n = len(flights)
    planes = list(plane_configs)

    def dist(a: str, b: str) -> float:
        return float(dist_dict.get((a, b), DEFAULT_DIST_KM))

    def travel_min(p: str, km: float) -> float:
        return km / (plane_configs[p]["speed"] / 60)

    def operating_cost(p: str, km: float) -> float:
        return plane_configs[p]["hourly_cost"] * travel_min(p, km) / 60

    flight_km = [dist(f["origin"], f["dest"]) for f in flights]
    home_km = {
        (p, i): dist(plane_configs[p]["initial_airport"], flights[i]["origin"]) for p in planes for i in range(n)
    }
    arcs = [(p, g, i) for p in planes for i in range(n) for g in range(i)]  # g listed before i
    arc_km = {(p, g, i): dist(flights[g]["dest"], flights[i]["origin"]) for p, g, i in arcs}
    delay_k = [f["pass"] * penalty_per_min for f in flights]

    m = Model(name="aircraft_assignment_env_twin")
    m.parameters.mip.tolerances.mipgap = 1e-9

    x = m.binary_var_dict([(p, i) for p in planes for i in range(n)], name="x")
    z = m.binary_var_dict([(p, i) for p in planes for i in range(n)], name="z")
    y = m.binary_var_dict(arcs, name="y")
    s = m.continuous_var_list(n, lb=[f["start"] for f in flights], name="s")
    # Delay split: d_quad (first 120 min, quadratic penalty) + d_lin (beyond 120, linear) are the
    # penalized delay; d_free is delay that costs nothing because the flight is capped/floored.
    d_quad = m.continuous_var_list(n, lb=0, ub=120, name="d_quad")
    d_lin = m.continuous_var_list(n, lb=0, name="d_lin")
    d_free = m.continuous_var_list(n, lb=0, ub=None if use_clipping else 0, name="d_free")
    past_120 = m.binary_var_list(n, name="past_120")

    for i, f in enumerate(flights):
        m.add_constraint(m.sum(x[p, i] for p in planes) == 1, f"cover_{i}")
        m.add_constraint(s[i] - f["start"] == d_quad[i] + d_lin[i] + d_free[i], f"delay_split_{i}")
        m.add_indicator(past_120[i], d_lin[i] <= 0, active_value=0, name=f"lin_only_past_120_{i}")
        m.add_indicator(past_120[i], d_quad[i] >= 120, active_value=1, name=f"quad_full_before_lin_{i}")

    for p in planes:
        m.add_constraint(m.sum(z[p, i] for i in range(n)) <= 1, f"one_sequence_{p}")
        for i in range(n):
            m.add_constraint(z[p, i] + m.sum(y[p, g, i] for g in range(i)) == x[p, i], f"flow_in_{p}_{i}")
            m.add_constraint(m.sum(y[p, i, h] for h in range(i + 1, n)) <= x[p, i], f"flow_out_{p}_{i}")
            m.add_indicator(z[p, i], s[i] >= travel_min(p, home_km[p, i]), name=f"ready_home_{p}_{i}")
        for g in range(n):
            for i in range(g + 1, n):
                ready = s[g] + travel_min(p, flight_km[g]) + travel_min(p, arc_km[p, g, i])
                m.add_indicator(y[p, g, i], s[i] >= ready, name=f"ready_{p}_{g}_{i}")

    objective = []
    for i, f in enumerate(flights):
        # Everything except the delay penalty: linear in the assignment binaries.
        lin_cost = -float(f.get("total_ticket_price", 0.0))
        for p in planes:
            lin_cost += (operating_cost(p, flight_km[i]) + capacity_slack_penalty(f, plane_configs[p]["seats"])) * x[
                p, i
            ]
            home = home_km[p, i]
            lin_cost += (
                plane_configs[p]["fixed_cost"] + operating_cost(p, home) + RELOCATION_PENALTY_PER_KM * home
            ) * z[p, i]
            for g in range(i):
                km = arc_km[p, g, i]
                lin_cost += (operating_cost(p, km) + RELOCATION_PENALTY_PER_KM * km) * y[p, g, i]

        delay_pen = delay_k[i] * d_quad[i] + (delay_k[i] / 60) * d_quad[i] * d_quad[i] + 3 * delay_k[i] * d_lin[i]

        if use_clipping:
            capped = m.binary_var(name=f"capped_{i}")
            floored = m.binary_var(name=f"floored_{i}")
            unpenalized = m.binary_var(name=f"unpenalized_{i}")  # capped or floored
            m.add_constraint(unpenalized == capped + floored, f"one_mode_{i}")
            m.add_indicator(unpenalized, d_free[i] <= 0, active_value=0, name=f"normal_delay_penalized_{i}")
            m.add_indicator(unpenalized, d_quad[i] + d_lin[i] <= 0, active_value=1, name=f"delay_free_{i}")
            step = m.continuous_var(lb=-m.infinity, name=f"step_cost_{i}")
            m.add_indicator(floored, step >= lin_cost, active_value=0, name=f"step_cost_{i}")
            m.add_indicator(floored, step >= STEP_COST_CAP, active_value=1, name=f"step_cost_floor_{i}")
            objective.append(step + delay_pen + DELAY_PENALTY_CAP * capped)
        else:
            objective.append(lin_cost + delay_pen)

    m.minimize(m.sum(objective))
    solution = m.solve()
    if not solution:
        raise BaselineSolverError(f"CPLEX returned no solution (status: {m.solve_details.status})")

    assigned = [next(p for p in planes if solution.get_value(x[p, i]) > 0.5) for i in range(n)]
    return CplexPlan(
        planes=assigned,
        starts_from_home=[solution.get_value(z[assigned[i], i]) > 0.5 for i in range(n)],
        objective=float(solution.objective_value),
    )


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


def _schedule_key(env: AirlineEnv) -> tuple:
    flights = tuple(
        (f["origin"], f["dest"], f["start"], f["pass"], f.get("total_ticket_price", 0.0)) for f in env.flights
    )
    planes = tuple((p, tuple(sorted(env.plane_configs[p].items()))) for p in env.planes)
    return flights, planes, tuple(sorted(env.dist_dict.items())), env.penalty_per_min, env.use_clipping


def plan_schedule(env: AirlineEnv) -> CplexPlan:
    """Optimal plan for env's schedule and reward settings, solved once per schedule and checked
    against the env itself. Raises BaselineSolverError (see solve_assignment); a plan whose env
    reward disagrees with CPLEX's prediction is reported as a retryable failure."""
    key = _schedule_key(env)
    if key not in _PLAN_CACHE:
        plan = solve_assignment(env.flights, env.plane_configs, env.dist_dict, env.penalty_per_min, env.use_clipping)
        actual = replay_plan(env, plan.planes)
        if not rewards_agree(plan.predicted_reward, actual):
            raise BaselineSolverError(
                f"CPLEX plan disagrees with AirlineEnv: predicted reward {plan.predicted_reward:,.2f}, env gives "
                f"{actual:,.2f}. If this happens on every schedule, AirlineEnv.step() has changed and "
                "src/baselines/cplex_solver.py must be updated to match it."
            )
        if len(_PLAN_CACHE) > 256:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[key] = plan
    return _PLAN_CACHE[key]


class CplexSolver(BaseSolver):
    """Optimal baseline: plans the whole schedule when an episode starts (see plan_schedule), then
    returns that plan's plane for each flight in turn."""

    def __init__(self):
        self.plan: CplexPlan | None = None

    def choose_action(self, state, env: AirlineEnv) -> int:
        if env.current_f_idx == 0 or self.plan is None:
            self.plan = plan_schedule(env)
        return env.planes.index(self.plan.planes[env.current_f_idx])
