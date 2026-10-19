"""CPLEX (docplex) backend of the optimal baseline -- the formulation is documented in optimal.py.

Needs `uv sync --extra cplex`. The free CPLEX edition allows 1000 variables/constraints (~20
flights x 3 planes).
"""

from cplex.exceptions import CplexError
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException, DOcplexLimitsExceeded

from src.baselines import BaselineSolverError
from src.baselines.optimal import STEP_COST_CAP, AssignmentData, SolverPlan
from src.utils.envs import DELAY_PENALTY_CAP

# CPLEX MIP status codes (cplex._internal._constants).
_PROVEN_OPTIMAL = {101, 102}  # CPXMIP_OPTIMAL, CPXMIP_OPTIMAL_TOL (within the 1e-9 mip gap)
_ABORTED_BY_USER = {113, 114}  # CPXMIP_ABORT_FEAS / _INFEAS: e.g. Ctrl+C during the solve


def solve_with_cplex(
    data: AssignmentData, use_clipping: bool, time_limit_s: float, require_optimal: bool
) -> SolverPlan:
    try:
        return _solve(data, use_clipping, time_limit_s, require_optimal)
    except DOcplexLimitsExceeded as e:
        raise BaselineSolverError(
            f"schedule too large for this CPLEX edition ({e}); use fewer flights/planes or a full CPLEX license",
            retryable=False,
        ) from e
    except (DOcplexException, CplexError) as e:
        raise BaselineSolverError(f"CPLEX error: {e}") from e


def _solve(data: AssignmentData, use_clipping: bool, time_limit_s: float, require_optimal: bool) -> SolverPlan:
    n, planes = data.n, data.planes
    m = Model(name="aircraft_assignment_env_twin")
    m.parameters.mip.tolerances.mipgap = 1e-9
    m.parameters.timelimit = time_limit_s

    x = m.binary_var_dict([(p, i) for p in planes for i in range(n)], name="x")
    z = m.binary_var_dict([(p, i) for p in planes for i in range(n)], name="z")
    y = m.binary_var_dict(data.arcs, name="y")
    s = m.continuous_var_list(n, lb=data.starts, name="s")
    # Delay split: d_quad (first 120 min, quadratic penalty) + d_lin (beyond 120, linear) are the
    # penalized delay; d_free is delay that costs nothing because the flight is capped/floored.
    d_quad = m.continuous_var_list(n, lb=0, ub=120, name="d_quad")
    d_lin = m.continuous_var_list(n, lb=0, name="d_lin")
    d_free = m.continuous_var_list(n, lb=0, ub=None if use_clipping else 0, name="d_free")
    past_120 = m.binary_var_list(n, name="past_120")

    for i in range(n):
        m.add_constraint(m.sum(x[p, i] for p in planes) == 1, f"cover_{i}")
        m.add_constraint(s[i] - data.starts[i] == d_quad[i] + d_lin[i] + d_free[i], f"delay_split_{i}")
        m.add_indicator(past_120[i], d_lin[i] <= 0, active_value=0, name=f"lin_only_past_120_{i}")
        m.add_indicator(past_120[i], d_quad[i] >= 120, active_value=1, name=f"quad_full_before_lin_{i}")

    for p in planes:
        m.add_constraint(m.sum(z[p, i] for i in range(n)) <= 1, f"one_sequence_{p}")
        for i in range(n):
            m.add_constraint(z[p, i] + m.sum(y[p, g, i] for g in range(i)) == x[p, i], f"flow_in_{p}_{i}")
            m.add_constraint(m.sum(y[p, i, h] for h in range(i + 1, n)) <= x[p, i], f"flow_out_{p}_{i}")
            m.add_indicator(z[p, i], s[i] >= data.home_min[p, i], name=f"ready_home_{p}_{i}")
    for p, g, i in data.arcs:
        ready = s[g] + data.flight_min[p, g] + data.arc_min[p, g, i]
        m.add_indicator(y[p, g, i], s[i] >= ready, name=f"ready_{p}_{g}_{i}")

    objective = []
    for i in range(n):
        lin_cost = -data.revenue[i] + m.sum(
            data.flight_cost[p, i] * x[p, i] + data.home_cost[p, i] * z[p, i] for p in planes
        )
        lin_cost += m.sum(data.arc_cost[p, g, i] * y[p, g, i] for p in planes for g in range(i))
        k = data.delay_k[i]
        delay_pen = k * d_quad[i] + (k / 60) * d_quad[i] * d_quad[i] + 3 * k * d_lin[i]

        if use_clipping:
            capped = m.binary_var(name=f"capped_{i}")
            floored = m.binary_var(name=f"floored_{i}")
            unpenalized = m.binary_var(name=f"unpenalized_{i}")  # capped or floored
            m.add_constraint(unpenalized == capped + floored, f"one_mode_{i}")
            m.add_indicator(unpenalized, d_free[i] <= 0, active_value=0, name=f"normal_delay_penalized_{i}")
            m.add_indicator(unpenalized, d_quad[i] + d_lin[i] <= 0, active_value=1, name=f"delay_free_{i}")
            # All costs are >= 0, so a flight costs at least -revenue; this finite bound keeps the
            # relaxation bounded (with lb=-inf only the indicators limit it, which the LP ignores).
            step = m.continuous_var(lb=-data.revenue[i], name=f"step_cost_{i}")
            m.add_indicator(floored, step >= lin_cost, active_value=0, name=f"step_cost_{i}")
            m.add_indicator(floored, step >= STEP_COST_CAP, active_value=1, name=f"step_cost_floor_{i}")
            objective.append(step + delay_pen + DELAY_PENALTY_CAP * capped)
        else:
            objective.append(lin_cost + delay_pen)

    m.minimize(m.sum(objective))
    solution = m.solve()
    details = m.solve_details
    if details.status_code in _ABORTED_BY_USER:
        raise KeyboardInterrupt(f"CPLEX solve interrupted ({details.status})")
    if not solution:
        raise BaselineSolverError(f"CPLEX returned no solution ({details.status}, {details.time:.0f}s)")
    proven = details.status_code in _PROVEN_OPTIMAL
    if require_optimal and not proven:
        raise BaselineSolverError(
            f"CPLEX did not prove optimality ({details.status} after {details.time:.0f}s; best plan found "
            f"{-solution.objective_value:,.0f}, gap to bound {details.mip_relative_gap:.0%})"
        )

    assigned = [next(p for p in planes if solution.get_value(x[p, i]) > 0.5) for i in range(n)]
    return SolverPlan(
        planes=assigned,
        starts_from_home=[solution.get_value(z[assigned[i], i]) > 0.5 for i in range(n)],
        objective=float(solution.objective_value),
        proven_optimal=proven,
        gap=0.0 if proven else float(details.mip_relative_gap),
        solve_seconds=float(details.time),
        solver="cplex",
    )


def solve_assignment(flights, plane_configs, dist_dict, penalty_per_min, use_clipping, **kwargs) -> SolverPlan:
    """Optimal plan solved with CPLEX (see optimal.solve_assignment for the keyword options)."""
    from src.baselines.optimal import solve_assignment as solve_with

    return solve_with(flights, plane_configs, dist_dict, penalty_per_min, use_clipping, solver="cplex", **kwargs)
