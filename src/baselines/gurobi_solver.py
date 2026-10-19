"""Gurobi (gurobipy) backend of the optimal baseline -- the formulation is documented in optimal.py.

Needs `uv sync --extra gurobi` and a Gurobi license. The license bundled with the pip package only
allows 200 variables once the model has quadratic terms (~7 flights x 3 planes); point Gurobi at a
full/academic license (~/gurobi.lic, or GRB_LICENSE_FILE) for anything larger.
"""

import gurobipy as gp
from gurobipy import GRB

from src.baselines import BaselineSolverError
from src.baselines.optimal import STEP_COST_CAP, AssignmentData, SolverPlan
from src.utils.envs import DELAY_PENALTY_CAP

_LICENSE_ERRORS = {GRB.Error.SIZE_LIMIT_EXCEEDED, GRB.Error.NO_LICENSE}
_env: gp.Env | None = None


def _quiet_env() -> gp.Env:
    """One shared Gurobi environment. Starting it prints Gurobi's license line once (e.g. "Academic
    license - for non-commercial use only - expires ..."); solver output is switched off afterwards."""
    global _env
    if _env is None:
        _env = gp.Env()
        _env.setParam("OutputFlag", 0)
    return _env


def solve_with_gurobi(
    data: AssignmentData, use_clipping: bool, time_limit_s: float, require_optimal: bool
) -> SolverPlan:
    try:
        return _solve(data, use_clipping, time_limit_s, require_optimal)
    except gp.GurobiError as e:
        if e.errno in _LICENSE_ERRORS:
            raise BaselineSolverError(
                f"Gurobi license does not allow this model ({e}); install a full/academic license "
                "(~/gurobi.lic or GRB_LICENSE_FILE) or use fewer flights/planes",
                retryable=False,
            ) from e
        raise BaselineSolverError(f"Gurobi error: {e}") from e


def _solve(data: AssignmentData, use_clipping: bool, time_limit_s: float, require_optimal: bool) -> SolverPlan:
    n, planes = data.n, data.planes
    with gp.Model("aircraft_assignment_env_twin", env=_quiet_env()) as m:
        m.Params.MIPGap = 1e-9
        m.Params.TimeLimit = time_limit_s

        x = m.addVars([(p, i) for p in planes for i in range(n)], vtype=GRB.BINARY, name="x")
        z = m.addVars([(p, i) for p in planes for i in range(n)], vtype=GRB.BINARY, name="z")
        y = m.addVars(data.arcs, vtype=GRB.BINARY, name="y")
        s = m.addVars(n, lb=data.starts, name="s")
        # Delay split: d_quad (first 120 min, quadratic penalty) + d_lin (beyond 120, linear) are the
        # penalized delay; d_free is delay that costs nothing because the flight is capped/floored.
        d_quad = m.addVars(n, lb=0, ub=120, name="d_quad")
        d_lin = m.addVars(n, lb=0, name="d_lin")
        d_free = m.addVars(n, lb=0, ub=GRB.INFINITY if use_clipping else 0, name="d_free")
        past_120 = m.addVars(n, vtype=GRB.BINARY, name="past_120")

        for i in range(n):
            m.addConstr(x.sum("*", i) == 1, f"cover_{i}")
            m.addConstr(s[i] - data.starts[i] == d_quad[i] + d_lin[i] + d_free[i], f"delay_split_{i}")
            m.addConstr((past_120[i] == 0) >> (d_lin[i] <= 0), f"lin_only_past_120_{i}")
            m.addConstr((past_120[i] == 1) >> (d_quad[i] >= 120), f"quad_full_before_lin_{i}")

        for p in planes:
            m.addConstr(z.sum(p, "*") <= 1, f"one_sequence_{p}")
            for i in range(n):
                m.addConstr(z[p, i] + gp.quicksum(y[p, g, i] for g in range(i)) == x[p, i], f"flow_in_{p}_{i}")
                m.addConstr(gp.quicksum(y[p, i, h] for h in range(i + 1, n)) <= x[p, i], f"flow_out_{p}_{i}")
                m.addConstr((z[p, i] == 1) >> (s[i] >= data.home_min[p, i]), f"ready_home_{p}_{i}")
        for p, g, i in data.arcs:
            ready = s[g] + data.flight_min[p, g] + data.arc_min[p, g, i]
            m.addConstr((y[p, g, i] == 1) >> (s[i] >= ready), f"ready_{p}_{g}_{i}")

        objective = gp.QuadExpr()
        for i in range(n):
            lin_cost = -data.revenue[i] + gp.quicksum(
                data.flight_cost[p, i] * x[p, i] + data.home_cost[p, i] * z[p, i] for p in planes
            )
            lin_cost += gp.quicksum(data.arc_cost[p, g, i] * y[p, g, i] for p in planes for g in range(i))
            k = data.delay_k[i]
            delay_pen = k * d_quad[i] + (k / 60) * d_quad[i] * d_quad[i] + 3 * k * d_lin[i]

            if use_clipping:
                capped = m.addVar(vtype=GRB.BINARY, name=f"capped_{i}")
                floored = m.addVar(vtype=GRB.BINARY, name=f"floored_{i}")
                unpenalized = m.addVar(vtype=GRB.BINARY, name=f"unpenalized_{i}")  # capped or floored
                m.addConstr(unpenalized == capped + floored, f"one_mode_{i}")
                m.addConstr((unpenalized == 0) >> (d_free[i] <= 0), f"normal_delay_penalized_{i}")
                m.addConstr((unpenalized == 1) >> (d_quad[i] + d_lin[i] <= 0), f"delay_free_{i}")
                # All costs are >= 0, so a flight costs at least -revenue; without this finite bound the
                # relaxation is unbounded (only indicators limit step) and Gurobi cannot prune at all.
                step = m.addVar(lb=-data.revenue[i], name=f"step_cost_{i}")
                m.addConstr((floored == 0) >> (step >= lin_cost), f"step_cost_{i}")
                m.addConstr((floored == 1) >> (step >= STEP_COST_CAP), f"step_cost_floor_{i}")
                objective += step + delay_pen + DELAY_PENALTY_CAP * capped
            else:
                objective += lin_cost + delay_pen

        m.setObjective(objective, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.INTERRUPTED:
            raise KeyboardInterrupt("Gurobi solve interrupted")
        if m.SolCount == 0:
            raise BaselineSolverError(f"Gurobi returned no solution (status {m.Status}, {m.Runtime:.0f}s)")
        proven = m.Status == GRB.OPTIMAL
        if require_optimal and not proven:
            raise BaselineSolverError(
                f"Gurobi did not prove optimality (status {m.Status} after {m.Runtime:.0f}s; best plan found "
                f"{-m.ObjVal:,.0f}, gap to bound {m.MIPGap:.0%})"
            )

        assigned = [next(p for p in planes if x[p, i].X > 0.5) for i in range(n)]
        return SolverPlan(
            planes=assigned,
            starts_from_home=[z[assigned[i], i].X > 0.5 for i in range(n)],
            objective=float(m.ObjVal),
            proven_optimal=proven,
            gap=0.0 if proven else float(m.MIPGap),
            solve_seconds=float(m.Runtime),
            solver="gurobi",
        )


def solve_assignment(flights, plane_configs, dist_dict, penalty_per_min, use_clipping, **kwargs) -> SolverPlan:
    """Optimal plan solved with Gurobi (see optimal.solve_assignment for the keyword options)."""
    from src.baselines.optimal import solve_assignment as solve_with

    return solve_with(flights, plane_configs, dist_dict, penalty_per_min, use_clipping, solver="gurobi", **kwargs)
