"""Checks that the optimal baseline is (1) aligned with AirlineEnv and (2) truly optimal.

    python -m src.baselines.verify_baseline                         # configured solver, 20 cases
    python -m src.baselines.verify_baseline --solver both           # CPLEX and Gurobi, cross-checked
    python -m src.baselines.verify_baseline --cases 50 --flights 7 --planes 2
    python -m src.baselines.verify_baseline --solver both --flights 12 --planes 3 --skip-brute-force

For each case it builds a small schedule (cycling through sample / random / trap, clipping on and
off, and every 4th case with shrunken planes so under-capacity assignments happen), then for each
selected solver:

1. Alignment: replays the solver's plan through the real AirlineEnv and compares the env's total
   reward with the reward the solver predicted (-objective). Equal means the model computes
   exactly the env's reward -- if any env rule were missing or different, they would disagree.
2. Optimality: tries EVERY possible plane-per-flight assignment (planes ** flights of them) in the
   real AirlineEnv and takes the best total reward. The solver's plan scoring the same as the best
   of all assignments means no assignment beats it.
3. With --solver both: CPLEX and Gurobi, two independent solvers, must reach the same optimum.

Check 2 uses only the environment, never a solver, so it is an independent referee -- but it only
scales to small cases (3 planes x 7 flights = 2187 assignments). --skip-brute-force drops it for
larger cases; checks 1 and 3 still apply there. A case where a solver returns no (proven) solution
is reported as "solver failed" (experiments retry those with a new schedule).
"""

import argparse
import itertools
import random

from src.baselines import BaselineSolverError
from src.baselines.optimal import SOLVERS, replay_plan, rewards_agree, solve_assignment, solver_display_name
from src.config import BASELINE_SOLVER_CONFIG
from src.experiments.experiment_setup import resolve_project_root
from src.instances import SCHEDULE_TYPES, build_schedule, discover_roadef_instances, load_roadef_instance
from src.utils import log_section, logger
from src.utils.envs import AirlineEnv


def best_by_enumeration(env: AirlineEnv) -> float:
    """Highest total env reward over all planes ** flights assignments."""
    best = float("-inf")
    for actions in itertools.product(range(len(env.planes)), repeat=len(env.flights)):
        env.reset()
        total = 0.0
        for action in actions:
            _, reward, _, _ = env.step(action)
            total += reward
        best = max(best, total)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solver", choices=[*SOLVERS, "both"], default=BASELINE_SOLVER_CONFIG["solver"])
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--flights", type=int, default=6)
    parser.add_argument("--planes", type=int, default=3)
    parser.add_argument("--instance", default="A01_6088570")
    parser.add_argument("--penalty", type=float, default=150.0, help="penalty_per_min")
    parser.add_argument("--skip-brute-force", action="store_true", help="skip check 2 (for larger cases)")
    args = parser.parse_args()

    solvers = list(SOLVERS) if args.solver == "both" else [args.solver]
    n_assignments = args.planes**args.flights
    if not args.skip_brute_force and n_assignments > 2_000_000:
        parser.error(f"{args.planes}**{args.flights} = {n_assignments:,} assignments is too many to enumerate")

    base = load_roadef_instance(discover_roadef_instances(resolve_project_root())[args.instance])
    names = " + ".join(solver_display_name(s) for s in solvers)
    log_section(f"{names} baseline verification: {args.cases} cases, {args.flights} flights, {args.planes} planes")
    header = f"  {'case':>4} {'schedule':<8} {'clip':<5} {'best of all':>14}"
    for solver in solvers:
        header += f" | {solver_display_name(solver) + ' predicts':>16} {'env(plan)':>14} {'s':>5}"
    logger.info(header + "  result")

    failures = solver_failures = 0
    for case in range(args.cases):
        random.seed(case)
        schedule_type = SCHEDULE_TYPES[case % len(SCHEDULE_TYPES)]
        use_clipping = case % 2 == 0
        flights, planes, airports, dist_dict = build_schedule(
            base, schedule_type, args.flights, args.planes, n_cities=3, seed=case
        )
        if case % 4 == 1:
            planes = {name: {**cfg, "seats": cfg["seats"] // 3} for name, cfg in planes.items()}
        env = AirlineEnv(flights, planes, dist_dict, airports, args.penalty, use_clipping=use_clipping)

        best = None if args.skip_brute_force else best_by_enumeration(env)
        row = f"  {case:>4} {schedule_type:<8} {use_clipping!s:<5} {'-' if best is None else f'{best:,.2f}':>14}"
        problems, plan_rewards = [], []
        for solver in solvers:
            try:
                plan = solve_assignment(flights, planes, dist_dict, args.penalty, use_clipping, solver=solver)
            except BaselineSolverError as e:
                solver_failures += 1
                problems.append(f"{solver} failed: {e}")
                row += f" | {'failed':>16} {'':>14} {'':>5}"
                continue
            plan_reward = replay_plan(env, plan.planes)
            plan_rewards.append(plan_reward)
            if not rewards_agree(plan.predicted_reward, plan_reward):
                problems.append(f"{solver} not aligned")
            if best is not None and not (rewards_agree(best, plan_reward) or plan_reward > best):
                problems.append(f"{solver} not optimal")
            row += f" | {plan.predicted_reward:>16,.2f} {plan_reward:>14,.2f} {plan.solve_seconds:>5.1f}"
        if len(plan_rewards) == 2 and not rewards_agree(*plan_rewards):
            problems.append("solvers disagree")

        failures += any("failed" not in p for p in problems)
        logger.info(row + "  " + ("ok" if not problems else "; ".join(problems)))

    logger.info(f"Solver failed (no proven solution) {solver_failures} time(s) over {args.cases} cases.")
    if failures:
        raise SystemExit(f"{failures}/{args.cases} cases FAILED")
    checks = "match the env reward" + ("" if args.skip_brute_force else " and the best of all assignments")
    if len(solvers) == 2:
        checks += ", and both solvers agree"
    logger.info(f"All solved cases passed: plans {checks}.")


if __name__ == "__main__":
    main()
