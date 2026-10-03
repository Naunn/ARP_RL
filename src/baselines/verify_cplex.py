"""Checks that the CPLEX baseline is (1) aligned with AirlineEnv and (2) truly optimal.

    python -m src.baselines.verify_cplex                      # 20 cases, 6 flights, 3 planes
    python -m src.baselines.verify_cplex --cases 50 --flights 7 --planes 2

For each case it builds a small schedule (cycling through sample / random / trap, clipping on and
off, and every 4th case with shrunken planes so under-capacity assignments happen), then:

1. Alignment: replays CPLEX's plan through the real AirlineEnv and compares the env's total reward
   with the reward CPLEX predicted (-objective). Equal means the model computes exactly the env's
   reward -- if any env rule were missing or different, they would disagree.
2. Optimality: tries EVERY possible plane-per-flight assignment (planes ** flights of them) in the
   real AirlineEnv and takes the best total reward. CPLEX's plan scoring the same as the best of
   all assignments means no assignment beats it.

A case where CPLEX returns no solution is reported as "solver failed" (experiments retry those with
a new schedule). Check 2 uses only the environment, never the CPLEX model, so it is an independent referee. It only
scales to small cases (3 planes x 7 flights = 2187 assignments); for real-size schedules check 1
still runs automatically every time CplexSolver plans, and optimality there rests on CPLEX's own
proof of optimality (mip gap 1e-9) of an objective that check 1 shows equals the env reward.
"""

import argparse
import itertools
import random

from src.baselines import BaselineSolverError
from src.baselines.cplex_solver import replay_plan, rewards_agree, solve_assignment
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
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--flights", type=int, default=6)
    parser.add_argument("--planes", type=int, default=3)
    parser.add_argument("--instance", default="A01_6088570")
    parser.add_argument("--penalty", type=float, default=150.0, help="penalty_per_min")
    args = parser.parse_args()

    n_assignments = args.planes**args.flights
    if n_assignments > 2_000_000:
        parser.error(f"{args.planes}**{args.flights} = {n_assignments:,} assignments is too many to enumerate")

    base = load_roadef_instance(discover_roadef_instances(resolve_project_root())[args.instance])
    log_section(f"CPLEX baseline verification: {args.cases} cases, {args.flights} flights, {args.planes} planes")
    logger.info(
        f"  {'case':>4} {'schedule':<8} {'clip':<5} {'CPLEX predicts':>15} {'env(CPLEX plan)':>16} "
        f"{'best of all':>14}  aligned  optimal"
    )

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
        try:
            plan = solve_assignment(flights, planes, dist_dict, args.penalty, use_clipping)
        except BaselineSolverError as e:
            solver_failures += 1
            logger.info(f"  {case:>4} {schedule_type:<8} {use_clipping!s:<5} solver failed: {e}")
            continue
        plan_reward = replay_plan(env, plan.planes)
        best = best_by_enumeration(env)

        aligned = rewards_agree(plan.predicted_reward, plan_reward)
        optimal = rewards_agree(best, plan_reward) or plan_reward > best
        failures += not (aligned and optimal)
        logger.info(
            f"  {case:>4} {schedule_type:<8} {use_clipping!s:<5} {plan.predicted_reward:>15,.2f} "
            f"{plan_reward:>16,.2f} {best:>14,.2f}  {'yes' if aligned else 'NO':<7}  {'yes' if optimal else 'NO'}"
        )

    solved = args.cases - solver_failures
    logger.info(f"Solver failed (no solution) on {solver_failures}/{args.cases} cases.")
    if failures:
        raise SystemExit(f"{failures}/{solved} solved cases FAILED")
    logger.info(f"All {solved} solved cases passed: CPLEX matches the env reward and the best of all assignments.")


if __name__ == "__main__":
    main()
