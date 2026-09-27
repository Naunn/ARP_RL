"""Multi-iteration comparison: train the chosen algorithms/variants for N_ITERATIONS iterations
on a chosen schedule (real "sample", synthetic "random", or synthetic "trap"), evaluating every
method against the Random/Greedy baselines after each iteration, and finish with a box plot of
each method's reward across iterations.

Two modes, via NEW_SCHEDULE_EACH_ITERATION:
- False: one schedule; the same agents keep training on it every iteration, so the box plot shows
  each method's reward spread over the course of training on that schedule.
- True: a fresh schedule (and fresh agents) every iteration, so the box plot shows each method's
  reward spread across many different schedules.

Results land in runs/<timestamp>_iter_training/ (results.pkl as {"<iteration>": (training_scores,
eval_results)}, plus the box plot); `python -m src.analysis.instance_sweep_plots --run <dir>`
adds per-iteration profit/delay trend plots.
"""

import numpy as np

from src.analysis.plots import plot_method_boxplot
from src.config import N_ITERATIONS as DEFAULT_N_ITERATIONS
from src.config import REWARD_CONFIG
from src.experiments.experiment_setup import (
    build_variant_agents,
    evaluate_models_on_schedule,
    resolve_project_root,
    train_agents_on_schedule,
)
from src.experiments.run_tracking import start_run
from src.instances import (
    build_schedule,
    discover_roadef_instances,
    load_roadef_instance,
)
from src.utils import log_section, logger, set_seed
from src.utils.envs import AirlineEnv

# ============================================================================
# WHAT TO RUN
# ============================================================================

# int -> reproducible (same schedules, same results every rerun); None -> fresh seed, i.e.
# genuinely new schedules every run (the seed used is still saved in the run's config.json).
SEED = None  # DEFAULT_SEED

# ROADEF instance the fleet, distances, and (for synthetic schedules) airports/fares come from:
# A01_6088570..A10_6088590 (~600 flights/84 aircraft) or B_01..B_10 (~1400 flights/255 aircraft).
INSTANCE_NAME = "A01_6088570"

# "sample" (real flights) / "random" / "trap" -- see src/instances/scenarios.py
SCHEDULE_TYPE = "trap"
N_CITIES = 4  # "random"/"trap" only (trap needs >= 3)
MAX_FLIGHTS = 25  # None = full instance
MAX_PLANES = 5  # None = full instance

# Methods to compare, besides the Random/Greedy baselines that are always evaluated.
ACTIVE_ALGOS = ["DQN", "DOUBLE_DQN"]  # any of AGENT_CLASSES in experiment_setup.py
ACTIVE_VARIANTS = ["idle"]  # any of config.AGENT_VARIANT_OVERRIDES: "full" / "no_bias" / "idle"

N_ITERATIONS = DEFAULT_N_ITERATIONS
N_EPISODES = 500  # episodes per iteration; None = MODEL_TRAINING_PARAMS default per algo
NEW_SCHEDULE_EACH_ITERATION = True

# ============================================================================


def main() -> None:
    seed = set_seed(SEED)
    run = start_run(
        "iter_training",
        {
            "instance": INSTANCE_NAME,
            "schedule_type": SCHEDULE_TYPE,
            "n_cities": N_CITIES if SCHEDULE_TYPE != "sample" else None,
            "max_flights": MAX_FLIGHTS,
            "max_planes": MAX_PLANES,
            "active_algos": ACTIVE_ALGOS,
            "active_variants": ACTIVE_VARIANTS,
            "n_iterations": N_ITERATIONS,
            "n_episodes": N_EPISODES,
            "new_schedule_each_iteration": NEW_SCHEDULE_EACH_ITERATION,
        },
        seed,
    )

    instances = discover_roadef_instances(resolve_project_root())
    if INSTANCE_NAME not in instances:
        raise ValueError(f"Unknown instance {INSTANCE_NAME!r}. Available: {sorted(instances)}")
    base = load_roadef_instance(instances[INSTANCE_NAME])
    penalty = REWARD_CONFIG["penalty_per_min"]

    def new_schedule_and_agents(i: int):
        # A new schedule can change the state dimensions (e.g. a different number of airports),
        # so it always gets freshly initialized agents.
        schedule = build_schedule(base, SCHEDULE_TYPE, MAX_FLIGHTS, MAX_PLANES, N_CITIES, seed=seed + i)
        flights, planes, airports, dist_dict = schedule
        dummy_env = AirlineEnv(
            flights,
            planes,
            dist_dict,
            airports,
            penalty,
            use_clipping=REWARD_CONFIG["train_use_clipping"],
        )
        return schedule, build_variant_agents(dummy_env, ACTIVE_ALGOS, ACTIVE_VARIANTS)

    (flights, planes, airports, dist_dict), agents = new_schedule_and_agents(0)
    iter_viz = {}
    for i in range(N_ITERATIONS):
        log_section(f"ITERATION {i + 1}/{N_ITERATIONS}")
        fresh = i == 0 or NEW_SCHEDULE_EACH_ITERATION
        if i > 0 and NEW_SCHEDULE_EACH_ITERATION:
            (flights, planes, airports, dist_dict), agents = new_schedule_and_agents(i)
        logger.info(
            f"Schedule: {SCHEDULE_TYPE} from {INSTANCE_NAME}, {len(flights)} flights, {len(planes)} aircraft, "
            f"{len(airports)} airports ({'new schedule, new agents' if fresh else 'same schedule, agents keep training'})"
        )
        meta_dims = (len(flights), len(airports), len(planes))

        training_scores = train_agents_on_schedule(
            agents,
            flights,
            planes,
            dist_dict,
            airports,
            penalty,
            meta_dims,
            run.checkpoint_dir,
            1,
            f"iter{i + 1}",
            n_episodes=N_EPISODES,
        )
        eval_results = evaluate_models_on_schedule(
            agents,
            flights,
            planes,
            dist_dict,
            airports,
            penalty,
            f"iteration {i + 1}/{N_ITERATIONS}",
        )
        iter_viz[f"{i}"] = (training_scores, eval_results)

    methods = list(iter_viz["0"][1])
    run.save_results(
        iter_viz,
        metrics={
            "mean_profit": {m: float(np.mean([iter_viz[k][1][m][0] for k in iter_viz])) for m in methods},
            "mean_delay_min": {m: float(np.mean([iter_viz[k][1][m][1] for k in iter_viz])) for m in methods},
        },
    )
    plot_method_boxplot(
        iter_viz,
        f"{INSTANCE_NAME} {SCHEDULE_TYPE}: reward across {N_ITERATIONS} iterations",
        save_path=run.run_dir / "method_boxplot.png",
    )


if __name__ == "__main__":
    main()
