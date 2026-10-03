"""Disruption-recovery experiment: train agent variants on a schedule, then repeatedly disrupt it
and measure how much profit each variant recovers after retraining on the disrupted schedule.

Results land in runs/<timestamp>_disruption_recovery/results.pkl, laid out as
    {"<run>": (initial_schedule, initial_eval,
               {"<disruption_iter>": (disrupted_schedule, pre_retrain_eval, post_retrain_eval,
                                      initial_schedule_post_retrain_eval)},
               final_reeval_per_disruption)}
Tabulate recovery with `python -m src.analysis.disruption_recovery_table`.
"""

from typing import Any, Dict, List, cast

from src.config import N_ITERATIONS as DEFAULT_N_DISRUPTIONS
from src.config import (
    REWARD_CONFIG,
)
from src.config import (  # noqa: F401  (kept for switching SEED back)
    SEED as DEFAULT_SEED,
)
from src.experiments.experiment_setup import (
    RETRY_SEED_STRIDE,
    build_disruption_actions,
    build_schedule_with_retries,
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
from src.utils import DisruptionGenerator, log_section, log_subsection, logger, set_seed
from src.utils.envs import AirlineEnv

# ============================================================================
# WHAT TO RUN
# ============================================================================

# int -> reproducible (same schedules/disruptions/results every rerun); None -> fresh seed, i.e.
# genuinely new schedules every run (the seed used is still saved in the run's config.json).
SEED = None  # DEFAULT_SEED

# ROADEF instance the fleet, distances, and (for synthetic schedules) airports/fares come from:
# A01_6088570..A10_6088590 (~600 flights/84 aircraft) or B_01..B_10 (~1400 flights/255 aircraft).
INSTANCE_NAME = "A01_6088570"

# "sample" (real flights) / "random" / "trap" -- see src/instances/scenarios.py
SCHEDULE_TYPE = "random"
N_CITIES = 3  # "random"/"trap" only (trap needs >= 3)
MAX_FLIGHTS = 15  # None = full instance
MAX_PLANES = 2  # None = full instance

# Methods to compare, besides the Random/Greedy baselines that are always evaluated.
ACTIVE_ALGOS = ["DQN", "DOUBLE_DQN"]  # any of AGENT_CLASSES in experiment_setup.py
ACTIVE_VARIANTS = ["idle"]  # any of config.AGENT_VARIANT_OVERRIDES

N_RUNS = 5  # independent schedules, each put through the full train -> disrupt/retrain cycle
N_DISRUPTIONS = DEFAULT_N_DISRUPTIONS  # disrupt -> retrain rounds per run
N_EPISODES = None  # episodes for the initial training; None = MODEL_TRAINING_PARAMS default per algo
N_RETRAIN_EPISODES = None  # episodes for each retraining on a disruption; None = same as N_EPISODES

# Also evaluate the optimal CPLEX baseline (src/baselines/cplex_solver.py) as a reference. Needs
# `uv sync --extra cplex`; the free CPLEX edition only handles ~10-12 flights with 3 planes.
INCLUDE_CPLEX = False

# ============================================================================


def main() -> None:
    seed = set_seed(SEED)
    run = start_run(
        "disruption_recovery",
        {
            "instance": INSTANCE_NAME,
            "schedule_type": SCHEDULE_TYPE,
            "n_cities": N_CITIES if SCHEDULE_TYPE != "sample" else None,
            "max_flights": MAX_FLIGHTS,
            "max_planes": MAX_PLANES,
            "active_algos": ACTIVE_ALGOS,
            "active_variants": ACTIVE_VARIANTS,
            "n_runs": N_RUNS,
            "n_disruptions": N_DISRUPTIONS,
            "n_episodes": N_EPISODES,
            "n_retrain_episodes": N_RETRAIN_EPISODES,
            "include_cplex": INCLUDE_CPLEX,
        },
        seed,
    )

    instances = discover_roadef_instances(resolve_project_root())
    if INSTANCE_NAME not in instances:
        raise ValueError(f"Unknown instance {INSTANCE_NAME!r}. Available: {sorted(instances)}")
    base = load_roadef_instance(instances[INSTANCE_NAME])
    penalty: int = REWARD_CONFIG["penalty_per_min"]
    retrain_episodes = N_RETRAIN_EPISODES if N_RETRAIN_EPISODES is not None else N_EPISODES

    iter_viz: Dict[str, Any] = {}
    for i in range(N_RUNS):
        log_section(f"RUN {i + 1}/{N_RUNS}")
        # If the CPLEX baseline is included and fails on a schedule, a new one is drawn for the
        # same run (next attempt's seed) before any training happens on it.
        flights, planes, airports, dist_dict = build_schedule_with_retries(
            lambda attempt: build_schedule(
                base, SCHEDULE_TYPE, MAX_FLIGHTS, MAX_PLANES, N_CITIES, seed=seed + i + attempt * RETRY_SEED_STRIDE
            ),
            penalty,
            INCLUDE_CPLEX,
            label=f"run {i + 1}, initial schedule",
        )
        logger.info(
            f"Schedule: {SCHEDULE_TYPE} from {INSTANCE_NAME}, {len(flights)} flights, {len(planes)} aircraft, "
            f"{len(airports)} airports"
        )

        dummy_env = AirlineEnv(
            flights,
            planes,
            dist_dict,
            airports,
            penalty,
            use_clipping=REWARD_CONFIG["train_use_clipping"],
        )
        agents = build_variant_agents(dummy_env, ACTIVE_ALGOS, ACTIVE_VARIANTS)
        meta_dims = (len(flights), len(airports), len(planes))
        dg = DisruptionGenerator(airports, dist_dict)

        def evaluate(schedule: List[Dict[str, Any]], label: str) -> Dict[str, tuple]:
            return evaluate_models_on_schedule(
                agents, schedule, planes, dist_dict, airports, penalty, label, include_cplex=INCLUDE_CPLEX
            )

        def train(schedule: List[Dict[str, Any]], phase_name: str, n_episodes: int | None) -> None:
            train_agents_on_schedule(
                agents,
                schedule,
                planes,
                dist_dict,
                airports,
                penalty,
                meta_dims,
                run.checkpoint_dir,
                1,
                phase_name,
                n_episodes=n_episodes,
            )

        log_section(f"RUN {i + 1}/{N_RUNS} | PHASE 1/3: train on the initial schedule")
        train(flights, f"run{i + 1}_initial", N_EPISODES)
        disruptions: Dict[str, tuple] = {}
        iter_viz[f"{i}"] = (
            flights,
            evaluate(flights, "initial schedule, after initial training"),
            disruptions,
        )

        plural = "s" if N_DISRUPTIONS != 1 else ""
        log_section(
            f"RUN {i + 1}/{N_RUNS} | PHASE 2/3: disrupt -> evaluate -> retrain ({N_DISRUPTIONS} disruption{plural})"
        )
        for d in range(1, N_DISRUPTIONS + 1):
            log_subsection(f"Disruption {d}/{N_DISRUPTIONS}: generated from the initial schedule")
            # A disruption the CPLEX baseline fails on is replaced by a newly generated one.
            disrupted, _, _, _ = build_schedule_with_retries(
                lambda _attempt: (
                    cast(List[Dict[str, Any]], dg.generate(flights, actions=build_disruption_actions(len(flights)))),
                    planes,
                    airports,
                    dist_dict,
                ),
                penalty,
                INCLUDE_CPLEX,
                label=f"run {i + 1}, disruption {d}",
            )
            pre_retrain_eval = evaluate(disrupted, f"disruption {d}, before retraining")
            train(disrupted, f"run{i + 1}_disruption{d}", retrain_episodes)
            disruptions[f"{d}"] = (
                disrupted,
                pre_retrain_eval,
                evaluate(disrupted, f"disruption {d}, after retraining"),
                evaluate(flights, f"initial schedule, after retraining on disruption {d}"),
            )

        log_section(f"RUN {i + 1}/{N_RUNS} | PHASE 3/3: re-evaluate final agents on every disruption")
        final_reeval = {
            key: evaluate(disruptions[key][0], f"disruption {key}, after all retraining")
            for key in sorted(disruptions, key=int)
        }
        iter_viz[f"{i}"] = (*iter_viz[f"{i}"], final_reeval)

    run.save_results(iter_viz)


if __name__ == "__main__":
    main()
