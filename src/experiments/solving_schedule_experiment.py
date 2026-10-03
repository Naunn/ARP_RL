"""Instance-generalization experiment: sample a fresh random flight/fleet instance every
iteration, train each active agent variant on it once, and track how solution quality compares
to the baselines across many different problem instances (not disruptions of one instance).

Results land in runs/<timestamp>_instance_sweep/results.pkl as
{"<iteration>": (training_scores, eval_results)}; plot them with
`python -m src.analysis.instance_sweep_plots --run runs/<run_id>`.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config import N_ITERATIONS, REWARD_CONFIG
from src.config import SEED as DEFAULT_SEED
from src.experiments.experiment_setup import (
    build_schedule_with_retries,
    build_variant_agents,
    evaluate_models_on_schedule,
    resolve_project_root,
    train_agents_on_schedule,
)
from src.experiments.run_tracking import start_run
from src.instances import build_flight_pool, build_planes, generate_random_flights, generate_trap_schedule
from src.utils import create_dist_dict_from_airports, log_section, logger, set_seed
from src.utils.envs import AirlineEnv

# int -> reproducible (same schedules/results every rerun); None -> fresh seed, i.e. genuinely new
# schedules every run (the seed used is still saved in the run's config.json).
SEED = DEFAULT_SEED

# Which ablation variants/algorithms this run trains and compares. See src.config.AGENT_VARIANT_OVERRIDES.
ACTIVE_ALGOS = ["DOUBLE_DQN"]
ACTIVE_VARIANTS = ["full", "idle"]

N = 25  # 15 for testing random + trap (100 iter); round(flights_df.shape[0]/3) for sampled - too long for 100 iter :(
C_N = 4  # 3 for testing random + trap (100 iter); for sampled does not matter
P_N = 4  # 2 for testing random + trap (100 iter); round(aircraft_df.shape[0]/3) for sampled - too long for 100 iter :(
TRAP = False

# Also evaluate the optimal CPLEX baseline (src/baselines/cplex_solver.py) as a reference. Needs
# `uv sync --extra cplex`; the free CPLEX edition only handles ~10-12 flights with 3 planes.
INCLUDE_CPLEX = False


def main() -> None:
    seed = set_seed(SEED)
    run = start_run(
        "instance_sweep",
        {
            "active_algos": ACTIVE_ALGOS,
            "active_variants": ACTIVE_VARIANTS,
            "n_flights": N,
            "n_cities": C_N,
            "n_planes": P_N,
            "trap": TRAP,
            "include_cplex": INCLUDE_CPLEX,
            "n_iterations": N_ITERATIONS,
        },
        seed,
    )

    training_data_dir = resolve_project_root() / "data" / "training"
    flights_df = pd.read_csv(training_data_dir / "flights.csv")
    itineraries_df = pd.read_csv(training_data_dir / "itineraries.csv")
    aircraft_df = pd.read_csv(training_data_dir / "aircraft.csv").drop_duplicates(
        subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
    )
    penalty: int = REWARD_CONFIG["penalty_per_min"]

    iter_viz: Dict[str, Any] = {}
    for i in range(N_ITERATIONS):
        log_section(f"ITERATION {i + 1}/{N_ITERATIONS}")

        def build_instance(_attempt: int):
            flights = build_flight_pool(flights_df.sample(N).sort_values("start_min", ascending=True), itineraries_df)

            schedule_generator = generate_trap_schedule if TRAP else generate_random_flights
            samp_flights = pd.DataFrame(
                schedule_generator(
                    n=N,
                    cities=list(np.random.choice(flights_df.origin.unique(), C_N)),
                    start_time_range=(flights_df.start_min.min(), flights_df.arrival_min.max()),
                    pass_range=(
                        itineraries_df.total_passenger_count.min(),
                        itineraries_df.total_passenger_count.max(),
                    ),
                )
            )
            samp_flights["total_ticket_price"] = samp_flights["pass"] * np.average(
                pd.DataFrame(flights)["total_ticket_price"] / pd.DataFrame(flights)["pass"],
                weights=pd.DataFrame(flights)["pass"],
            )
            # NOTE: `flights` intentionally stays the real-data-sampled pool here rather than the
            # synthetic `samp_flights` schedule generated above (that generated schedule currently goes
            # unused) -- carried over as-is from the prior version of this script; worth confirming
            # this is the intended instance source before relying on results from this script.

            planes = build_planes(aircraft_df.sample(P_N))
            airports = sorted(
                {f["origin"] for f in flights}
                | {f["dest"] for f in flights}
                | {p["initial_airport"] for p in planes.values()}
            )
            dist_dict = create_dist_dict_from_airports(airports_list=airports)
            return flights, planes, airports, dist_dict

        # If the CPLEX baseline is included and fails on an instance, a new one is sampled for the
        # same iteration before any training happens on it.
        flights, planes, airports, dist_dict = build_schedule_with_retries(
            build_instance, penalty, INCLUDE_CPLEX, label=f"iteration {i + 1}"
        )

        dummy_env = AirlineEnv(
            flights=flights,
            plane_configs=planes,
            dist_dict=dist_dict,
            cities=airports,
            penalty_per_min=penalty,
            use_clipping=REWARD_CONFIG["train_use_clipping"],
        )
        agents = build_variant_agents(dummy_env, ACTIVE_ALGOS, ACTIVE_VARIANTS)
        meta_dims = (len(flights), len(airports), len(planes))
        logger.info(f"Schedule: {len(flights)} flights, {len(planes)} aircraft, {len(airports)} airports")

        training_scores = train_agents_on_schedule(
            agents, flights, planes, dist_dict, airports, penalty, meta_dims, run.checkpoint_dir, 1, f"iter{i + 1}"
        )
        eval_results = evaluate_models_on_schedule(
            agents,
            flights,
            planes,
            dist_dict,
            airports,
            penalty,
            f"iteration {i + 1}/{N_ITERATIONS}",
            include_cplex=INCLUDE_CPLEX,
        )
        iter_viz[f"{i}"] = (training_scores, eval_results)

    model_names = list(iter_viz["0"][1].keys())
    run.save_results(
        iter_viz,
        metrics={
            "mean_profit": {m: float(np.mean([iter_viz[k][1][m][0] for k in iter_viz])) for m in model_names},
            "mean_delay_min": {m: float(np.mean([iter_viz[k][1][m][1] for k in iter_viz])) for m in model_names},
        },
    )


if __name__ == "__main__":
    main()
