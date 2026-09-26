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

import numpy as np
import pandas as pd

from src.config import N_ITERATIONS, REWARD_CONFIG, SEED
from src.experiments.experiment_setup import (
    build_disruption_actions,
    build_variant_agents,
    evaluate_models_on_schedule,
    resolve_project_root,
    train_agents_on_schedule,
)
from src.experiments.run_tracking import start_run
from src.instances import build_flight_pool, build_planes, generate_trap_schedule
from src.utils import (
    DisruptionGenerator,
    create_dist_dict_from_airports,
    logger,
    set_seed,
)
from src.utils.envs import AirlineEnv

# Which ablation variants/algorithms this run trains and compares. See src.config.AGENT_VARIANT_OVERRIDES.
ACTIVE_ALGOS = ["DQN", "DOUBLE_DQN"]
ACTIVE_VARIANTS = ["full", "no_bias", "idle"]

N_RUNS = 1  # independent instances, each put through the full train -> disrupt/retrain cycle
N = 15  # 15 for testing random + trap (100 iter); round(flights_df.shape[0]/3) for sampled - too long for 100 iter :(
C_N = 3  # 3 for testing random + trap (100 iter); for sampled does not matter
P_N = 2  # 2 for testing random + trap (100 iter); round(aircraft_df.shape[0]/3) for sampled - too long for 100 iter :(


def main() -> None:
    set_seed(SEED)
    run = start_run(
        "disruption_recovery",
        {
            "active_algos": ACTIVE_ALGOS,
            "active_variants": ACTIVE_VARIANTS,
            "n_runs": N_RUNS,
            "n_flights": N,
            "n_cities": C_N,
            "n_planes": P_N,
            "schedule_generator": "trap",
            "n_disruption_iterations": N_ITERATIONS,
        },
    )

    training_data_dir = resolve_project_root() / "data" / "training"
    flights_df = pd.read_csv(training_data_dir / "flights.csv")
    itineraries_df = pd.read_csv(training_data_dir / "itineraries.csv")
    aircraft_df = pd.read_csv(training_data_dir / "aircraft.csv").drop_duplicates(
        subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
    )
    penalty: int = REWARD_CONFIG["penalty_per_min"]

    iter_viz: Dict[str, Any] = {}
    for i in range(N_RUNS):
        logger.info(f"\n[initial_train_cycle] Run {i + 1}/{N_RUNS}: building instance")
        real_flights = build_flight_pool(
            flights_df.sample(N).sort_values("start_min", ascending=True),
            itineraries_df,
        )
        samp_flights = pd.DataFrame(
            generate_trap_schedule(
                n=N,
                cities=list(np.random.choice(flights_df.origin.unique(), C_N)),
                start_time_range=(
                    flights_df.start_min.min(),
                    flights_df.arrival_min.max(),
                ),
                pass_range=(
                    itineraries_df.total_passenger_count.min(),
                    itineraries_df.total_passenger_count.max(),
                ),
            )
        )
        # Price the synthetic flights at the real sample's passenger-weighted average fare.
        samp_flights["total_ticket_price"] = samp_flights["pass"] * np.average(
            pd.DataFrame(real_flights)["total_ticket_price"] / pd.DataFrame(real_flights)["pass"],
            weights=pd.DataFrame(real_flights)["pass"],
        )
        flights = cast(List[Dict[str, Any]], samp_flights.to_dict("records"))

        planes = build_planes(aircraft_df.sample(P_N))
        airports = sorted(
            {f["origin"] for f in flights}
            | {f["dest"] for f in flights}
            | {p["initial_airport"] for p in planes.values()}
        )
        dist_dict = create_dist_dict_from_airports(airports_list=airports)

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
        dg = DisruptionGenerator(airports, dist_dict)

        def evaluate(schedule: List[Dict[str, Any]], label: str) -> Dict[str, tuple]:
            return evaluate_models_on_schedule(agents, schedule, planes, dist_dict, airports, penalty, label)

        def train(schedule: List[Dict[str, Any]], phase_name: str) -> None:
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
            )

        logger.info("\nPhase 1/3: Training on initial schedule...")
        train(flights, f"run{i}_initial")
        disruptions: Dict[str, tuple] = {}
        iter_viz[f"{i}"] = (
            flights,
            evaluate(flights, "POST INITIAL TRAIN"),
            disruptions,
        )

        logger.info("\nPhase 2/3: Iterative disruption cycle (generate -> evaluate -> retrain)...")
        for d in range(1, N_ITERATIONS + 1):
            logger.info(f"[disruption_cycle] Disruption {d}/{N_ITERATIONS}: generating from original schedule")
            disrupted = cast(
                List[Dict[str, Any]],
                dg.generate(flights, actions=build_disruption_actions(N)),
            )
            pre_retrain_eval = evaluate(disrupted, f"DISRUPTED PRE-RETRAIN {d}")
            train(disrupted, f"run{i}_disrupted_retrain_iter{d}")
            disruptions[f"{d}"] = (
                disrupted,
                pre_retrain_eval,
                evaluate(disrupted, f"DISRUPTED POST-RETRAIN {d}"),
                evaluate(flights, f"INITIAL SCHEDULE POST-RETRAIN {d}"),
            )

        logger.info("\nPhase 3/3: Re-evaluating final models on every disrupted schedule...")
        final_reeval = {
            key: evaluate(disruptions[key][0], f"DISRUPTED REEVAL AFTER ALL RETRAINS {key}")
            for key in sorted(disruptions, key=int)
        }
        iter_viz[f"{i}"] = (*iter_viz[f"{i}"], final_reeval)

    run.save_results(iter_viz)


if __name__ == "__main__":
    main()
