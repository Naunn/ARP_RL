"""Simplest baseline pipeline: train DQN + Double DQN (base config, no ablation variants, no
disruptions) on the full data/training instance for N_ITERATIONS, evaluating against the
Random/Greedy baselines after each iteration and once more at the end.

Results land in runs/<timestamp>_iter_training/.
"""

import pandas as pd
import torch

from src.config import MODEL_HYPERPARAMS, MODEL_TRAINING_PARAMS, N_ITERATIONS, REWARD_CONFIG, SEED
from src.experiments.experiment_setup import (
    AGENT_CLASSES,
    evaluate_agent_performance,
    print_results_table,
    resolve_project_root,
)
from src.experiments.run_tracking import start_run
from src.instances import build_flight_pool, build_planes
from src.utils import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
    create_dist_dict_from_airports,
    get_model_filename,
    initialize_agent,
    log_iteration_start,
    logger,
    reset_agent_exploration,
    set_seed,
    train_dqn_iteration,
)

ALGO_DISPLAY = {"DQN": "DQN", "DOUBLE_DQN": "Double DQN"}


def main() -> None:
    set_seed(SEED)
    run = start_run("iter_training", {"algos": list(ALGO_DISPLAY), "n_iterations": N_ITERATIONS})

    training_data_dir = resolve_project_root() / "data" / "training"
    flights_df = pd.read_csv(training_data_dir / "flights.csv")
    itineraries_df = pd.read_csv(training_data_dir / "itineraries.csv")
    aircraft_df = pd.read_csv(training_data_dir / "aircraft.csv").drop_duplicates(
        subset="fixed_cost  hourly_cost initial_airport  seats  speed".split()
    )

    planes = build_planes(aircraft_df)
    flights = build_flight_pool(flights_df, itineraries_df)
    airports = sorted(
        {f["origin"] for f in flights} | {f["dest"] for f in flights} | {p["initial_airport"] for p in planes.values()}
    )
    dist_dict = create_dist_dict_from_airports(airports_list=airports)
    penalty: int = REWARD_CONFIG["penalty_per_min"]
    meta_dims = (len(flights), len(airports), len(planes))

    def make_env(clipping_key: str) -> AirlineEnv:
        return AirlineEnv(flights, planes, dist_dict, airports, penalty, use_clipping=REWARD_CONFIG[clipping_key])

    agents = {
        algo: initialize_agent(make_env("train_use_clipping"), AGENT_CLASSES[algo], MODEL_HYPERPARAMS[algo])
        for algo in ALGO_DISPLAY
    }

    def evaluate(env: AirlineEnv, label: str) -> dict:
        solvers = {
            "Random Baseline": RandomSolver(),
            "Greedy Baseline": ClosestPlaneGreedySolver(),
            **{ALGO_DISPLAY[algo]: DQNSolver(agent) for algo, agent in agents.items()},
        }
        results = {name: evaluate_agent_performance(env, solver, name) for name, solver in solvers.items()}
        print_results_table(results, label)
        return results

    per_iteration_eval = {}
    for iteration in range(1, N_ITERATIONS + 1):
        log_iteration_start(iteration, N_ITERATIONS)
        for algo, agent in agents.items():
            n_episodes = MODEL_TRAINING_PARAMS[algo]["n_episodes"]
            reset_agent_exploration(agent, n_episodes, MODEL_HYPERPARAMS[algo])
            train_dqn_iteration(
                agent,
                make_env("train_use_clipping"),
                n_episodes,
                iteration,
                model_name=algo,
                checkpoint_dir=run.checkpoint_dir,
            )
            torch.save(
                agent.policy_net.state_dict(),
                get_model_filename(run.checkpoint_dir, iteration, *meta_dims, n_episodes, algo),
            )
        per_iteration_eval[iteration] = evaluate(make_env("eval_use_clipping"), f"ITER {iteration}")

    logger.info("Global training cycle finished across all iterations; running final evaluation.")
    final_eval = evaluate(make_env("final_eval_use_clipping"), "FINAL")

    run.save_results(
        {"per_iteration_eval": per_iteration_eval, "final_eval": final_eval},
        metrics={name: {"profit": profit, "delay_min": delay} for name, (profit, delay) in final_eval.items()},
    )


if __name__ == "__main__":
    main()
