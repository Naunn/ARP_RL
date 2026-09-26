"""Single-model, single-instance dev playground.

Trains ONE model configuration on ONE real ROADEF instance for a single training iteration, and
reports how it behaved: the per-episode reward curve, wall-clock timing (for judging how a
change scales), and a post-training evaluation against the Random/Greedy baselines on that same
instance. This is the fast loop for tweaking a model or its hyperparameters and sanity-checking
it still trains and scales sensibly -- before spending the time to run it through the full
multi-iteration comparison scripts (solving_schedule_experiment.py / disruption_training_experiment.py).

Everything a run produces (config, log, results, training-curve plot, checkpoint) lands in
runs/<timestamp>_single_model/.
"""

import copy
import time

import torch

from src.analysis.plots import plot_training_curve
from src.config import AGENT_VARIANT_OVERRIDES, MODEL_HYPERPARAMS, REWARD_CONFIG, SEED
from src.experiments.experiment_setup import (
    AGENT_CLASSES,
    evaluate_agent_performance,
    print_results_table,
    resolve_project_root,
)
from src.experiments.run_tracking import start_run
from src.instances import (
    discover_roadef_instances,
    load_roadef_instance,
    subsample_instance,
)
from src.utils import (
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
    get_model_filename,
    initialize_agent,
    logger,
    reset_agent_exploration,
    set_seed,
    train_dqn_iteration,
)
from src.utils.envs import AirlineEnv

# ============================================================================
# WHAT TO RUN -- edit these to try a different model, instance, or scale.
# ============================================================================

# Any name from `discover_roadef_instances()`: A01_6088570..A10_6088590 (~600 flights/84
# aircraft) or B_01..B_10 (~1400 flights/255 aircraft, for testing at ~3x scale).
INSTANCE_NAME = "A01_6088570"

# Downsize the loaded instance for a faster dev loop -- set either to None to use the full
# instance's count instead. Start small, then raise these (and/or switch to a B_* instance
# above) once training speed at the current size is no longer the bottleneck.
MAX_FLIGHTS = 60
MAX_PLANES = 15

ALGO = "DOUBLE_DQN"  # "DQN" or "DOUBLE_DQN"
VARIANT = "idle"  # base overrides from AGENT_VARIANT_OVERRIDES: "full" / "no_bias" / "idle"
N_EPISODES = 1000  # how long to train for; MODEL_TRAINING_PARAMS[ALGO]["n_episodes"] is the default used elsewhere

# One-off agent hyperparameter overrides on top of MODEL_HYPERPARAMS[ALGO] + AGENT_VARIANT_OVERRIDES[VARIANT],
# for quick tweaks without touching config.py. Leave empty to just use the variant as-is.
CUSTOM_OVERRIDES = {
    # "hidden_dim": 512,
    # "lr": 1e-4,
}

# ============================================================================


def main() -> None:
    set_seed(SEED)

    hyperparams = copy.deepcopy(MODEL_HYPERPARAMS[ALGO])
    hyperparams.update(AGENT_VARIANT_OVERRIDES[VARIANT])
    hyperparams.update(CUSTOM_OVERRIDES)

    run = start_run(
        "single_model",
        {
            "instance": INSTANCE_NAME,
            "max_flights": MAX_FLIGHTS,
            "max_planes": MAX_PLANES,
            "algo": ALGO,
            "variant": VARIANT,
            "n_episodes": N_EPISODES,
            "custom_overrides": CUSTOM_OVERRIDES,
            "effective_hyperparams": hyperparams,
        },
    )

    instances = discover_roadef_instances(resolve_project_root())
    if INSTANCE_NAME not in instances:
        raise ValueError(f"Unknown instance {INSTANCE_NAME!r}. Available: {sorted(instances)}")

    flights, planes, airports, dist_dict = load_roadef_instance(instances[INSTANCE_NAME])
    logger.info(
        f"Loaded instance {INSTANCE_NAME} (full size): "
        f"{len(flights)} flights, {len(planes)} aircraft, {len(airports)} airports"
    )
    flights, planes, airports = subsample_instance(
        flights, planes, max_flights=MAX_FLIGHTS, max_planes=MAX_PLANES, seed=SEED
    )
    logger.info(f"Using subsample: {len(flights)} flights, {len(planes)} aircraft, {len(airports)} airports")

    penalty = REWARD_CONFIG["penalty_per_min"]
    train_env = AirlineEnv(
        flights,
        planes,
        dist_dict,
        airports,
        penalty,
        use_clipping=REWARD_CONFIG["train_use_clipping"],
    )
    agent = initialize_agent(train_env, AGENT_CLASSES[ALGO], hyperparams)

    # initialize_agent sets epsilon_decay from the static MODEL_HYPERPARAMS default (tuned for the
    # ~500-episode runs the other scripts use); reset it here so epsilon actually reaches
    # min_epsilon within N_EPISODES, whatever N_EPISODES is currently set to.
    reset_agent_exploration(agent, N_EPISODES, hyperparams)

    logger.info(f"Training {ALGO} ({VARIANT}) on {INSTANCE_NAME} for {N_EPISODES} episodes...")
    start_time = time.time()
    episode_rewards = train_dqn_iteration(
        agent,
        train_env,
        N_EPISODES,
        iteration=1,
        model_name=ALGO,
        training_name=f"single_{INSTANCE_NAME}_{VARIANT}",
        checkpoint_dir=run.checkpoint_dir,
    )
    elapsed = time.time() - start_time
    logger.info(
        f"Training finished in {elapsed:.1f}s ({elapsed / N_EPISODES * 1000:.1f}ms/episode), "
        f"final epsilon={agent.epsilon:.4f}"
    )

    meta_dims = (len(flights), len(airports), len(planes))
    checkpoint_path = get_model_filename(
        run.checkpoint_dir,
        1,
        *meta_dims,
        N_EPISODES,
        f"{ALGO}_single_{INSTANCE_NAME}_{VARIANT}",
    )
    torch.save(agent.policy_net.state_dict(), checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    eval_env = AirlineEnv(
        flights,
        planes,
        dist_dict,
        airports,
        penalty,
        use_clipping=REWARD_CONFIG["final_eval_use_clipping"],
    )
    solvers = {
        "Random Baseline": RandomSolver(),
        "Greedy Baseline": ClosestPlaneGreedySolver(),
        f"{ALGO} ({VARIANT})": DQNSolver(agent),
    }
    eval_results = {name: evaluate_agent_performance(eval_env, solver, name) for name, solver in solvers.items()}
    print_results_table(eval_results, f"{INSTANCE_NAME} POST-TRAIN")

    run.save_results(
        {"episode_rewards": episode_rewards, "eval_results": eval_results},
        metrics={
            "training_seconds": round(elapsed, 2),
            "ms_per_episode": round(elapsed / N_EPISODES * 1000, 2),
            "final_epsilon": agent.epsilon,
            "instance_size": {
                "flights": len(flights),
                "planes": len(planes),
                "airports": len(airports),
            },
            "eval": {name: {"profit": profit, "delay_min": delay} for name, (profit, delay) in eval_results.items()},
        },
    )
    plot_training_curve(
        episode_rewards,
        f"{ALGO} ({VARIANT}) on {INSTANCE_NAME} -- training reward",
        save_path=run.run_dir / "training_curve.png",
    )


if __name__ == "__main__":
    main()
