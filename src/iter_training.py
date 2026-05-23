import os
import time

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.config import (
    CITIES,
    FIRST_FLIGHT_HOUR,
    LAST_FLIGHT_HOUR,
    MAX_PASS,
    MIN_PASS,
    N_FLIGHTS,
)
from src.log_config import get_logger
from src.utils.dist import create_dist_dict
from src.utils.envs import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    RandomSolver,
    run_unified_execution,
)
from src.utils.fleet import generate_fleet
from src.utils.schedule import (
    check_global_feasibility,
    generate_random_flights,
)

logger = get_logger("plane_assignment")

# Create a checkpoints directory to keep things organized
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_model_filename(iteration, flights_count, cities_count, fleet_size, episodes):
    """Generates a standardized, easily sortable filename for historical tracking."""
    return os.path.join(
        CHECKPOINT_DIR,
        f"dqn_airline_episodes{episodes}_flights{flights_count}_cities{cities_count}_fleet{fleet_size}_iter{iteration:03d}.pth",
    )


# --- GENERAL CONFIGURATION ---
N_ITERATIONS = 10  # Number of times we regenerate schedules and continue training
N_DQN_EPISODES = 5_000  # Episodes per schedule iteration
DQN_LOG_INTERVAL = 200

# --- STATIC DATA SETUP ---
dist_dict = create_dist_dict(CITIES)
fleet_config = {"BOEING": 2, "AIRBUS": 1}
PLANES = generate_fleet(fleet_config)
num_planes = len(PLANES)

# =========================================
# --- AGENT INITIALIZATION ---
# =========================================
dummy_env = AirlineEnv(
    generate_random_flights(
        n=N_FLIGHTS,
        cities=CITIES,
        start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
        pass_range=(MIN_PASS, MAX_PASS),
    ),
    PLANES,
    dist_dict,
    cities=CITIES,
)

init_epsilon = 1.0
min_epsilon_target = 0.1

# Initialize the agent with baseline configuration
dqn_agent = DQNAgent(
    state_dim=dummy_env.get_state_dim(),
    n_actions=len(dummy_env.planes),
    lr=0.0001,
    gamma=0.95,
    epsilon=init_epsilon,
    epsilon_decay=0.999,  # Placeholder, overridden dynamically inside the loop
    min_epsilon=min_epsilon_target,
    batch_size=64,
    tau=0.005,
)

# ===================================================
# --- GLOBAL ITERATIVE TRAINING LOOP ---
# ===================================================
for iteration in range(1, N_ITERATIONS + 1):
    logger.info("\n" + "=" * 80)
    logger.info(f"STARTING SCHEDULE REGENERATION ITERATION {iteration}/{N_ITERATIONS}")
    logger.info("=" * 80)

    # 1. Reset Epsilon Exploration for the new schedule environment
    dqn_agent.epsilon = init_epsilon

    # 2. Re-calculate the precise decay multiplier to finish at 2/3 of *this* schedule's window
    computed_decay = (min_epsilon_target / init_epsilon) ** (
        1.0 / int(N_DQN_EPISODES * (2 / 3))
    )
    dqn_agent.epsilon_decay = computed_decay

    logger.info(
        f"Exploration reset: Epsilon = {dqn_agent.epsilon:.2f} | Step Decay = {dqn_agent.epsilon_decay:.6f}"
    )

    # Generate New Schedules for this iteration
    FLIGHTS = generate_random_flights(
        n=N_FLIGHTS,
        cities=CITIES,
        start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
        pass_range=(MIN_PASS, MAX_PASS),
    )
    FLIGHTS_TEST = generate_random_flights(
        n=N_FLIGHTS,
        cities=CITIES,
        start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
        pass_range=(MIN_PASS, MAX_PASS),
    )

    # Check Feasibility of the new environment
    max_req_planes, utilization = check_global_feasibility(
        FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
    )

    logger.info(f"Schedule Iteration {iteration} Feasibility:")
    logger.info(
        f"Global Utilization: {utilization:.1f}% | Peak Concurrency: {max_req_planes} planes"
    )
    if max_req_planes > num_planes or utilization > 100:
        logger.warning(
            "Status: CURRENT ENVIRONMENT IS UNSOLVABLE - Triage mode active."
        )
    else:
        logger.info("Status: ENVIRONMENT IS SOLVABLE.")

    # Instantiate the environment with the fresh flights
    env = AirlineEnv(FLIGHTS, PLANES, dist_dict, cities=CITIES, use_clipping=False)

    dqn_scores = []
    dqn_start_time = time.time()

    # Train on this environment
    for i in range(1, N_DQN_EPISODES + 1):
        raw_state = env.reset()
        state = env.get_vector_state(raw_state)
        done = False
        episode_reward = 0

        while not done:
            action = dqn_agent.choose_action(state, use_epsilon=True)
            next_raw_state, reward, done, _ = env.step(action)
            next_state = env.get_vector_state(next_raw_state)

            scaled_reward = reward * 0.001
            dqn_agent.store_transition(state, action, scaled_reward, next_state, done)
            dqn_agent.learn()

            state = next_state
            episode_reward += reward

        dqn_scores.append(episode_reward)
        dqn_agent.decay_epsilon()

        if i % DQN_LOG_INTERVAL == 0:
            avg_score_dqn = np.mean(dqn_scores[-DQN_LOG_INTERVAL:])
            pct_dqn = (i / N_DQN_EPISODES) * 100

            dqn_elapsed_time = time.time() - dqn_start_time
            avg_time_per_dqn_ep = dqn_elapsed_time / i
            dqn_remaining_sec = int((N_DQN_EPISODES - i) * avg_time_per_dqn_ep)
            dqn_eta_str = time.strftime("%H:%M:%S", time.gmtime(dqn_remaining_sec))

            logger.info(
                f"[Iter {iteration}] Progress: {pct_dqn:>5.1f}% | "
                f"Epsilon: {dqn_agent.epsilon:.4f} | "
                f"Avg Profit: ${avg_score_dqn:>10.0f} | "
                f"ETA: {dqn_eta_str}"
            )

    # Save Model Weights for this iteration
    model_path = get_model_filename(
        iteration, N_FLIGHTS, len(CITIES), num_planes, N_DQN_EPISODES
    )
    torch.save(dqn_agent.policy_net.state_dict(), model_path)
    logger.info(f"Saved checkpoint to: {model_path}")

    # Evaluate current iteration performance
    solvers = {
        "Random": RandomSolver(),
        "Greedy": ClosestPlaneGreedySolver(),
        "DQN_Agent": DQNSolver(dqn_agent),
    }

    schedules_eval = {"TRAINING_DATA": FLIGHTS, "DISRUPTION_TEST": FLIGHTS_TEST}
    results = {}

    for sched_name, flight_list in schedules_eval.items():
        results[sched_name] = {}
        for solver_name, solver_obj in solvers.items():
            p, d = run_unified_execution(env, solver_obj, flight_list, solver_name)
            results[sched_name][solver_name] = {"profit": p, "delay": d}

    # Print a mini-scoreboard for this loop iteration
    logger.info("\n" + "-" * 95)
    logger.info(f"SCOREBOARD FOR ITERATION {iteration}")
    logger.info(
        f"{'STRATEGY':<15} | {'TRAIN PROFIT':>14} | {'TRAIN DELAY':>11} | {'TEST PROFIT':>14} | {'TEST DELAY':>11}"
    )
    logger.info("-" * 95)
    for solver_name in solvers.keys():
        tr = results["TRAINING_DATA"][solver_name]
        ts = results["DISRUPTION_TEST"][solver_name]
        logger.info(
            f"{solver_name:<15} | "
            f"${tr['profit']:>13,.0f} | {tr['delay']:>10.0f}m | "
            f"${ts['profit']:>13,.0f} | {ts['delay']:>10.0f}m"
        )
    logger.info("=" * 95)

logger.info("\nGlobal training cycle finished across all iterations.")

# ===================================================
# --- FINAL POST-TRAINING EVALUATION PHASE ---
# ===================================================
logger.info("\n" + "=" * 80)
logger.info("STARTING FINAL EVALUATION ON UNSEEN TEST SCHEDULE")
logger.info("=" * 80)

# Generate an entirely fresh test schedule the models never saw
FINAL_TEST_FLIGHTS = generate_random_flights(
    n=100,
    cities=CITIES,
    start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
    pass_range=(MIN_PASS, MAX_PASS),
)

# Instantiate a fresh evaluation environment
final_eval_env = AirlineEnv(
    FINAL_TEST_FLIGHTS, PLANES, dist_dict, cities=CITIES, use_clipping=False
)

# Check Feasibility of this final environment
max_req_planes, utilization = check_global_feasibility(
    FINAL_TEST_FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
)
logger.info(
    f"Final Test Schedule Global Utilization: {utilization:.1f}% | Peak Concurrency: {max_req_planes} planes"
)

# Print the Final Comparison Table Header
logger.info("\n" + "=" * 55)
logger.info(
    f"{'MODEL SOURCE':<25} | {'FINAL TEST PROFIT':>14} | {'FINAL TEST DELAY':>11}"
)
logger.info("-" * 55)

# Evaluate Baselines once first
baseline_solvers = {
    "Random Baseline": RandomSolver(),
    "Greedy Baseline": ClosestPlaneGreedySolver(),
}

for name, solver_obj in baseline_solvers.items():
    p, d = run_unified_execution(final_eval_env, solver_obj, FINAL_TEST_FLIGHTS, name)
    logger.info(f"{name:<25} | ${p:>13,.0f} | {d:>10.0f}m")
logger.info("-" * 55)

# Iteratively load each historically saved model weight and test it
for iteration in range(1, N_ITERATIONS + 1):
    model_path = get_model_filename(
        iteration, N_FLIGHTS, len(CITIES), num_planes, N_DQN_EPISODES
    )

    if os.path.exists(model_path):
        try:
            # Load historical weights back into the agent
            dqn_agent.policy_net.load_state_dict(torch.load(model_path))
            dqn_agent.policy_net.eval()  # Ensure evaluation mode

            # Wrap in the solver
            historical_solver = DQNSolver(dqn_agent)

            # Execute on the completely fresh test schedule
            p, d = run_unified_execution(
                final_eval_env,
                historical_solver,
                FINAL_TEST_FLIGHTS,
                f"DQN_Iter_{iteration}",
            )

            logger.info(
                f"DQN Checkpoint (Iter {iteration:02d}) | ${p:>13,.0f} | {d:>10.0f}m"
            )

        except Exception as e:
            logger.error(f"Failed to evaluate checkpoint {model_path}: {e}")
    else:
        logger.warning(f"Checkpoint not found for iteration {iteration}: {model_path}")

logger.info("=" * 55)
