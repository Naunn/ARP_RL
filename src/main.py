import time

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QAgent
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
    QLearningSolver,
    RandomSolver,
    run_unified_execution,
)
from src.utils.fleet import generate_fleet
from src.utils.schedule import (
    check_global_feasibility,
    generate_random_flights,
)

logger = get_logger("plane_assignment")

# --- DATA SETUP ---
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

dist_dict = create_dist_dict(CITIES)

# --- FLEET GENERATION ---
fleet_config = {"BOEING": 2, "AIRBUS": 1}
PLANES = generate_fleet(fleet_config)

# # Add custom one-off planes
# PLANES["private_jet_custom"] = {
#     "fuel_use": 400.0,
#     "seats": 10,
#     "speed": 950,
#     "base_fare": 500,
#     "rate_per_km": 0.80,
# }

# Log the generated fleet for verification
logger.info(f"Fleet generated with {len(PLANES)} aircraft:")
for name in PLANES.keys():
    logger.info(f" - {name.upper()}")

# --- FEASIBILITY CHECKS ---
# The function returns (max_planes_needed, utilization)
max_req_planes, utilization = check_global_feasibility(
    FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
)
num_planes = len(PLANES)

logger.info("=" * 60)
logger.info("SCHEDULE FEASIBILITY REPORT")
logger.info(f"Total Flights: {N_FLIGHTS} | Fleet Size: {num_planes}")
logger.info(f"Global Utilization: {utilization:.1f}%")
logger.info(f"Peak Concurrency: {max_req_planes} planes required simultaneously")

# Logic check for the "Challenge" status
if max_req_planes > num_planes:
    logger.warning(
        f"!!! BOTTLE NECK: You may need {max_req_planes} planes at once, but only have {num_planes} !!!"
    )
    logger.warning("Status: UNSOLVABLE - Agent will focus on Damage Control/Triage.")
elif utilization > 100:
    logger.warning(
        "!!! OVERWORK: Total flight hours exceed fleet's total time budget !!!"
    )
    logger.warning("Status: UNSOLVABLE - Planes physically cannot finish all work.")
elif utilization > 80:
    logger.info(
        "Status: TIGHT - Solvable, but relocation delays will likely trigger penalties."
    )
else:
    logger.info(
        "Status: HEALTHY - Fleet should handle this with high profit potential."
    )

logger.info("=" * 60)

# ==========================================
# --- LOOP 1: TABULAR Q-LEARNING TRAINING ---
# ==========================================
env = AirlineEnv(FLIGHTS, PLANES, dist_dict, cities=CITIES, use_clipping=False)
agent = QAgent(n_actions=len(env.planes), epsilon=0.2, use_decay=False)

# --- TRAINING SETTINGS ---
n_episodes = 250_000
log_interval = 10_000
scores = []
start_wall_time = time.time()

logger.info(f"Starting training for {n_episodes:,} episodes...")

for i in range(1, n_episodes + 1):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

    scores.append(episode_reward)
    agent.decay_epsilon()

    if i % log_interval == 0:
        avg_score = np.mean(scores[-log_interval:])
        percent_complete = (i / n_episodes) * 100

        # Calculate ETA
        elapsed_time = time.time() - start_wall_time
        avg_time_per_ep = elapsed_time / i
        remaining_sec = int((n_episodes - i) * avg_time_per_ep)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))

        logger.info(
            f"Progress: {percent_complete:>5.1f}% | "
            f"Epsilon: {agent.epsilon:.4f} | "
            f"Avg Profit: ${avg_score:>10.0f} | "
            f"ETA: {eta_str}"
        )


# ==========================================
# --- LOOP 2: DEEP Q-NETWORK TRAINING ---
# ==========================================
logger.info("\n" + "=" * 60)
logger.info("STARTING INDEPENDENT DQN TRAINING PHASE")
logger.info("=" * 60)

n_dqn_episodes = 25_000  # Neural networks generalize state vectors incredibly quickly
dqn_log_interval = 200

init_epsilon = 1.0
min_epsilon_target = 0.1

# calculate the precise decay multiplier to reach target epsilon at 2/3 training window
computed_decay = (min_epsilon_target / init_epsilon) ** (
    1.0 / int(n_dqn_episodes * (2 / 3))
)

dqn_agent = DQNAgent(
    state_dim=env.get_state_dim(),
    n_actions=len(env.planes),
    lr=0.0001,
    gamma=0.95,
    epsilon=init_epsilon,
    epsilon_decay=computed_decay,
    min_epsilon=min_epsilon_target,
    batch_size=64,
    tau=0.005,
)

dqn_scores = []
dqn_start_time = time.time()

for i in range(1, n_dqn_episodes + 1):
    raw_state = env.reset()
    state = env.get_vector_state(raw_state)
    done = False
    episode_reward = 0

    while not done:
        action = dqn_agent.choose_action(state, use_epsilon=True)
        next_raw_state, reward, done, _ = env.step(action)
        next_state = env.get_vector_state(next_raw_state)

        # Scale down reward to ensure stable weight gradients inside the network
        scaled_reward = reward * 0.001
        dqn_agent.store_transition(state, action, scaled_reward, next_state, done)
        dqn_agent.learn()

        state = next_state
        episode_reward += reward

    dqn_scores.append(episode_reward)
    dqn_agent.decay_epsilon()

    if i % dqn_log_interval == 0:
        avg_score_dqn = np.mean(dqn_scores[-dqn_log_interval:])
        pct_dqn = (i / n_dqn_episodes) * 100

        dqn_elapsed_time = time.time() - dqn_start_time
        avg_time_per_dqn_ep = dqn_elapsed_time / i
        dqn_remaining_sec = int((n_dqn_episodes - i) * avg_time_per_dqn_ep)
        dqn_eta_str = time.strftime("%H:%M:%S", time.gmtime(dqn_remaining_sec))

        logger.info(
            f"[DQN Phase] Progress: {pct_dqn:>5.1f}% | "
            f"Epsilon: {dqn_agent.epsilon:.4f} | "
            f"Avg Profit: ${avg_score_dqn:>10.0f} | "
            f"ETA: {dqn_eta_str}"
        )


logger.info("Training complete. Moving to final schedule execution...")

# --- FINAL COMPARISON EXECUTION ---
# Initialize the Solvers
solvers = {
    "Random": RandomSolver(),
    "Greedy": ClosestPlaneGreedySolver(),
    "Q_Agent": QLearningSolver(agent),
    "DQN_Agent": DQNSolver(dqn_agent),
}

# Define the Schedules to test
schedules = {"TRAINING_DATA": FLIGHTS, "DISRUPTION_TEST": FLIGHTS_TEST}

# Execution Loop
results = {}
for sched_name, flight_list in schedules.items():
    results[sched_name] = {}
    for solver_name, solver_obj in solvers.items():
        # Capturing both values from the tuple return
        p, d = run_unified_execution(env, solver_obj, flight_list, solver_name)
        # Wrapping them in a dictionary for the scoreboard to read
        results[sched_name][solver_name] = {"profit": p, "delay": d}

# --- FINAL UNIFIED SCOREBOARD ---
# Total width is 95 characters
logger.info("\n" + "=" * 95)
header = f"{'STRATEGY':<15} | {'TRAIN PROFIT':>14} | {'TRAIN DELAY':>11} | {'TEST PROFIT':>14} | {'TEST DELAY':>11}"
logger.info(header)
logger.info("-" * 95)

for solver_name in solvers.keys():
    # Retrieve nested data dictionaries
    tr = results["TRAINING_DATA"][solver_name]
    ts = results[
        "DISRUPTION_TEST"
    ][
        solver_name
    ]  # TODO create actual scalable distruptions, but in a way of how schedule would look like after beeing distrupted, i.e.:
    # drop plane from the fleet,
    # overwrite airport with different airport,
    # extend the "landing" time (which we will be treating as plane start to be available at).

    # Format strings with currency/units to ensure fixed-width alignment
    train_p_str = f"${tr['profit']:>13,.0f}"
    train_d_str = f"{tr['delay']:>10.0f}m"
    test_p_str = f"${ts['profit']:>13,.0f}"
    test_d_str = f"{ts['delay']:>10.0f}m"

    logger.info(
        f"{solver_name:<15} | {train_p_str} | {train_d_str} | {test_p_str} | {test_d_str}"
    )

logger.info("=" * 95)
logger.info("ANALYSIS GUIDE:")
logger.info("Check 'Test Profit': Primary metric for agent robustness.")
logger.info("Check 'Test Delay': If RL Delay > Greedy but RL Profit is higher, agent")
logger.info("learned to prioritize high-value flights during disruptions.")
logger.info("=" * 95)
