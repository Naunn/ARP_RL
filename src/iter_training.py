import numpy as np
import torch

from src.config import (
    AIRPORTS,
    CITIES,
    FIRST_FLIGHT_HOUR,
    LAST_FLIGHT_HOUR,
    MAX_PASS,
    MIN_PASS,
    N_DQN_EPISODES,
    N_EVAL_FLIGHTS,
    N_FLIGHTS,
    N_ITERATIONS,
)
from src.log_config import get_logger
from src.training import (
    TrainingLogger,
    evaluate_final_test,
    evaluate_iteration,
    get_model_filename,
    initialize_dqn_agent,
    initialize_dummy_environment,
    initialize_static_data,
    log_final_evaluation_start,
    log_training_complete,
    reset_agent_exploration,
    setup_checkpoint_dir,
    train_dqn_iteration,
)
from src.utils.disruptions import DisruptionGenerator
from src.utils.envs import AirlineEnv, ClosestPlaneGreedySolver, DQNSolver, RandomSolver
from src.utils.schedule import (
    check_global_feasibility,
    generate_trap_schedule,
)

logger = get_logger("plane_assignment")

# Setup
setup_checkpoint_dir()

# Initialize static data
dist_dict, PLANES, num_planes = initialize_static_data()

# Initialize agent
dummy_env = initialize_dummy_environment(dist_dict, PLANES)
dqn_agent = initialize_dqn_agent(dummy_env)


# ===================================================
# --- MAIN TRAINING LOOP ---
# ===================================================
for iteration in range(1, N_ITERATIONS + 1):
    TrainingLogger.log_iteration_start(iteration, N_ITERATIONS)

    # Reset exploration for new iteration
    reset_agent_exploration(dqn_agent, N_DQN_EPISODES)
    TrainingLogger.log_exploration_reset(dqn_agent.epsilon, dqn_agent.epsilon_decay)

    # Generate training schedule
    FLIGHTS = generate_trap_schedule(
        n=N_FLIGHTS,
        cities=AIRPORTS,
        start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
        pass_range=(MIN_PASS, MAX_PASS),
    )

    # Generate disrupted variant for robustness testing
    dg = DisruptionGenerator(CITIES, dist_dict)
    FLIGHTS_TEST = dg.generate(
        FLIGHTS,
        actions=[
            {
                "action": "add_delay",
                "target": "random",
                "count": max(1, N_FLIGHTS // 4),
                "min_delay": 60,
                "max_delay": 180,
            },
            {
                "action": "replace_airport",
                "target": list(np.random.choice(AIRPORTS)),
                "field": "origin",
                "method": "closest",
            },
        ],
    )

    # Check feasibility
    max_req_planes, utilization = check_global_feasibility(
        FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
    )
    TrainingLogger.log_feasibility(iteration, utilization, max_req_planes, num_planes)

    # Create environments for training and evaluation
    train_env = AirlineEnv(
        FLIGHTS, PLANES, dist_dict, cities=AIRPORTS, use_clipping=False
    )
    eval_env = AirlineEnv(
        FLIGHTS, PLANES, dist_dict, cities=AIRPORTS, use_clipping=False
    )

    # Train DQN agent with early stopping
    train_dqn_iteration(
        dqn_agent, train_env, N_DQN_EPISODES, iteration, early_stopping=True
    )

    # Save final model weights
    model_path = get_model_filename(
        iteration, N_FLIGHTS, len(AIRPORTS), num_planes, N_DQN_EPISODES
    )
    torch.save(dqn_agent.policy_net.state_dict(), model_path)
    TrainingLogger.log_model_saved(model_path)

    # Evaluate on both training and disrupted schedules
    solvers = {
        "Random": RandomSolver(),
        "Greedy": ClosestPlaneGreedySolver(),
        "DQN_Agent": DQNSolver(dqn_agent),
    }
    schedules_eval = {"TRAINING_DATA": FLIGHTS, "DISRUPTION_TEST": FLIGHTS_TEST}

    evaluate_iteration(eval_env, solvers, schedules_eval, iteration)

log_training_complete()


# ===================================================
# --- FINAL EVALUATION ON UNSEEN TEST SET ---
# ===================================================
FINAL_TEST_FLIGHTS = generate_trap_schedule(
    n=N_EVAL_FLIGHTS,
    cities=AIRPORTS,
    start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
    pass_range=(MIN_PASS, MAX_PASS),
)

final_eval_env = AirlineEnv(
    FINAL_TEST_FLIGHTS, PLANES, dist_dict, cities=AIRPORTS, use_clipping=False
)

max_req_planes, utilization = check_global_feasibility(
    FINAL_TEST_FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
)

log_final_evaluation_start(utilization, max_req_planes)

# Prepare model paths for evaluation
model_paths = {
    iteration: get_model_filename(
        iteration, N_FLIGHTS, len(AIRPORTS), num_planes, N_DQN_EPISODES
    )
    for iteration in range(1, N_ITERATIONS + 1)
}

evaluate_final_test(final_eval_env, FINAL_TEST_FLIGHTS, dqn_agent, model_paths)
