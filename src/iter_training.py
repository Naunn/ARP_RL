import copy

import numpy as np
import torch

from src.agents.dqn_agent import DoubleDQNAgent
from src.config import (
    AIRPORTS,
    CITIES,
    DOUBLE_DQN_HYPERPARAMS,
    DOUBLE_DQN_LOG_INTERVAL,
    DQN_LOG_INTERVAL,
    EVAL_USE_CLIPPING,
    FINAL_EVAL_USE_CLIPPING,
    FIRST_FLIGHT_HOUR,
    LAST_FLIGHT_HOUR,
    MAX_PASS,
    MIN_PASS,
    N_DOUBLE_DQN_EPISODES,
    N_DQN_EPISODES,
    N_EVAL_FLIGHTS,
    N_FLIGHTS,
    N_ITERATIONS,
    N_Q_EPISODES,
    PENALTY_PER_MIN,
    TRAIN_USE_CLIPPING,
)
from src.log_config import get_logger
from src.training import (
    TrainingLogger,
    evaluate_final_test,
    evaluate_iteration,
    get_model_filename,
    initialize_dqn_agent,
    initialize_dummy_environment,
    initialize_q_agent,
    initialize_static_data,
    log_final_evaluation_start,
    log_training_complete,
    reset_agent_exploration,
    reset_q_agent_exploration,
    setup_checkpoint_dir,
    train_dqn_iteration,
    train_q_learning_iteration,
)
from src.utils.disruptions import DisruptionGenerator
from src.utils.envs import (
    AirlineEnv,
    ClosestPlaneGreedySolver,
    DQNSolver,
    QLearningSolver,
    RandomSolver,
)
from src.utils.schedule import (
    check_global_feasibility,
    generate_trap_schedule,
)

logger = get_logger("plane_assignment")

# Setup
setup_checkpoint_dir()

# Initialize static data
dist_dict, PLANES, num_planes = initialize_static_data()

# Initialize agents
dummy_env = initialize_dummy_environment(dist_dict, PLANES)
dqn_agent = initialize_dqn_agent(dummy_env)
double_dqn_agent = initialize_dqn_agent(
    dummy_env,
    agent_cls=DoubleDQNAgent,
    hyperparams=DOUBLE_DQN_HYPERPARAMS,
)
q_agent = initialize_q_agent(n_actions=len(dummy_env.planes))


# ===================================================
# --- MAIN TRAINING LOOP ---
# ===================================================
for iteration in range(1, N_ITERATIONS + 1):
    TrainingLogger.log_iteration_start(iteration, N_ITERATIONS)

    # Reset exploration for new iteration
    reset_agent_exploration(dqn_agent, N_DQN_EPISODES)
    reset_q_agent_exploration(q_agent)
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
    q_train_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=TRAIN_USE_CLIPPING,
    )
    dqn_train_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=TRAIN_USE_CLIPPING,
    )
    double_dqn_train_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=TRAIN_USE_CLIPPING,
    )
    eval_env = AirlineEnv(
        FLIGHTS,
        PLANES,
        dist_dict,
        cities=AIRPORTS,
        penalty_per_min=PENALTY_PER_MIN,
        use_clipping=EVAL_USE_CLIPPING,
    )

    # Train Q-learning and both DQN variants
    q_scores = train_q_learning_iteration(q_agent, q_train_env, N_Q_EPISODES, iteration)
    dqn_scores = train_dqn_iteration(
        dqn_agent,
        dqn_train_env,
        N_DQN_EPISODES,
        iteration,
        early_stopping=True,
        log_interval=DQN_LOG_INTERVAL,
        model_name="DQN",
    )
    double_dqn_scores = train_dqn_iteration(
        double_dqn_agent,
        double_dqn_train_env,
        N_DOUBLE_DQN_EPISODES,
        iteration,
        early_stopping=True,
        log_interval=DOUBLE_DQN_LOG_INTERVAL,
        model_name="DOUBLE_DQN",
    )

    # Save final model weights
    model_path = get_model_filename(
        iteration, N_FLIGHTS, len(AIRPORTS), num_planes, N_DQN_EPISODES, model_tag="DQN"
    )
    q_learning_path = get_model_filename(
        iteration,
        N_FLIGHTS,
        len(AIRPORTS),
        num_planes,
        N_Q_EPISODES,
        model_tag="Q_LEARNING",
    )
    double_dqn_path = get_model_filename(
        iteration,
        N_FLIGHTS,
        len(AIRPORTS),
        num_planes,
        N_DQN_EPISODES,
        model_tag="DOUBLE_DQN",
    )
    torch.save(dqn_agent.policy_net.state_dict(), model_path)
    torch.save(q_agent.q_table, q_learning_path)
    torch.save(double_dqn_agent.policy_net.state_dict(), double_dqn_path)
    TrainingLogger.log_model_saved(model_path)
    TrainingLogger.log_model_saved(q_learning_path)
    TrainingLogger.log_model_saved(double_dqn_path)

    # Evaluate on both training and disrupted schedules
    solvers = {
        "Random": RandomSolver(),
        "Greedy": ClosestPlaneGreedySolver(),
        "Q_Learning": QLearningSolver(copy.deepcopy(q_agent)),
        "DQN_Agent": DQNSolver(copy.deepcopy(dqn_agent)),
        "Double_DQN": DQNSolver(copy.deepcopy(double_dqn_agent)),
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
    FINAL_TEST_FLIGHTS,
    PLANES,
    dist_dict,
    cities=AIRPORTS,
    penalty_per_min=PENALTY_PER_MIN,
    use_clipping=FINAL_EVAL_USE_CLIPPING,
)

max_req_planes, utilization = check_global_feasibility(
    FINAL_TEST_FLIGHTS, list(PLANES.keys()), PLANES, dist_dict
)

log_final_evaluation_start(utilization, max_req_planes)

# Prepare model paths for evaluation
model_paths = {
    iteration: get_model_filename(
        iteration, N_FLIGHTS, len(AIRPORTS), num_planes, N_DQN_EPISODES, model_tag="DQN"
    )
    for iteration in range(1, N_ITERATIONS + 1)
}

double_dqn_paths = {
    iteration: get_model_filename(
        iteration,
        N_FLIGHTS,
        len(AIRPORTS),
        num_planes,
        N_DQN_EPISODES,
        model_tag="DOUBLE_DQN",
    )
    for iteration in range(1, N_ITERATIONS + 1)
}

q_learning_paths = {
    iteration: get_model_filename(
        iteration,
        N_FLIGHTS,
        len(AIRPORTS),
        num_planes,
        N_Q_EPISODES,
        model_tag="Q_LEARNING",
    )
    for iteration in range(1, N_ITERATIONS + 1)
}

evaluate_final_test(
    final_eval_env,
    FINAL_TEST_FLIGHTS,
    dqn_agent,
    model_paths,
    q_agent=q_agent,
    q_learning_model_paths=q_learning_paths,
    double_dqn_agent=double_dqn_agent,
    double_dqn_model_paths=double_dqn_paths,
)
