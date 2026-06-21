CITIES = [
    "praga",
    "milan",
    "lodz",
    "paris",  # Added for more complexity
    "madryt",
    "berlin",
    "london",
    "barcelona",
    "qatar",
    "dubai",
]

AIRPORTS = [
    # "praga",
    # "milan",
    "lodz",
    # "paris",
    # "madryt",
    "berlin",
    # "london",
    "barcelona",
    # "qatar",
    # "dubai",
]
N_FLIGHTS = 10
N_EVAL_FLIGHTS = N_FLIGHTS * 5
FIRST_FLIGHT_HOUR = 5
LAST_FLIGHT_HOUR = 23
MIN_PASS = 100
MAX_PASS = 180

PLANES_TEMPLATES = {
    "BOEING": {
        "fuel_use": 900.0,
        "seats": 150,
        "speed": 900,
        "base_fare": 50,
        "rate_per_km": 0.15,
    },
    "AIRBUS": {
        "fuel_use": 850.0,
        "seats": 120,
        "speed": 850,
        "base_fare": 120,
        "rate_per_km": 0.28,
    },
}

# --- TRAINING / EXPERIMENT CONFIG ---
# Iteration control and model-specific training hyperparameters
N_ITERATIONS = 1

MODEL_HYPERPARAMS = {
    "DQN": {
        "lr": 0.0001,
        "gamma": 0.95,
        "epsilon_decay": 0.999,
        "init_epsilon": 0.5,
        "min_epsilon": 0.1,
        "batch_size": 64,
        "tau": 0.005,
    },
    "DOUBLE_DQN": {
        "lr": 0.000075,
        "gamma": 0.95,
        "epsilon_decay": 0.999,
        "init_epsilon": 0.5,
        "min_epsilon": 0.1,
        "batch_size": 64,
        "tau": 0.005,
    },
    "Q_LEARNING": {
        "lr": 0.1,
        "gamma": 0.9,
        "epsilon": 1.0,
        "epsilon_decay": 0.999995,
        "min_epsilon": 0.01,
        "use_decay": True,
    },
}

MODEL_TRAINING_PARAMS = {
    "DQN": {"n_episodes": 50_000, "log_interval": 500},
    "DOUBLE_DQN": {"n_episodes": 50_000, "log_interval": 500},
    "Q_LEARNING": {"n_episodes": 100_000, "log_interval": 2_000},
}

# Training stability controls for value-based agents.
RL_TRAINING_CONFIG = {
    "dqn_reward_scale": 0.001,
}

# Reward-shaping controls. Keeping clipping enabled prevents delay penalties from
# dominating training and evaluation when schedules become highly congested.
REWARD_CONFIG = {
    "train_use_clipping": True,
    "eval_use_clipping": True,
    "final_eval_use_clipping": True,
    "penalty_per_min": 5,
}

# Legacy aliases for compatibility with existing training code
DQN_HYPERPARAMS = MODEL_HYPERPARAMS["DQN"]
DOUBLE_DQN_HYPERPARAMS = MODEL_HYPERPARAMS["DOUBLE_DQN"]
Q_LEARNING_PARAMS = MODEL_HYPERPARAMS["Q_LEARNING"]

# Legacy training parameter aliases
N_DQN_EPISODES = MODEL_TRAINING_PARAMS["DQN"]["n_episodes"]
N_DOUBLE_DQN_EPISODES = MODEL_TRAINING_PARAMS["DOUBLE_DQN"]["n_episodes"]
N_Q_EPISODES = MODEL_TRAINING_PARAMS["Q_LEARNING"]["n_episodes"]
DQN_LOG_INTERVAL = MODEL_TRAINING_PARAMS["DQN"]["log_interval"]
DOUBLE_DQN_LOG_INTERVAL = MODEL_TRAINING_PARAMS["DOUBLE_DQN"]["log_interval"]
Q_LOG_INTERVAL = MODEL_TRAINING_PARAMS["Q_LEARNING"]["log_interval"]

# Legacy RL tuning aliases
DQN_REWARD_SCALE = RL_TRAINING_CONFIG["dqn_reward_scale"]

# Legacy reward aliases
TRAIN_USE_CLIPPING = REWARD_CONFIG["train_use_clipping"]
EVAL_USE_CLIPPING = REWARD_CONFIG["eval_use_clipping"]
FINAL_EVAL_USE_CLIPPING = REWARD_CONFIG["final_eval_use_clipping"]
PENALTY_PER_MIN = REWARD_CONFIG["penalty_per_min"]

# Legacy epsilon aliases
INIT_EPSILON = DQN_HYPERPARAMS["init_epsilon"]
MIN_EPSILON_TARGET = DQN_HYPERPARAMS["min_epsilon"]

# Early stopping defaults
EARLY_STOPPING_CONFIG = {
    "patience": 3000,
    "rolling_window_size": 300,
    "improvement_threshold": 3000,
    "min_epsilon_to_stop": 0.12,
}

# Checkpoint directory
CHECKPOINT_DIR = "checkpoints"

# Default fleet configuration used by training utilities
FLEET_CONFIG = {"BOEING": 2, "AIRBUS": 2}
