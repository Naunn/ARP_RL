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
    # "lodz",
    # "paris",
    # "madryt",
    "berlin",
    # "london",
    "barcelona",
    # "qatar",
    "dubai",
]
N_FLIGHTS = 20
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
# Iteration control and DQN training hyperparameters
N_ITERATIONS = 1

# DQN training parameters
N_DQN_EPISODES = 25_000
DQN_LOG_INTERVAL = 200

# Agent hyperparameters
INIT_EPSILON = 0.5
MIN_EPSILON_TARGET = 0.1

DQN_HYPERPARAMS = {
    "lr": 0.0001,
    "gamma": 0.95,
    "epsilon_decay": 0.999,
    "batch_size": 64,
    "tau": 0.005,
}

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
