"""Central application configurations for reinforcement learning airline schedules."""

from typing import Any, Dict, List

# --- ENVIRONMENT SEED DATA ---
CITIES: List[str] = [
    "praga",
    "milan",
    "lodz",
    "paris",
    "madryt",
    "berlin",
    "london",
    "barcelona",
    "qatar",
    "dubai",
]

AIRPORTS: List[str] = ["lodz", "berlin", "barcelona"]

N_FLIGHTS: int = 10
N_EVAL_FLIGHTS: int = N_FLIGHTS * 5
FIRST_FLIGHT_HOUR: int = 5
LAST_FLIGHT_HOUR: int = 23
MIN_PASS: int = 100
MAX_PASS: int = 180

PLANES_TEMPLATES: Dict[str, Dict[str, Any]] = {
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

N_ITERATIONS: int = 1

# --- STRUCTURAL CONFIGURATIONS ---
MODEL_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "DQN": {
        "lr": 0.000075,
        "gamma": 0.95,
        "epsilon_decay": 0.999,
        "init_epsilon": 0.5,
        "min_epsilon": 0.1,
        "batch_size": 128,
        "tau": 0.001,
    },
    "DOUBLE_DQN": {
        "lr": 0.000075,
        "gamma": 0.95,
        "epsilon_decay": 0.999,
        "init_epsilon": 0.5,
        "min_epsilon": 0.1,
        "batch_size": 128,
        "tau": 0.001,
    },
}

MODEL_TRAINING_PARAMS: Dict[str, Dict[str, int]] = {
    "DQN": {"n_episodes": 10, "log_interval": 1},
    "DOUBLE_DQN": {"n_episodes": 10, "log_interval": 1},
}

RL_TRAINING_CONFIG: Dict[str, float] = {
    "dqn_reward_scale": 0.001,
}

REWARD_CONFIG: Dict[str, Any] = {
    "train_use_clipping": True,
    "eval_use_clipping": True,
    "final_eval_use_clipping": True,
    "penalty_per_min": 5,
}

EARLY_STOPPING_CONFIG: Dict[str, Any] = {
    "patience": 3000,
    "rolling_window_size": 300,
    "improvement_threshold": 3000,
    "min_epsilon_to_stop": 0.12,
}

CHECKPOINT_DIR: str = "checkpoints"
FLEET_CONFIG: Dict[str, int] = {"BOEING": 2, "AIRBUS": 2}
