"""Central application configurations for reinforcement learning airline schedules."""

from typing import Any, Dict

# Global RNG seed applied by every experiment via src.utils.set_seed, so reruns are reproducible.
SEED: int = 42

N_ITERATIONS: int = 5

# --- STRUCTURAL CONFIGURATIONS ---
MODEL_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "DQN": {
        "lr": 0.0005,
        "gamma": 0.99,
        "epsilon_decay": 0.9995,
        "init_epsilon": 0.8,
        "min_epsilon": 0.1,
        "batch_size": 256,
        "tau": 0.001,
        "use_attention": True,
        "use_expert_bias": True,
        "expert_bias_weight": 0.05,
        "use_action_masking": False,
    },
    "DOUBLE_DQN": {
        "lr": 0.0005,
        "gamma": 0.99,
        "epsilon_decay": 0.9995,
        "init_epsilon": 0.8,
        "min_epsilon": 0.1,
        "batch_size": 256,
        "tau": 0.001,
        "use_attention": True,
        "use_expert_bias": True,
        "expert_bias_weight": 0.05,
        "use_action_masking": False,
    },
}

MODEL_TRAINING_PARAMS: Dict[str, Dict[str, int]] = {
    "DQN": {"n_episodes": 500, "log_interval": 10},
    "DOUBLE_DQN": {"n_episodes": 500, "log_interval": 10},
}

# Ablation variants compared across experiments: each maps to hyperparameter overrides applied
# on top of MODEL_HYPERPARAMS[algo]. "full" is the base config (no overrides). Order here is the
# order variants are trained/displayed in.
AGENT_VARIANT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_bias": {
        "use_expert_bias": False,
        "use_action_masking": False,
    },
    "idle": {
        "use_attention": False,
        "use_expert_bias": False,
        "use_action_masking": False,
    },
}

# Severity parameters for the synthetic disruptions injected during disruption-recovery training.
DISRUPTION_ACTIONS_CONFIG: Dict[str, Any] = {
    "delay_count_fraction": 1 / 3,  # fraction of flights delayed, e.g. N // 3
    "delay_min_minutes": 60,
    "delay_max_minutes": 180,
    "replace_airport_field": "origin",
    "replace_airport_method": "closest",
}

RL_TRAINING_CONFIG: Dict[str, float] = {
    "dqn_reward_scale": 0.001,
}

REWARD_CONFIG: Dict[str, Any] = {
    "train_use_clipping": True,
    "eval_use_clipping": True,
    "final_eval_use_clipping": True,
    "penalty_per_min": 150,
}

EARLY_STOPPING_CONFIG: Dict[str, Any] = {
    "patience": 10000,
    "rolling_window_size": 500,
    "improvement_threshold": 100.0,
    "min_epsilon_to_stop": 0.02,
}

# Every experiment run writes its config, log, results, and checkpoints to RUNS_DIR/<run_id>/.
RUNS_DIR: str = "runs"
