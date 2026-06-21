"""
Agent and environment initialization for the training pipeline.
Handles setup of DQN agent, dummy environment, and fleet/dist data.
"""

import os

import torch

from src.agents.dqn_agent import DQNAgent
from src.config import (
    AIRPORTS,
    CHECKPOINT_DIR,
    DQN_HYPERPARAMS,
    FIRST_FLIGHT_HOUR,
    FLEET_CONFIG,
    INIT_EPSILON,
    LAST_FLIGHT_HOUR,
    MAX_PASS,
    MIN_EPSILON_TARGET,
    MIN_PASS,
    N_FLIGHTS,
)
from src.utils.dist import create_dist_dict
from src.utils.envs import AirlineEnv
from src.utils.fleet import generate_fleet
from src.utils.schedule import generate_trap_schedule


def setup_checkpoint_dir():
    """Create checkpoints directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_model_filename(
    iteration,
    flights_count,
    cities_count,
    fleet_size,
    episodes,
    model_tag="DQN",
):
    """Generates a standardized, easily sortable filename for historical tracking."""
    normalized_tag = model_tag.upper().replace(" ", "_")
    return os.path.join(
        CHECKPOINT_DIR,
        f"{normalized_tag}_airline_episodes{episodes}_flights{flights_count}_cities{cities_count}_fleet{fleet_size}_iter{iteration:03d}.pth",
    )


def initialize_static_data():
    """Initialize static data: distance dict, fleet, and planes.

    Returns:
        tuple: (dist_dict, PLANES, num_planes)
    """
    dist_dict = create_dist_dict(AIRPORTS)
    PLANES = generate_fleet(FLEET_CONFIG)
    num_planes = len(PLANES)
    return dist_dict, PLANES, num_planes


def initialize_dummy_environment(dist_dict, PLANES):
    """Create a dummy environment to resolve state dimensions.

    Args:
        dist_dict: Distance dictionary for airports
        PLANES: Fleet of aircraft

    Returns:
        AirlineEnv: Dummy environment for state dimension inference
    """
    dummy_env = AirlineEnv(
        generate_trap_schedule(
            n=N_FLIGHTS,
            cities=AIRPORTS,
            start_time_range=(FIRST_FLIGHT_HOUR * 60, LAST_FLIGHT_HOUR * 60),
            pass_range=(MIN_PASS, MAX_PASS),
        ),
        PLANES,
        dist_dict,
        cities=AIRPORTS,
    )
    return dummy_env


def initialize_dqn_agent(dummy_env, agent_cls=DQNAgent, hyperparams=None):
    """Initialize a DQN-style agent with hyperparameters from config.

    Args:
        dummy_env: Environment to extract state dimensions from
        agent_cls: Agent class to instantiate (DQNAgent or DoubleDQNAgent)
        hyperparams: Optional hyperparameter mapping to override defaults

    Returns:
        DQNAgent: Initialized agent ready for training
    """
    if hyperparams is None:
        hyperparams = DQN_HYPERPARAMS

    fleet_dim, flight_feature_dim = dummy_env.get_state_dim()

    agent = agent_cls(
        fleet_dim=fleet_dim,
        flight_feature_dim=flight_feature_dim,
        n_actions=len(dummy_env.planes),
        lr=hyperparams["lr"],
        gamma=hyperparams["gamma"],
        epsilon=hyperparams.get("init_epsilon", INIT_EPSILON),
        epsilon_decay=hyperparams["epsilon_decay"],
        min_epsilon=hyperparams.get("min_epsilon", MIN_EPSILON_TARGET),
        batch_size=hyperparams["batch_size"],
        tau=hyperparams["tau"],
        hidden_dim=hyperparams.get("hidden_dim", 256),
    )
    return agent


def load_model_checkpoint(dqn_agent, model_path):
    """Load a trained model checkpoint into the agent.

    Args:
        dqn_agent: DQNAgent instance
        model_path: Path to the checkpoint file

    Returns:
        bool: True if loading succeeded, False otherwise
    """
    try:
        dqn_agent.policy_net.load_state_dict(torch.load(model_path))
        dqn_agent.policy_net.eval()
        return True
    except Exception:
        return False


def reset_agent_exploration(dqn_agent, n_episodes):
    """Reset DQN exploration parameters for a new training iteration.

    Args:
        dqn_agent: DQNAgent instance
        n_episodes: Number of episodes for this training iteration
    """
    dqn_agent.epsilon = max(dqn_agent.epsilon, INIT_EPSILON)

    # Re-calculate the precise decay multiplier
    computed_decay = (MIN_EPSILON_TARGET / INIT_EPSILON) ** (
        1.0 / int(n_episodes * (2 / 3))
    )
    dqn_agent.epsilon_decay = computed_decay
