"""Computational reinforcement learning training loop engines and weight serialization hooks."""

import os
import time
from typing import Any

import numpy as np
import torch

from src.config import (
    CHECKPOINT_DIR,
    EARLY_STOPPING_CONFIG,
    MODEL_TRAINING_PARAMS,
    RL_TRAINING_CONFIG,
)
from src.utils.envs import ClosestPlaneGreedySolver
from src.utils.logging import log_checkpoint, log_early_stop, log_progress, logger


def setup_checkpoint_dir():
    """Ensures directories for historical weights exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_model_filename(
    iteration: int,
    flights_count: int,
    cities_count: int,
    fleet_size: int,
    episodes: int,
    model_tag: str = "DQN",
) -> str:
    """Generates a standardized, sortable filename path for historical weights tracking."""
    return os.path.join(
        CHECKPOINT_DIR,
        (
            f"{model_tag.upper().replace(' ', '_')}_airline_"
            f"episodes{episodes}_flights{flights_count}_"
            f"cities{cities_count}_fleet{fleet_size}_"
            f"iter{iteration:03d}.pth"
        ),
    )


def initialize_dqn_agent(env, agent_cls, hyperparams) -> Any:
    """Instantiates the specified neural agent class using dimensions from the environment."""
    fleet_dim, flight_feature_dim = env.get_state_dim()
    return agent_cls(
        fleet_dim=fleet_dim,
        flight_feature_dim=flight_feature_dim,
        n_actions=len(env.planes),
        lr=hyperparams["lr"],
        gamma=hyperparams["gamma"],
        epsilon=hyperparams["init_epsilon"],
        epsilon_decay=hyperparams["epsilon_decay"],
        min_epsilon=hyperparams["min_epsilon"],
        batch_size=int(hyperparams["batch_size"]),
        tau=hyperparams["tau"],
        hidden_dim=int(hyperparams.get("hidden_dim", 256)),
        use_attention=bool(hyperparams.get("use_attention", True)),
        use_expert_bias=bool(hyperparams.get("use_expert_bias", False)),
    )


def reset_agent_exploration(agent, n_episodes: int, hyperparams: dict):
    """Resets the exploration decay schedule window profile for the active iteration."""
    init_eps, min_eps = hyperparams["init_epsilon"], hyperparams["min_epsilon"]
    agent.epsilon = max(agent.epsilon, init_eps)
    agent.epsilon_decay = (min_eps / init_eps) ** (1.0 / int(n_episodes * (2 / 3)))


def train_dqn_episode(agent, env) -> float:
    """Executes a single continuous experience collection phase episode loop step."""
    state_tuple = env.get_vector_state(env.reset())
    action_mask = env.get_action_mask()
    expert_solver = ClosestPlaneGreedySolver() if agent.use_expert_bias else None
    done, episode_reward, scale = False, 0.0, RL_TRAINING_CONFIG["dqn_reward_scale"]

    while not done:
        expert_action = expert_solver.choose_action(state_tuple, env) if expert_solver is not None else -1
        action = agent.choose_action(
            state_tuple,
            action_mask=action_mask,
            use_epsilon=True,
        )
        next_raw_state, reward, done, _ = env.step(action)
        next_state_tuple = env.get_vector_state(next_raw_state)
        next_action_mask = env.get_action_mask(next_raw_state)

        agent.store_transition(
            *state_tuple,
            action,
            reward * scale,
            *next_state_tuple,
            done,
            action_mask,
            next_action_mask,
            expert_action,
        )
        agent.learn()

        state_tuple = next_state_tuple
        action_mask = next_action_mask
        episode_reward += reward

    agent.decay_epsilon()
    return episode_reward


def train_dqn_iteration(
    agent,
    train_env,
    n_episodes: int,
    iteration: int,
    early_stopping: bool = True,
    model_name: str = "DQN",
) -> list[float]:
    """Runs a complete generational training lifecycle iteration containing n_episodes."""
    scores = []
    start_time = time.time()
    log_interval: int = MODEL_TRAINING_PARAMS[model_name]["log_interval"]
    logger.info(f"[{model_name}] Starting execution loop ({n_episodes} eps)")

    best_rolling_profit = float("-inf")
    patience_counter = 0
    cfg = EARLY_STOPPING_CONFIG

    for ep in range(1, n_episodes + 1):
        scores.append(train_dqn_episode(agent, train_env))

        if early_stopping and len(scores) >= cfg["rolling_window_size"]:
            current_rolling = float(np.mean(scores[-cfg["rolling_window_size"] :]))

            if current_rolling > (best_rolling_profit + cfg["improvement_threshold"]):
                best_rolling_profit = current_rolling
                patience_counter = 0
                if agent.epsilon <= cfg["min_epsilon_to_stop"]:
                    torch.save(
                        agent.policy_net.state_dict(),
                        f"checkpoints/best_{model_name.lower()}_iter{iteration:03d}.pth",
                    )
                    log_checkpoint(best_rolling_profit, ep)
            elif agent.epsilon <= cfg["min_epsilon_to_stop"]:
                patience_counter += 1

            if patience_counter >= cfg["patience"]:
                log_early_stop(ep, n_episodes, cfg["patience"], best_rolling_profit)
                break

        if ep == 1 or ep % log_interval == 0 or ep == n_episodes:
            avg_score = float(np.mean(scores[-log_interval:]))
            eta_str = time.strftime(
                "%H:%M:%S",
                time.gmtime(int((n_episodes - ep) * ((time.time() - start_time) / ep))),
            )
            log_progress(
                iteration,
                (ep / n_episodes) * 100,
                agent.epsilon,
                avg_score,
                eta_str,
                model_name=model_name,
            )

    return scores
