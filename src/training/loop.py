"""
DQN training loop with early stopping logic.
Encapsulates the core training mechanics and checkpoint management.
"""

import time

import numpy as np
import torch

from src.config import (
    DQN_LOG_INTERVAL,
    DQN_REWARD_SCALE,
    EARLY_STOPPING_CONFIG,
)
from src.log_config import get_logger

logger = get_logger("plane_assignment")


class TrainingLogger:
    """Manages logging of training progress and metrics."""

    @staticmethod
    def log_iteration_start(iteration, total_iterations):
        """Log the start of a schedule regeneration iteration."""
        logger.info("\n" + "=" * 80)
        logger.info(
            f"STARTING SCHEDULE REGENERATION ITERATION {iteration}/{total_iterations}"
        )
        logger.info("=" * 80)

    @staticmethod
    def log_exploration_reset(epsilon, epsilon_decay):
        """Log epsilon and decay parameters."""
        logger.info(
            f"Exploration reset: Epsilon = {epsilon:.2f} | Step Decay = {epsilon_decay:.6f}"
        )

    @staticmethod
    def log_feasibility(iteration, utilization, max_req_planes, num_planes):
        """Log schedule feasibility analysis."""
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

    @staticmethod
    def log_progress(
        iteration, progress_pct, epsilon, avg_score, eta_str, model_name="MODEL"
    ):
        """Log periodic training progress with model context."""
        logger.info(
            f"[{model_name}] [Iter {iteration}] Progress: {progress_pct:>5.1f}% | "
            f"Epsilon: {epsilon:.4f} | "
            f"Avg Profit: ${avg_score:>10.0f} | "
            f"ETA: {eta_str}"
        )

    @staticmethod
    def log_checkpoint(best_rolling_profit, episode_num):
        """Log checkpoint save."""
        logger.info(
            f"[CHECKPOINT] Episode {episode_num}: New best rolling avg: ${best_rolling_profit:,.0f}"
        )

    @staticmethod
    def log_early_stop(episode_num, total_episodes, patience, best_rolling_profit):
        """Log early stopping event."""
        logger.info(
            f"\n[EARLY STOP] Terminated at Episode {episode_num}/{total_episodes}. "
            f"No improvement for {patience} episodes. "
            f"Final policy at rolling average: ${best_rolling_profit:,.0f}."
        )

    @staticmethod
    def log_model_saved(model_path):
        """Log model checkpoint save."""
        logger.info(f"Saved checkpoint to: {model_path}")


class EarlyStoppingManager:
    """Manages early stopping logic with rolling average tracking."""

    def __init__(self, config=None):
        """Initialize early stopping manager with configuration.

        Args:
            config: Dictionary with early stopping parameters. Uses defaults from
                    EARLY_STOPPING_CONFIG if not provided.
        """
        if config is None:
            config = EARLY_STOPPING_CONFIG

        self.patience = config["patience"]
        self.rolling_window_size = config["rolling_window_size"]
        self.improvement_threshold = config["improvement_threshold"]
        self.min_epsilon_to_stop = config["min_epsilon_to_stop"]

        self.best_rolling_profit = float("-inf")
        self.patience_counter = 0

    def should_save_checkpoint(self, epsilon):
        """Check if model should be saved based on epsilon threshold."""
        return epsilon <= self.min_epsilon_to_stop

    def should_stop(self):
        """Check if early stopping condition is met."""
        return self.patience_counter >= self.patience

    def update(self, episode_scores, epsilon):
        """Update early stopping state based on latest scores.

        Args:
            episode_scores: List of all episode scores so far
            epsilon: Current exploration parameter

        Returns:
            tuple: (should_stop, should_save_checkpoint)
        """
        if len(episode_scores) < self.rolling_window_size:
            return False, False

        current_rolling_profit = np.mean(episode_scores[-self.rolling_window_size :])

        # Use improvement threshold to reset patience
        if current_rolling_profit > (
            self.best_rolling_profit + self.improvement_threshold
        ):
            self.best_rolling_profit = current_rolling_profit
            self.patience_counter = 0
            should_save = self.should_save_checkpoint(epsilon)
            return False, should_save
        else:
            if self.should_save_checkpoint(epsilon):
                self.patience_counter += 1

        return self.should_stop(), False


def train_dqn_episode(agent, env):
    """Run one training episode on the environment.

    Args:
        agent: DQNAgent instance
        env: AirlineEnv instance

    Returns:
        float: Total episode reward
    """
    raw_state = env.reset()
    state_tuple = env.get_vector_state(raw_state)
    done = False
    episode_reward = 0

    while not done:
        action = agent.choose_action(state_tuple, use_epsilon=True)
        next_raw_state, reward, done, _ = env.step(action)
        next_state_tuple = env.get_vector_state(next_raw_state)

        scaled_reward = reward * DQN_REWARD_SCALE

        # Unpack tuple elements for storage
        fleet_state, flight_matrix = state_tuple
        next_fleet, next_flights = next_state_tuple

        agent.store_transition(
            fleet_state,
            flight_matrix,
            action,
            scaled_reward,
            next_fleet,
            next_flights,
            done,
        )
        agent.learn()

        state_tuple = next_state_tuple
        episode_reward += reward

    agent.decay_epsilon()
    return episode_reward


def train_dqn_iteration(
    agent,
    train_env,
    n_episodes,
    iteration,
    early_stopping=True,
    log_interval=None,
    model_name=None,
):
    """Run one full training iteration on the environment.

    Args:
        agent: DQNAgent instance
        train_env: AirlineEnv instance for training
        n_episodes: Number of episodes to train
        iteration: Current iteration number (for logging)
        early_stopping: Whether to apply early stopping
        log_interval: Optional logging interval for DQN-style agents
        model_name: Optional name used in log output

    Returns:
        list: Scores from all episodes
    """
    dqn_scores = []
    dqn_start_time = time.time()

    if log_interval is None:
        log_interval = DQN_LOG_INTERVAL

    logger.info(
        f"[{model_name or agent.__class__.__name__}] Starting training for {n_episodes} episodes"
    )

    early_stop_manager = EarlyStoppingManager() if early_stopping else None

    for i in range(1, n_episodes + 1):
        episode_reward = train_dqn_episode(agent, train_env)
        dqn_scores.append(episode_reward)

        # Check early stopping
        if early_stopping:
            assert early_stop_manager is not None
            should_stop, should_save = early_stop_manager.update(
                dqn_scores, agent.epsilon
            )

            if should_save:
                checkpoint_path = f"checkpoints/best_policy_iter{iteration:03d}.pth"
                torch.save(agent.policy_net.state_dict(), checkpoint_path)
                TrainingLogger.log_checkpoint(early_stop_manager.best_rolling_profit, i)

            if should_stop:
                TrainingLogger.log_early_stop(
                    i,
                    n_episodes,
                    early_stop_manager.patience,
                    early_stop_manager.best_rolling_profit,
                )
                break

        # Periodic logging
        if i == 1 or i % log_interval == 0 or i == n_episodes:
            avg_score_dqn = np.mean(dqn_scores[-log_interval:])
            pct_dqn = (i / n_episodes) * 100

            dqn_elapsed_time = time.time() - dqn_start_time
            avg_time_per_dqn_ep = dqn_elapsed_time / i
            dqn_remaining_sec = int((n_episodes - i) * avg_time_per_dqn_ep)
            dqn_eta_str = time.strftime("%H:%M:%S", time.gmtime(dqn_remaining_sec))

            TrainingLogger.log_progress(
                iteration,
                pct_dqn,
                agent.epsilon,
                avg_score_dqn,
                dqn_eta_str,
                model_name=model_name or agent.__class__.__name__,
            )

    return dqn_scores
