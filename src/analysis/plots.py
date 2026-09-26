"""Reusable plotting helpers for experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d


def plot_training_curve(episode_rewards: list[float], title: str, save_path: Path | None = None) -> None:
    """Per-episode reward (raw scatter + smoothed line); saved to save_path if given, then shown."""
    plt.figure(figsize=(10, 5))
    plt.scatter(
        range(len(episode_rewards)),
        episode_rewards,
        alpha=0.3,
        s=15,
        label="Episode reward",
    )
    if len(episode_rewards) > 1:
        smoothed = uniform_filter1d(episode_rewards, size=max(1, len(episode_rewards) // 20))
        plt.plot(smoothed, color="tab:orange", linewidth=2, label="Smoothed")
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    plt.show()
