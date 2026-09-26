"""Single entry point for seeding every RNG the pipeline uses."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds Python's `random` (epsilon checks, synthetic schedules, disruptions, RandomSolver),
    numpy (replay sampling, pandas `.sample()`), and torch (network init, CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
