"""Single entry point for seeding every RNG the pipeline uses."""

import random

import numpy as np
import torch


def set_seed(seed: int | None) -> int:
    """Seeds Python's `random` (epsilon checks, synthetic schedules, disruptions, RandomSolver),
    numpy (replay sampling, pandas `.sample()`), and torch (network init, CPU and CUDA).

    seed=None draws a fresh seed, so every run gets genuinely new schedules; the seed actually
    used is returned either way, so it can be recorded and that exact run reproduced later.
    """
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
