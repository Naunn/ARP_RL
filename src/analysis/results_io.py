"""Loading saved experiment results, from either a run folder or a bare pickle file."""

import pickle
from pathlib import Path
from typing import Any


def load_results(path: str | Path) -> Any:
    """Accepts a run folder (reads its results.pkl) or a direct path to a results pickle."""
    path = Path(path)
    if path.is_dir():
        path = path / "results.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def sorted_iteration_keys(results: dict) -> list[str]:
    return sorted(results.keys(), key=int)
