"""Per-run output folders, so every result can be traced back to exactly what produced it.

Each experiment run gets RUNS_DIR/<timestamp>_<name>/ containing:
- config.json   -- the run's own parameters, a snapshot of every src.config constant (incl. SEED),
                   and the git commit + whether the working tree had uncommitted changes
- run.log       -- a copy of everything logged during the run
- results.pkl   -- the raw results object the experiment produced (read by src/analysis/*)
- metrics.json  -- a small human-readable summary, if the experiment provides one
- checkpoints/  -- model weights saved during the run
"""

import json
import logging
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src import config
from src.experiments.experiment_setup import resolve_project_root
from src.utils.logging import logger


@dataclass
class Run:
    name: str
    run_dir: Path

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def save_results(self, results: Any, metrics: dict | None = None) -> None:
        with open(self.run_dir / "results.pkl", "wb") as f:
            pickle.dump(results, f)
        if metrics is not None:
            (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        logger.info(f"Saved results to {self.run_dir}")


def _git_state(project_root: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=project_root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def config_snapshot() -> dict:
    """Every UPPER_CASE constant in src.config, so new settings are captured without editing this."""
    return {name: getattr(config, name) for name in dir(config) if name.isupper()}


def start_run(name: str, params: dict) -> Run:
    """Creates the run folder, writes config.json, and mirrors the logger into run.log."""
    project_root = resolve_project_root()
    run_dir = project_root / config.RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    (run_dir / "checkpoints").mkdir(parents=True)

    payload = {
        "name": name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_state(project_root),
        "params": params,
        "config": config_snapshot(),
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(file_handler)

    logger.info(f"Run directory: {run_dir}")
    return Run(name=name, run_dir=run_dir)
