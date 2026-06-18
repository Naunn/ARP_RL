# ARP_RL

Reinforcement learning project for airline plane assignment and schedule disruption handling.

This repository contains:
- Tabular Q-learning
- DQN
- Double DQN
- Iterative training/evaluation pipelines
- Baselines (random and greedy)

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

### 2. Install dependencies

From repository root:

```bash
uv sync
```

### 3. Run the main training pipeline

```bash
uv run python -m src.iter_training
```

This is the primary entrypoint for current training/evaluation flow.

## Environment Options

Use one of these approaches:

- Standard local `.venv` (recommended for most users):
```bash
uv sync
source .venv/bin/activate
```

- Shared environment helper (project-specific workflow):
```bash
./activate_env.sh
```

## Which Script Should I Run?

- `python -m src.iter_training`
	- Current modular training pipeline
	- Trains Q-learning, DQN, Double DQN
	- Saves checkpoints and runs final evaluation

- `python -m src.main`
	- Legacy/experimental script
	- Useful for quick prototyping
	- Not the canonical benchmark pipeline

## Repository Navigation Guide

### Top-level

- `pyproject.toml`: dependencies and project metadata
- `activate_env.sh`: helper for shared venv workflow
- `checkpoints/`: saved model weights
- `src/logs/`: training and evaluation logs

### Core code

- `src/config.py`
	- Central configuration for:
		- model hyperparameters
		- episode counts / logging cadence
		- reward scaling/clipping
		- fleet and schedule defaults

- `src/iter_training.py`
	- Main orchestration script
	- Builds environments/agents
	- Runs train -> checkpoint -> evaluation loop

- `src/agents/`
	- `dqn_agent.py`: DQN + Double DQN implementations (with replay buffer)
	- `q_learning_agent.py`: tabular Q-learning agent

- `src/training/`
	- `initialization.py`: setup helpers (agents, filenames, static data)
	- `loop.py`: training loops and early stopping
	- `evaluation.py`: iteration and final scoreboards

- `src/utils/`
	- `envs.py`: `AirlineEnv`, solver wrappers, execution runner
	- `schedule.py`: schedule generators and feasibility checks
	- `disruptions.py`: disruption injection utilities
	- `dist.py`, `fleet.py`: support utilities

## Typical Workflow for New Contributors

1. Read `src/config.py` first to understand experiment settings.
2. Run `python -m src.iter_training` and inspect `src/logs/plane_assignment.log`.
3. Modify one layer at a time:
	 - agent logic in `src/agents/`
	 - reward/state dynamics in `src/utils/envs.py`
	 - training behavior in `src/training/loop.py`
4. Re-run and compare scoreboard outputs.

## Output Artifacts

- Checkpoints: `checkpoints/*.pth`
- Runtime logs: `src/logs/plane_assignment.log`

## Linting and Checks

If dev tools are installed:

```bash
pre-commit
```

Or run targeted checks:

```bash
uv run ruff check src
uv run python -m py_compile src/iter_training.py src/training/*.py src/agents/*.py src/utils/*.py
```

## Notes

- Prefer `uv` commands to keep dependencies in sync.
- Treat `src.iter_training` as the source of truth for experiments.