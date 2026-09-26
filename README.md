# ARP_RL

Reinforcement learning for airline aircraft-to-flight assignment and schedule disruption recovery,
benchmarked on the ROADEF2009 competition instances.

This repository contains:
- DQN and Double DQN agents (attention pooling over a flight lookahead window, prioritized replay,
  optional imitation "expert bias" toward a greedy heuristic)
- Random and greedy baselines
- Experiment pipelines for instance generalization and disruption recovery

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync                                            # install dependencies
uv sync --extra cplex                              # optional: exact MILP baseline (src/workspace/cplex.py)
uv run python -m src.experiments.single_model_experiment
```

## Which Script Should I Run?

All experiment scripts live in `src/experiments/`, build their agents/instances through
`src.experiments.experiment_setup` and `src.instances`, seed every RNG with `SEED` from
`src/config.py`, and write everything they produce to their own run folder (see *Run outputs*).

- `python -m src.experiments.single_model_experiment` — dev playground: one model, one ROADEF
  instance (optionally downsized), one training iteration. Edit the knobs at the top of the file.
- `python -m src.experiments.iter_training` — DQN + Double DQN, base config, full training
  instance, `N_ITERATIONS` iterations.
- `python -m src.experiments.solving_schedule_experiment` — fresh sampled instance every
  iteration, trains the active variants on each: generalization across instances.
- `python -m src.experiments.disruption_training_experiment` — train on a schedule, then
  repeatedly disrupt + retrain: disruption resilience.

## Run outputs

Every run writes to `runs/<timestamp>_<name>/` (gitignored):

- `config.json` — the script's own parameters, a snapshot of every `src/config.py` constant
  (including `SEED`), and the git commit + whether the working tree was dirty
- `run.log` — everything logged during the run
- `results.pkl` — the raw results (what the analysis scripts read)
- `metrics.json` — a short human-readable summary (where the experiment provides one)
- `checkpoints/` — model weights from this run

## Analysis

Plots and tables are separate from training, so results can be re-analysed without retraining:

```bash
python -m src.analysis.instance_sweep_plots --run runs/<run_id>          # one instance-sweep run
python -m src.analysis.instance_sweep_plots --compare                     # random/trap/sample comparison
python -m src.analysis.disruption_recovery_table                          # historical recovery table
python -m src.analysis.disruption_recovery_table Trap=runs/<run_id> ...   # any LABEL=run pairs
```

The `--compare` / default inputs are the historical pickles in `data/experiments/`.

## Problem Instances

`src/instances/` is the single source of truth for building the FLIGHTS/PLANES/AIRPORTS/
dist_dict inputs `AirlineEnv` expects:

- `roadef.py`: `discover_roadef_instances(project_root)` lists the 20 real ROADEF2009 instances in
  `data/` (A01-A10 at ~600 flights / 84 aircraft, B01-B10 at ~1300 flights / 251 aircraft), and
  `load_roadef_instance(path)` loads one.
- `common.py`: `subsample_instance` (downsize a loaded instance, seedable), plus
  `build_flight_pool` / `build_planes`, the converters from raw tables into env inputs.
- `synthetic.py`: `generate_random_flights` / `generate_trap_schedule` synthetic generators.

## Repository Layout

- `src/config.py` — hyperparameters, ablation variants (`AGENT_VARIANT_OVERRIDES`), disruption
  severity, reward settings, `SEED`, `RUNS_DIR`
- `src/agents/dqn_agent.py` — DQN + Double DQN, attention Q-network, numpy-backed prioritized replay buffer
- `src/instances/` — problem-instance loading (see above)
- `src/experiments/` — the entrypoints above, plus `experiment_setup.py` (shared variant building,
  training, evaluation) and `run_tracking.py` (run folders)
- `src/analysis/` — result plots and tables, reading saved results only
- `src/utils/`
	- `envs.py`: `AirlineEnv`, baseline solvers, evaluation runner
	- `training_engine.py`: agent initialization and the training loop
	- `disruptions.py`: disruption injection
	- `seeding.py`: `set_seed`
	- `dist.py`: geodesic airport distances; `data_prep.py`: raw ROADEF table readers
- `src/workspace/cplex.py` — exact CPLEX baseline (needs `uv sync --extra cplex`)

## Linting and Checks

```bash
pre-commit            # or:
uv run ruff check src
uv run ruff format src
```
